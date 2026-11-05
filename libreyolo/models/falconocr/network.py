# SPDX-License-Identifier: Apache-2.0
# Adapted from Falcon OCR (Technology Innovation Institute, Apache-2.0):
# modeling_falcon_ocr.py, rope.py and attention.py of the Hugging Face release
# tiiuae/Falcon-OCR @ 0bc28f89c90fbe8bdb063097fab144d4ce4c19b5, the same code TII
# publishes in https://github.com/tiiuae/Falcon-Perception (falcon_perception/
# model.py, rope.py, attention.py, commit
# c457916c9974efbacfa91f0f6ecc2c49c6543e56, Copyright (c) 2025 TII). See NOTICE.
"""The Falcon OCR network: one early-fusion transformer over patches and text.

Image patches are projected by a single linear layer and scattered into the
token sequence; 22 decoder blocks then process both. Each block uses
weight-free RMS norms, grouped-query attention with per-head attention sinks,
3D rotary embeddings (the first half of each head rotates by the 1D token
position, the second half of image-patch heads by a learned 2D "golden"
frequency table) and a squared-ReLU gated MLP.

LibreYOLO changes: the FlexAttention block masks become a dense boolean mask
(image tokens attend bidirectionally inside their image, everything else is
causal), computed in float32 query chunks so the log-sum-exp needed by the
attention sink is exact on every device; the Triton squared-ReLU kernel becomes
plain PyTorch; ``torch.compile`` and the batched left-padded engine are
dropped (one request at a time).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Query rows per attention chunk: bounds the float32 score matrix to
# n_heads * chunk * seq_len (16 * 1024 * 9.5k * 4 B ~= 0.6 GB at 1536 px).
_ATTN_CHUNK = 1024


@dataclass
class FalconOCRConfig:
    dim: int = 768
    n_layers: int = 22
    n_heads: int = 16
    head_dim: int = 64
    n_kv_heads: int = 8
    vocab_size: int = 65536
    ffn_dim: int = 2304
    norm_eps: float = 1e-5
    max_seq_len: int = 16384
    rope_theta: float = 10000.0
    channel_size: int = 3
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    img_id: int = 227
    eos_id: int = 11
    image_cls_token_id: int = 244
    img_end_id: int = 230

    @classmethod
    def from_dict(cls, data: dict) -> "FalconOCRConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # (end, dim // 2) complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """1D interleaved rotary embedding; ``freqs_cis`` is ``(B, S, F)``."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_ * freqs_cis.unsqueeze(2)).flatten(3)
    return out.type_as(x)


def _full_precision(device: torch.device):
    """Autocast off: the rotary and attention-sink math must stay float32."""
    return torch.autocast(device_type=device.type, enabled=False)


def golden_freqs(freqs_hfp: torch.Tensor, pos_tp: torch.Tensor) -> torch.Tensor:
    """Angles of the 2D rotation for each image patch: ``(T, H, F)`` complex."""
    with _full_precision(pos_tp.device):
        theta = torch.einsum("tp,hfp->thf", pos_tp.float(), freqs_hfp.float())
        return torch.polar(torch.ones_like(theta), theta)


def apply_golden_rotary_emb(
    x: torch.Tensor, freqs_thf: torch.Tensor, idx_b: torch.Tensor, idx_s: torch.Tensor
) -> torch.Tensor:
    """Rotate only the image-patch rows ``(idx_b, idx_s)`` of ``x`` ``(B, S, H, D)``."""
    sel = x[idx_b, idx_s].float()
    x_even, x_odd = sel[..., 0::2], sel[..., 1::2]
    cos, sin = freqs_thf.real, freqs_thf.imag
    rotated = torch.empty_like(sel)
    rotated[..., 0::2] = x_even * cos - x_odd * sin
    rotated[..., 1::2] = x_even * sin + x_odd * cos
    out = x.clone()
    out[idx_b, idx_s] = rotated.type_as(x)
    return out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)


class SequenceMask:
    """The release's attention rule for one sequence, evaluated densely.

    ``allowed(q, k) = (img[q] & img[k] & same image) |
    (k <= q & same document & k is not left padding)``. ``img`` spans
    ``<|image_cls|>`` up to the last patch (``<|end_of_image|>`` is causal);
    documents are split after each ``eos``. Built once from the prompt;
    generated tokens are never image tokens, so it extends by causality.
    """

    def __init__(self, ids: torch.Tensor, cfg: FalconOCRConfig):
        soi = torch.cumsum(ids == cfg.image_cls_token_id, dim=0)
        eoi = torch.cumsum(ids == cfg.img_end_id, dim=0)
        self.img = (soi - eoi) > 0
        self.img_index = soi * self.img
        eos = ids == cfg.eos_id
        doc = torch.zeros_like(ids)
        doc[1:] = torch.cumsum(eos.long(), dim=0)[:-1]
        self.doc = doc
        self.length = ids.numel()

    def dense(self, q_start: int, q_end: int, kv_len: int) -> torch.Tensor:
        """Boolean ``(q_end - q_start, kv_len)`` block (keys past the prompt are text)."""
        device = self.img.device
        q = torch.arange(q_start, q_end, device=device)
        k = torch.arange(kv_len, device=device)

        def ext(t: torch.Tensor, idx: torch.Tensor, fill) -> torch.Tensor:
            out = torch.full(idx.shape, fill, dtype=t.dtype, device=device)
            inside = idx < self.length
            out[inside] = t[idx[inside]]
            return out

        img_q, img_k = ext(self.img, q, False), ext(self.img, k, False)
        ii_q, ii_k = ext(self.img_index, q, 0), ext(self.img_index, k, 0)
        last_doc = self.doc[-1]
        doc_q = ext(self.doc, q, 0)
        doc_k = ext(self.doc, k, 0)
        doc_q[q >= self.length] = last_doc
        doc_k[k >= self.length] = last_doc
        image = img_q[:, None] & img_k[None, :] & (ii_q[:, None] == ii_k[None, :])
        causal = (k[None, :] <= q[:, None]) & (doc_q[:, None] == doc_k[None, :])
        return image | causal


def sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: SequenceMask,
    q_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked attention returning output ``(B, H, S, D)`` and natural-log LSE ``(B, H, S)``.

    Float32 scores and softmax, scale ``1/sqrt(head_dim)`` as FlexAttention's
    default. Rows with no allowed key give a zero output and ``-inf`` LSE.
    """
    with _full_precision(q.device):
        scale = q.shape[-1] ** -0.5
        acc = torch.promote_types(q.dtype, torch.float32)
        kf, vf = k.to(acc), v.to(acc)
        outs, lses = [], []
        s = q.shape[2]
        for start in range(0, s, _ATTN_CHUNK):
            stop = min(start + _ATTN_CHUNK, s)
            scores = (
                torch.matmul(q[:, :, start:stop].to(acc), kf.transpose(-1, -2)) * scale
            )
            allowed = mask.dense(q_offset + start, q_offset + stop, k.shape[2])
            scores = scores.masked_fill(~allowed, float("-inf"))
            lse = torch.logsumexp(scores, dim=-1)
            probs = torch.exp(scores - lse.unsqueeze(-1).nan_to_num(neginf=0.0))
            out = torch.matmul(probs.nan_to_num(0.0), vf)
            outs.append(out)
            lses.append(lse)
        return torch.cat(outs, dim=2).to(q.dtype), torch.cat(lses, dim=2)


class KVCache:
    """Pre-allocated per-layer key/value store for one sequence."""

    def __init__(self, n_layers: int, capacity: int):
        self.n_layers = n_layers
        self.capacity = capacity
        self.k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.v: List[Optional[torch.Tensor]] = [None] * n_layers
        self.pos = 0

    def insert(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        b, h, t, d = k.shape
        if self.k[layer] is None:
            shape = (b, h, self.capacity, d)
            self.k[layer] = torch.empty(shape, dtype=k.dtype, device=k.device)
            self.v[layer] = torch.empty(shape, dtype=v.dtype, device=v.device)
        t0, t1 = self.pos, self.pos + t
        self.k[layer][:, :, t0:t1] = k
        self.v[layer][:, :, t0:t1] = v
        if layer == self.n_layers - 1:
            self.pos = t1
        return self.k[layer][:, :, :t1], self.v[layer][:, :, :t1]


class Attention(nn.Module):
    def __init__(self, cfg: FalconOCRConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.n_rep = cfg.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim or cfg.dim // cfg.n_heads
        self.q_dim = cfg.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.wqkv = nn.Linear(cfg.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.sinks = nn.Parameter(torch.empty((cfg.n_heads,)))

    def forward(self, x, freqs_cis, golden, mask, cache, q_offset):
        b, s, _ = x.shape
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        xq = xq.reshape(b, s, -1, self.head_dim)
        xk = xk.reshape(b, s, -1, self.head_dim)
        xv = xv.reshape(b, s, -1, self.head_dim)
        xq = F.rms_norm(xq, (xq.size(-1),))
        xk = F.rms_norm(xk, (xk.size(-1),))
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 3D rotary: 1D token position on the first half of each head, the 2D
        # golden rotation on the second half of image-patch rows only.
        xq_t, xq_hw = xq.chunk(2, dim=-1)
        xk_t, xk_hw = xk.chunk(2, dim=-1)
        xq_t = apply_rotary_emb(xq_t, freqs_cis)
        xk_t = apply_rotary_emb(xk_t, freqs_cis)
        if golden is not None:
            freqs_thf, idx_b, idx_s = golden
            xq_hw = apply_golden_rotary_emb(xq_hw, freqs_thf, idx_b, idx_s)
            xk_hw = apply_golden_rotary_emb(xk_hw, freqs_thf, idx_b, idx_s)
        xq = torch.cat([xq_t, xq_hw], dim=-1).type_as(x)
        xk = torch.cat([xk_t, xk_hw], dim=-1).type_as(x)

        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))
        keys, values = cache.insert(self.layer_id, xk, xv)
        out, lse = sink_attention(xq, keys, values, mask, q_offset)
        # Sink: sigmoid(lse - sink) * out, i.e. one extra key with logit = sink.
        scale = torch.sigmoid(lse - self.sinks.to(lse.dtype).view(1, -1, 1))
        out = (out * scale.unsqueeze(-1)).to(out.dtype)
        out = out.permute(0, 2, 1, 3).contiguous().flatten(2)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.w13(F.rms_norm(x, (x.size(-1),)))
        # Interleaved [gate, up, gate, up, ...]: relu(gate)^2 * up.
        gate, up = packed[..., 0::2], packed[..., 1::2]
        gate = torch.where(gate > 0, gate, torch.zeros_like(gate))
        return self.w2(gate * gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: FalconOCRConfig):
        super().__init__()
        self.attention = Attention(cfg, layer_id)
        self.feed_forward = FeedForward(cfg.dim, cfg.ffn_dim)

    def forward(self, x, freqs_cis, golden, mask, cache, q_offset):
        x = x + self.attention(x, freqs_cis, golden, mask, cache, q_offset)
        return x + self.feed_forward(x)


class FalconOCRNetwork(nn.Module):
    """Checkpoint layout of ``model.safetensors`` (no prefix)."""

    def __init__(self, cfg: FalconOCRConfig):
        super().__init__()
        self.config = cfg
        img_in = cfg.temporal_patch_size * cfg.spatial_patch_size**2 * cfg.channel_size
        self.img_projector = nn.Linear(img_in, cfg.dim, bias=False)
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(i, cfg) for i in range(cfg.n_layers)}
        )
        self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        rope_dim = cfg.head_dim // 2
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(rope_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )
        self.register_buffer(
            "freqs_cis_golden",
            torch.empty((cfg.n_heads, rope_dim // 2, 2), dtype=torch.float),
            persistent=True,
        )

    def reset_rope_cache(self) -> None:
        """Rebuild the non-persistent 1D table (after meta-device loading)."""
        cfg = self.config
        device = self.tok_embeddings.weight.device
        self.freqs_cis = precompute_freqs_cis(
            cfg.head_dim // 2, cfg.max_seq_len, cfg.rope_theta
        ).to(device)

    def _run(self, h, pos_t, golden, mask, cache, q_offset):
        freqs_cis = self.freqs_cis[pos_t]
        for layer in self.layers.values():
            h = layer(h, freqs_cis, golden, mask, cache, q_offset)
        return self.output(self.norm(h))

    def prefill(self, ids, pos_t, pos_hw, patches, mask, cache) -> torch.Tensor:
        """Logits for the whole prompt ``(1, S, V)``; fills ``cache``."""
        h = self.tok_embeddings(ids)
        dtype = h.dtype
        feats = self.img_projector(patches.to(dtype))
        img_rows = (ids == self.config.img_id).unsqueeze(-1).expand_as(h)
        if feats.numel() != int(img_rows.sum()):
            raise ValueError("patch count does not match the <|image|> tokens")
        h = torch.masked_scatter(h, img_rows, feats)
        on_image = ~torch.isnan(pos_hw).any(dim=-1)
        idx_b, idx_s = torch.nonzero(on_image, as_tuple=True)
        freqs_thf = golden_freqs(self.freqs_cis_golden, pos_hw[idx_b, idx_s])
        return self._run(h, pos_t, (freqs_thf, idx_b, idx_s), mask, cache, 0)

    def decode_step(self, token, pos_t, mask, cache) -> torch.Tensor:
        h = self.tok_embeddings(token)
        return self._run(h, pos_t, None, mask, cache, cache.pos)

    @torch.inference_mode()
    def generate(
        self,
        ids: torch.Tensor,
        pos_t: torch.Tensor,
        pos_hw: torch.Tensor,
        patches: torch.Tensor,
        *,
        max_new_tokens: int,
        stop_ids: Iterable[int],
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> List[int]:
        """Greedy decode (the release default, temperature 0); returns new ids.

        Every token after the prompt has the same mask row as in the release:
        causal over everything before it.

        The stop token, when reached, is the last id returned, as upstream.
        """
        cfg = self.config
        ids = ids.view(1, -1)
        length = ids.shape[1]
        block = 128
        capacity = (length + max_new_tokens + block - 1) // block * block
        if capacity > cfg.max_seq_len:
            raise ValueError(
                f"prompt ({length} tokens) plus max_new_tokens ({max_new_tokens}) "
                f"exceeds the model's {cfg.max_seq_len}-token context"
            )
        mask = SequenceMask(ids[0], cfg)
        cache = KVCache(cfg.n_layers, capacity)
        stops = set(int(s) for s in stop_ids)
        device_type = ids.device.type
        ctx = (
            torch.autocast(device_type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device_type, enabled=False)
        )
        generated: List[int] = []
        with ctx:
            logits = self.prefill(
                ids, pos_t.view(1, -1), pos_hw.view(1, -1, 2), patches, mask, cache
            )
            t = pos_t.view(1, -1)[:, -1:]
            # As upstream, the budget is the 128-rounded capacity, so up to
            # 127 tokens past ``max_new_tokens`` can be generated.
            while cache.pos < capacity:
                nxt = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
                token = int(nxt.item())
                generated.append(token)
                if token in stops:
                    break
                t = t + 1
                logits = self.decode_step(nxt, t, mask, cache)
        return generated


__all__ = [
    "FalconOCRConfig",
    "FalconOCRNetwork",
    "KVCache",
    "SequenceMask",
    "sink_attention",
]
