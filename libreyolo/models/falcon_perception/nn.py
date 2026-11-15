# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from https://github.com/tiiuae/Falcon-Perception (Apache-2.0),
# commit c457916c9974efbacfa91f0f6ecc2c49c6543e56:
#   falcon_perception/{model,anyup,rope,attention,kv_cache,__init__}.py
# The AnyUp section (ResBlock through AnyUp) descends from
# https://github.com/wimmerth/anyup, licensed CC-BY-4.0 (Wimmer et al.);
# upstream's anyup.py is a flattened copy. Changes here: eager chunked
# window attention instead of FlexAttention. See
# libreyolo/models/falcon_perception/NOTICE.
#
# LibreYOLO adaptations: inference only, no triton / FlexAttention / paged
# engine / torch.compile. Attention runs through an eager masked-softmax that
# mirrors the operation order of torch's FlexAttention reference, so the
# result matches the upstream engine run in eager mode. Module and parameter
# names are unchanged so the released safetensors load strictly.
"""Falcon Perception network: early-fusion transformer plus coord/size/seg heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T


# =============================================================================
# Config
# =============================================================================


@dataclass
class ModelArgs:
    max_seq_len: int = 8192
    rope_theta: int = 10000
    # base
    dim: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    head_dim: int = 128
    n_kv_heads: int = 8
    vocab_size: int = 65536
    ffn_dim: int = 3072
    norm_eps: float = 1e-5
    # vision
    channel_size: int = 3
    spatial_patch_size: int = 16
    temporal_patch_size: int = 1
    # heads
    coord_enc_dim: int = 512
    coord_dec_dim: int = 8192
    coord_out_dim: int = 2048
    size_enc_dim: int = 512
    size_dec_dim: int = 8192
    size_out_dim: int = 2048
    perception_heads: bool = True
    do_segmentation: bool = True
    segm_out_dim: int = 256
    num_segm_layers: int = 3
    # token ids the network reads, filled from the tokenizer by ``update``
    img_id: int = 0
    coord_token_id: int = 0
    size_token_id: int = 0

    def update(self, tokenizer) -> None:
        self.img_id = tokenizer.image_token_id
        self.coord_token_id = tokenizer.coord_token_id
        self.size_token_id = tokenizer.size_token_id


def get_model_args(size: str) -> ModelArgs:
    """Architecture hyperparameters per released size."""
    if size == "600m":
        return ModelArgs(perception_heads=True)
    if size == "300m":
        return ModelArgs(
            n_layers=22,
            head_dim=64,
            dim=768,
            ffn_dim=2304,
            perception_heads=True,
            do_segmentation=False,
        )
    raise ValueError(f"Unknown Falcon Perception size {size!r}")


# =============================================================================
# RoPE (1D temporal on half the head dim, 2D "golden" spatial on the other half)
# =============================================================================


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> T:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64, [S, D//2]


def apply_rotary_emb(xq: T, xk: T, freqs_cis: T) -> Tuple[T, T]:
    """1D rotary embedding."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_golden_freqs_cis_to_visual_pos(freqs_hFP: T, pos_BSP: T) -> T:
    """2D golden-gate RoPE frequencies per token; text tokens (pos 0) are identity."""
    theta_BShF = torch.einsum("bsp,hfp->bshf", pos_BSP.float(), freqs_hFP.float())
    return torch.polar(torch.ones_like(theta_BShF), theta_BShF)  # (B, S, H, F)


def apply_golden_rotary_emb(input_BShd: T, freqs_cis_BShF: T) -> T:
    x = input_BShd.float()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = freqs_cis_BShF.real
    sin = freqs_cis_BShF.imag
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.type_as(input_BShd)


def apply_3d_rotary_emb(
    xq: T, xk: T, freqs_cis: T, freqs_cis_2d: Optional[T]
) -> Tuple[T, T]:
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    xq_t, xk_t = apply_rotary_emb(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None:
        xq_hw = apply_golden_rotary_emb(xq_hw, freqs_cis_2d)
        xk_hw = apply_golden_rotary_emb(xk_hw, freqs_cis_2d)

    xq_out = torch.concat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.concat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out


# =============================================================================
# Masked attention (mirrors torch's FlexAttention eager reference)
# =============================================================================


def masked_attention(
    q: T, k: T, v: T, mask: T, return_lse: bool = False
) -> Tuple[T, Optional[T]]:
    """Scaled dot-product attention with a boolean keep-mask.

    Same operation order as ``torch._higher_order_ops.flex_attention.math_attention``:
    fp32 scores, additive -inf mask, ``_safe_softmax`` and a natural-log LSE.
    ``q``/``k``/``v`` are ``(B, H, S, D)``; ``mask`` broadcasts to ``(B, H, Sq, Skv)``.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = (q @ k.transpose(-2, -1)).to(torch.float32)
    scores = scores * scale
    scores = torch.where(mask, scores, torch.full_like(scores, -float("inf")))
    lse = None
    if return_lse:
        lse = scores.logsumexp(dim=-1)
        masked_rows = torch.all(scores == -float("inf"), dim=-1)
        lse = torch.where(masked_rows, torch.full_like(lse, -float("inf")), lse)
        # FlexAttention computes the LSE in log2 space and rescales it on the
        # way out; the round trip rounds differently, so mirror it exactly.
        lse = (lse / math.log(2)) * math.log(2)
    probs = torch._safe_softmax(scores, dim=-1)
    out = probs.to(q.dtype) @ v.to(q.dtype)
    return out, lse


# =============================================================================
# Heads
# =============================================================================


class FourierEncoder(nn.Module):
    """Random-Fourier-feature encoder (https://bmild.github.io/fourfeat/)."""

    def __init__(self, in_dim: int, feat_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Linear(in_dim, feat_dim // 2, bias=False)
        self.transform = nn.Linear(feat_dim, out_dim, bias=False)

    def forward(self, x):
        f = 2 * math.pi * self.embed(x)
        f = torch.cat([f.cos(), f.sin()], dim=-1)
        return self.transform(f)


class BboxDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: T) -> T:
        return self.w2(F.relu(self.w1(x)).square())


class SegmDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, in_dim) for _ in range(num_layers - 1)]
        )
        self.pixel_layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x) -> T:
        for layer in self.layers:
            x = F.relu(layer(x)).square()
        return self.pixel_layer(x)


# =============================================================================
# Transformer
# =============================================================================


def repeat_kv(x: T, n_rep: int) -> T:
    """``torch.repeat_interleave(x, dim=2, repeats=n_rep)`` for (B, S, H, D)."""
    B, S, H, D = x.shape
    if n_rep == 1:
        return x
    return torch.unsqueeze(x, dim=3).expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class KVCache:
    """Batch KV cache; keys/values are stored after GQA expansion."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        num_layers: int,
    ):
        self.kv_shape = (num_layers, 2, max_batch_size, n_heads, max_seq_length, head_dim)
        self.kv_cache: Optional[T] = None
        self.pos = 0
        self.pos_t: Optional[T] = None

    def get_pos(self) -> int:
        return self.pos

    def increment_and_get_pos_t(self) -> T:
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        self.pos_t += 1
        return self.pos_t

    def insert_kv(self, layer_id: int, k: T, v: T) -> Tuple[T, T]:
        assert self.pos_t is not None, "pos_t for rope is not initialized."
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        t0, t1 = self.pos, self.pos + k.size(2)
        self.kv_cache[layer_id, 0, :, :, t0:t1] = k
        self.kv_cache[layer_id, 1, :, :, t0:t1] = v
        key_view = self.kv_cache[layer_id, 0, :, :, :t1]
        value_view = self.kv_cache[layer_id, 1, :, :, :t1]
        if layer_id == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.q_dim = args.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wqkv = nn.Linear(args.dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.sinks = nn.Parameter(torch.empty((args.n_heads,)))

    def _pre_attention(self, x, freqs_cis, freqs_cis_2d) -> Tuple[T, T, T]:
        qkv = self.wqkv(F.rms_norm(x, (x.size(-1),)))
        xq, xk, xv = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        xq = xq.unflatten(-1, (-1, self.head_dim))
        xk = xk.unflatten(-1, (-1, self.head_dim))
        xv = xv.unflatten(-1, (-1, self.head_dim))
        xq = F.rms_norm(xq, (xq.size(-1),))
        xk = F.rms_norm(xk, (xk.size(-1),))
        xk = repeat_kv(xk, n_rep=self.n_rep)
        xv = repeat_kv(xv, n_rep=self.n_rep)
        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d)
        return xq.permute(0, 2, 1, 3), xk.permute(0, 2, 1, 3), xv.permute(0, 2, 1, 3)

    def _post_attention(self, output: T, lse: T) -> T:
        # Attention sink: rescale by sigmoid(lse - sink) per head and query.
        sinks_BHS = self.sinks.view(1, -1, 1)
        sink_scale = torch.sigmoid(lse - sinks_BHS)
        output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)
        output = output.permute(0, 2, 1, 3).contiguous().flatten(2)
        return self.wo(output)

    def forward(self, x, mask, kv_cache: KVCache, freqs_cis, freqs_cis_2d=None):
        xq, xk, xv = self._pre_attention(x, freqs_cis, freqs_cis_2d)
        xk, xv = kv_cache.insert_kv(self.layer_id, xk, xv)
        output, lse = masked_attention(xq, xk, xv, mask, return_lse=True)
        return self._post_attention(output, lse)


class FeedForward(nn.Module):
    """Squared-ReLU gated FFN with a fused, interleaved gate+up projection.

    ``w13`` stores gate and up weights interleaved as ``[g0, u0, g1, u1, ...]``;
    the gate weights carry a pre-baked ``sqrt(2)`` factor.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x: T) -> T:
        x = F.rms_norm(x, (x.size(-1),))
        packed = self.w13(x)
        gate = packed[..., 0::2]
        up = packed[..., 1::2]
        gate = torch.where(gate > 0, gate, torch.zeros_like(gate))
        return self.w2(gate * gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.attention = Attention(model_args, layer_id)
        self.feed_forward = FeedForward(model_args.dim, model_args.ffn_dim)

    def forward(self, x, kv_cache, freqs_cis, freqs_cis_2d=None, mask=None):
        B, S, D = x.shape
        x = x + self.attention(
            x, mask=mask, kv_cache=kv_cache, freqs_cis=freqs_cis, freqs_cis_2d=freqs_cis_2d
        )
        out = x + self.feed_forward(x)
        return out.reshape(B, S, D)


# =============================================================================
# AnyUp: feature upsampler used by the segmentation head
# =============================================================================


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=p, padding_mode=pad_mode, bias=False,
            ),
            N(out_channels),
            activation_fn(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size,
                padding=p, padding_mode=pad_mode, bias=False,
            ),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.basis = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))

    def forward(self, features: T) -> T:
        b, c, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(
            b, self.out_channels, c, h, w
        )
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p), value=0)
        x = F.conv2d(x, basis.repeat(c, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv2d(
            F.pad(mask, (p, p, p, p), value=0),
            torch.ones(1, 1, k, k, device=x.device, dtype=x.dtype),
        )
        return x / denom


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class AnyUpRoPE(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def forward(self, x: T, coords: T) -> T:
        angle = coords @ self.freqs
        return x * angle.cos() + _rotate_half(x) * angle.sin()


def window2d(
    low_res: Tuple[int, int], high_res: Tuple[int, int], ratio: float, *, device="cpu"
) -> T:
    """Per-pixel [row0, row1, col0, col1) key window for a high-res query grid."""
    H, W = high_res
    Lh, Lw = low_res
    r_pos = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    c_pos = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    pos_r, pos_c = torch.meshgrid(r_pos, c_pos, indexing="ij")

    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    r0 = (r_lo * Lh).floor().long()
    r1 = (r_hi * Lh).ceil().long()
    c0 = (c_lo * Lw).floor().long()
    c1 = (c_hi * Lw).ceil().long()
    return torch.stack([r0, r1, c0, c1], dim=2)


def window_keep_mask(q_start: int, q_end: int, W: int, h: int, w: int, windows: T) -> T:
    """Boolean keep-mask (q_end - q_start, h * w) for flat query indices [q_start, q_end)."""
    q_idx = torch.arange(q_start, q_end, device=windows.device)
    q_r = q_idx // W
    q_c = q_idx % W
    kv_idx = torch.arange(h * w, device=windows.device)
    kv_r = (kv_idx // w)[None, :]
    kv_c = (kv_idx % w)[None, :]
    r0 = windows[q_r, q_c, 0][:, None]
    r1 = windows[q_r, q_c, 1][:, None]
    c0 = windows[q_r, q_c, 2][:, None]
    c1 = windows[q_r, q_c, 3][:, None]
    return (kv_r >= r0) & (kv_r < r1) & (kv_c >= c0) & (kv_c < c1)


class AttentionWrapper(nn.Module):
    def __init__(self, qk_dim: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.empty([qk_dim * 3, qk_dim]))
        self.in_proj_bias = nn.Parameter(torch.empty([qk_dim * 3]))

    def forward(self, x_q, x_k, x_v):
        w_q, w_k, _ = self.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, _ = self.in_proj_bias.chunk(3)
        x_q = x_q @ w_q.T + b_q
        x_k = x_k @ w_k.T + b_k
        return x_q, x_k, x_v


class FlexCrossAttention(nn.Module):
    # Queries are processed in chunks so the (HW x hw) score matrix of a large
    # canvas never has to exist at once; softmax is per query row, so chunking
    # does not change the maths.
    Q_CHUNK = 4096

    def __init__(self, qk_dim: int, num_heads: int, **kwargs):
        super().__init__()
        self.dim = qk_dim
        self.num_head = num_heads
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.attention = AttentionWrapper(qk_dim)

    def forward(self, query, key, value, window_info):
        x_q = self.norm_q(query)
        x_k = self.norm_k(key)
        x_q, x_k, x_v = self.attention(x_q, x_k, value)
        x_q = x_q.unflatten(-1, (self.num_head, -1)).transpose(1, 2)
        x_k = x_k.unflatten(-1, (self.num_head, -1)).transpose(1, 2)
        x_v = value.unflatten(-1, (self.num_head, -1)).transpose(1, 2)

        H, W, h, w, windows = window_info
        outs = []
        for q0 in range(0, H * W, self.Q_CHUNK):
            q1 = min(q0 + self.Q_CHUNK, H * W)
            keep = window_keep_mask(q0, q1, W, h, w, windows)
            out, _ = masked_attention(
                x_q[:, :, q0:q1], x_k, x_v, keep[None, None], return_lse=False
            )
            outs.append(out)
        output = torch.cat(outs, dim=2)
        return output.transpose(1, 2).flatten(2)


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1, **kwargs):
        super().__init__()
        self.cross_attn = FlexCrossAttention(qk_dim, num_heads)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, window_info):
        b, _, h, w = q.shape
        q = self.conv2d(q)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        features = self.cross_attn(q, k, v, window_info)
        return features.transpose(1, 2).unflatten(2, (h, w))


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def _pool_to(x: T, size: Tuple[int, int]) -> T:
    """Exact area pooling by reshape + mean (spatial dims divisible by ``size``)."""
    b, c, H, W = x.shape
    oh, ow = size
    if H == oh and W == ow:
        return x
    return x.reshape(b, c, oh, H // oh, ow, W // ow).mean(dim=(3, 5))


def create_coordinate(h, w, start=0.0, end=1.0, device=None, dtype=None):
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy), -1).view(1, h * w, 2)


class AnyUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(
            kernel_size=1,
            num_groups=8,
            pad_mode="reflect",
            norm_fn=nn.GroupNorm,
            activation_fn=nn.SiLU,
        )
        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None, 1, first_layer_k=kernel_size_lfu,
            init_gaussian_derivatives=init_gaussian_derivatives,
        )
        self.cross_decode = CrossAttentionBlock(
            qk_dim=qk_dim, num_heads=num_heads, window_ratio=window_ratio
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)
        self.rope = AnyUpRoPE(qk_dim)

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2d(in_ch, self.qk_dim, k, padding=k // 2, padding_mode="reflect", bias=False)
            if first_layer_k == 0
            else LearnedFeatureUnification(
                self.qk_dim, first_layer_k, init_gaussian_derivatives=init_gaussian_derivatives
            )
        )
        blocks = [ResBlock(self.qk_dim, self.qk_dim, **self._rb_args) for _ in range(layers)]
        return nn.Sequential(pre, *blocks)

    def upsample(self, enc_img, feats, window_info, out_size):
        b, c, h, w = feats.shape
        q = _pool_to(self.query_encoder(enc_img), out_size)
        k = _pool_to(self.key_encoder(enc_img), (h, w))
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)
        return self.cross_decode(q, k, feats, window_info)

    def forward(self, images, features, output_size=None):
        output_size = output_size if output_size is not None else images.shape[-2:]
        images = images * 0.5 + 0.5
        images = (images - IMAGENET_MEAN.to(images)) / IMAGENET_STD.to(images)
        images = images.to(features)
        enc = self.image_encoder(images)
        h = enc.shape[-2]
        coords = create_coordinate(h, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], h, -1, enc.shape[-1]).permute(0, 3, 1, 2)

        out_H, out_W = output_size
        fh, fw = features.shape[-2:]
        windows = window2d((fh, fw), (out_H, out_W), self.window_ratio, device=enc.device)
        window_info = (out_H, out_W, fh, fw, windows)
        return self.upsample(enc, features, window_info, output_size)


# =============================================================================
# Full model
# =============================================================================


class FalconPerception(nn.Module):
    """Early-fusion image+text transformer with coord / size / segmentation heads."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        img_in_dim = args.temporal_patch_size * args.spatial_patch_size**2 * args.channel_size
        self.img_projector = nn.Linear(img_in_dim, args.dim, bias=False)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, args)
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        if args.perception_heads:
            self.coord_encoder = FourierEncoder(2, args.coord_enc_dim, args.dim)
            self.coord_decoder = BboxDecoder(args.dim, args.coord_dec_dim, args.coord_out_dim)
            self.size_encoder = FourierEncoder(2, args.size_enc_dim, args.dim)
            self.size_decoder = BboxDecoder(args.dim, args.size_dec_dim, args.size_out_dim)
            if args.do_segmentation:
                self.itok_upsampler = AnyUp()
                self.proj_segm = SegmDecoder(args.dim, args.segm_out_dim, args.num_segm_layers)
                self.conv_segm = nn.Conv2d(args.dim, args.segm_out_dim, kernel_size=3, padding=1)

        rope_dim = self.args.head_dim // 2
        freqs_cis = precompute_freqs_cis(rope_dim, args.max_seq_len, args.rope_theta)
        freqs_cis_golden = torch.empty((self.args.n_heads, rope_dim // 2, 2), dtype=torch.float)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.register_buffer("freqs_cis_golden", freqs_cis_golden, persistent=True)

    @property
    def device(self):
        return self.tok_embeddings.weight.device

    @property
    def dtype(self):
        return self.tok_embeddings.weight.dtype

    def to(self, *args, **kwargs):
        # RoPE buffers are complex64 / fp32 tables; casting them to a half
        # dtype is unsupported or lossy. Detach, cast the rest, re-register.
        freqs_cis = self.freqs_cis
        freqs_cis_golden = self.freqs_cis_golden
        del self.freqs_cis
        del self.freqs_cis_golden
        super().to(*args, **kwargs)
        device = self.tok_embeddings.weight.device
        self.register_buffer("freqs_cis", freqs_cis.to(device=device), persistent=False)
        self.register_buffer(
            "freqs_cis_golden", freqs_cis_golden.to(device=device), persistent=True
        )
        return self

    # -- embedding scatter / encode ------------------------------------------------

    def scatter_img_tokens(self, h_BSD, pixel_values_NTHWC, img_scatter_info):
        """Replace image-placeholder embeddings with projected patch features."""
        pt = self.args.temporal_patch_size
        ps = self.args.spatial_patch_size

        valid_parts = []
        for img, (_, _, _, h_v, w_v) in zip(pixel_values_NTHWC, img_scatter_info):
            img = img.to(self.dtype)
            T_i, H_i, W_i, C_i = img.shape
            # (T, H, W, C) -> one row per (t, h, w) patch, features ordered (pt, ph, pw, c)
            patches = (
                img.reshape(T_i // pt, pt, H_i // ps, ps, W_i // ps, ps, C_i)
                .permute(0, 2, 4, 1, 3, 5, 6)
                .reshape(-1, pt * ps * ps * C_i)
            )
            grid = patches.reshape(T_i // pt, H_i // ps, W_i // ps, patches.shape[-1])
            valid_parts.append(grid[:, :h_v, :w_v, :].reshape(-1, patches.shape[-1]))
        valid_feats = self.img_projector(torch.cat(valid_parts, dim=0))

        offset = 0
        for b, tok_start, n_tok, _, _ in img_scatter_info:
            h_BSD[b, tok_start : tok_start + n_tok, :] = valid_feats[offset : offset + n_tok]
            offset += n_tok
        return h_BSD

    def _encode_special(self, h_BSD, tokens_BS, values, token_id, encoder):
        mask = tokens_BS == token_id
        if values.numel() == 0:
            return h_BSD
        enc = encoder(values.reshape(-1, 2))
        if enc.shape[0] == h_BSD.shape[0]:
            return torch.where(
                mask.unsqueeze(-1), enc.view(h_BSD.shape[0], -1, h_BSD.shape[-1]), h_BSD
            )
        return h_BSD.masked_scatter_(mask.unsqueeze(-1), enc)

    @staticmethod
    def process_sizes(logits: T) -> T:
        # log2-space size bins -> normalized size in (1/num_bins, 1]
        num_bins = logits.shape[-1]
        pred = torch.argmax(logits, dim=-1).float() / (num_bins - 1)
        min_size = torch.log2(torch.tensor(1 / num_bins))
        max_size = 0.0
        pred = pred * (max_size - min_size) + min_size
        return torch.pow(2.0, pred)

    # -- segmentation features -----------------------------------------------------

    def upsample_single_img_features(
        self,
        h_SD: T,
        pixel_values_THWC: T,
        img_token_start: int,
        h_valid: int,
        w_valid: int,
        output_size: Optional[Tuple[int, int]] = None,
        shrink_image: bool = True,
    ) -> T:
        """High-res ``(D, H, W)`` features for one image from its hidden states."""
        ps = self.args.spatial_patch_size
        _, H, W, _ = pixel_values_THWC.shape
        h_patch, w_patch = H // ps, W // ps
        image = pixel_values_THWC.permute(0, 3, 1, 2)

        D = h_SD.shape[-1]
        n_img = h_valid * w_valid
        img_feats = h_SD[img_token_start : img_token_start + n_img, :]
        lr_img_features = img_feats.reshape(1, h_valid, w_valid, D)
        if h_valid < h_patch or w_valid < w_patch:
            lr_img_features = F.pad(
                lr_img_features, (0, 0, 0, w_patch - w_valid, 0, h_patch - h_valid)
            )
        lr_img_features = lr_img_features.permute(0, 3, 1, 2)
        lr_img_features = self.conv_segm(lr_img_features)

        if output_size == (h_patch, w_patch):
            return lr_img_features[0]
        if output_size is None:
            output_size = (H, W)

        out_H, out_W = output_size
        if (
            shrink_image
            and H > out_H
            and W > out_W
            and H % out_H == 0
            and W % out_W == 0
        ):
            image = _pool_to(image, (out_H, out_W))

        hr_img_features = self.itok_upsampler(
            images=image, features=lr_img_features, output_size=output_size
        )
        return hr_img_features[0]

    # -- forward -------------------------------------------------------------------

    def forward(
        self,
        tokens: T,
        mask: T,
        kv_cache: KVCache,
        rope_pos_t: Optional[T] = None,
        rope_pos_hw: Optional[T] = None,
        pixel_values: Optional[T] = None,
        coord_xy: Optional[T] = None,
        size_hw: Optional[T] = None,
        img_scatter_info=None,
    ):
        """One prefill (S > 1) or decode (S == 1) step.

        ``mask`` is the boolean keep-mask for the current query rows against
        every cached key, broadcastable to ``(B, 1, S, kv_len)``.
        """
        B, S = tokens.size()
        T_pos = kv_cache.get_pos()
        if S != 1:
            assert rope_pos_t is not None and rope_pos_hw is not None
            pos_t = rope_pos_t[:, T_pos : T_pos + S].long()
            kv_cache.pos_t = pos_t[:, -1:]
            freqs_cis = self.freqs_cis[pos_t]
            hw = rope_pos_hw[:, T_pos : T_pos + S]
            freqs_cis_golden = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden, hw)
        else:
            pos_t = kv_cache.increment_and_get_pos_t()
            freqs_cis = self.freqs_cis[pos_t]
            freqs_cis_golden = None

        h_BSD = self.tok_embeddings(tokens)

        if self.args.perception_heads:
            coord_xy = coord_xy if coord_xy is not None else h_BSD.new_empty(0)
            size_hw = size_hw if size_hw is not None else h_BSD.new_empty(0)
            h_BSD = self._encode_special(
                h_BSD, tokens, coord_xy, self.args.coord_token_id, self.coord_encoder
            )
            h_BSD = self._encode_special(
                h_BSD, tokens, size_hw, self.args.size_token_id, self.size_encoder
            )

        if pixel_values is not None:
            assert img_scatter_info is not None
            h_BSD = self.scatter_img_tokens(h_BSD, pixel_values, img_scatter_info)

        for layer in self.layers.values():
            h_BSD = layer(
                h_BSD,
                kv_cache=kv_cache,
                freqs_cis=freqs_cis,
                freqs_cis_2d=freqs_cis_golden,
                mask=mask,
            )

        h_BSD = self.norm(h_BSD)
        logits_BSV = self.output(h_BSD)
        return logits_BSV, h_BSD

    # -- per-step heads ------------------------------------------------------------

    def sample_bbox(self, h_BD: T, tokens_B: T):
        """Decode (x, y) and (h, w) from the last hidden state, plus masks."""
        B = h_BD.shape[0]
        if not self.args.perception_heads:
            zeros = torch.zeros(B, 2, device=h_BD.device, dtype=torch.float32)
            false = torch.zeros(B, device=h_BD.device, dtype=torch.bool)
            return zeros, zeros, false, false, zeros

        is_coord_B = tokens_B == self.args.coord_token_id
        is_size_B = tokens_B == self.args.size_token_id

        coord_logits = self.coord_decoder(h_BD).view(B, 2, -1)
        num_bins = coord_logits.size(-1)
        xy_B2 = torch.argmax(coord_logits, dim=-1).float() / num_bins

        size_logits = self.size_decoder(h_BD).view(B, 2, -1)
        hw_B2 = self.process_sizes(size_logits)
        return xy_B2, hw_B2, is_coord_B, is_size_B, coord_logits

    @staticmethod
    def dedup_single_coord(
        xy_2: T,
        is_coord: T,
        all_xy_S2: T,
        is_coord_mask_S: T,
        coord_logits_2N: T,
        threshold: float = 0.01,
        max_attempts: int = 10,
    ) -> None:
        """Replace a duplicate center prediction in place (iterative bin masking)."""
        diffs = (all_xy_S2 - xy_2.unsqueeze(0)).abs()
        is_close = (diffs.amax(dim=-1) < threshold) & is_coord_mask_S
        if not is_close.any():
            return

        num_bins = coord_logits_2N.size(-1)
        logits = coord_logits_2N.clone()

        for _ in range(max_attempts):
            pred_bins = torch.argmax(logits, dim=-1)
            pred_xy = pred_bins.float() / num_bins

            diffs = (all_xy_S2 - pred_xy.unsqueeze(0)).abs()
            is_repeat = ((diffs.amax(dim=-1) < threshold) & is_coord_mask_S).any().item()

            if not is_repeat:
                xy_2.copy_(torch.where(is_coord, pred_xy, xy_2))
                return

            logits[0, pred_bins[0]] = float("-inf")
            logits[1, pred_bins[1]] = float("-inf")
