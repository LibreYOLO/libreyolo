# Ported from TripoSplat (https://github.com/VAST-AI-Research/TripoSplat,
# commit d8db9e018b413dd9c4a9fe22463781bf98e8e68d), MIT, Copyright (c) 2026 VAST.
"""TripoSplat networks: the denoiser and the octree Gaussian decoder.

Module and parameter names mirror upstream ``model.py`` so the released
safetensors load with ``strict=True``, and every forward keeps upstream's
operation order and precision casts. Upstream's ``Gaussian`` holder and
``_build_gaussians`` become ``OctreeGaussianDecoder.to_gaussians``; the
octree sampler draws from an explicit generator instead of the global one.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Upstream configuration (hard-coded in upstream's triposplat.py) ---------

FLOW_MODEL_ARGS = dict(
    q_token_length=8192,
    in_channels=16,
    cam_channels=5,
    out_channels=16,
    model_channels=1024,
    cond_channels=1280,
    cond2_channels=128,
    num_refiner_blocks=2,
    num_blocks=24,
    num_heads=16,
    mlp_ratio=4,
    qk_rms_norm=True,
    share_mod=True,
    use_shift_table=True,
)

OCTREE_DECODER_ARGS = dict(
    model_channels=1024,
    cond_channels=16,
    num_blocks=4,
    num_heads=16,
    mlp_ratio=4,
    share_mod=True,
)

GS_DECODER_ARGS = dict(
    in_channels=3,
    model_channels=1024,
    cond_channels=16,
    num_blocks=16,
    num_heads=16,
    mlp_ratio=4,
    use_learned_offset_scale=True,
    use_per_offset=True,
    representation_config=dict(
        lr=dict(_xyz=1.0, _features_dc=1.0, _opacity=1.0, _scaling=1.0, _rotation=0.1),
        perturb_offset=True,
        perturbe_size=1.5,
        offset_scale=0.05,
        num_gaussians=32,
        filter_kernel_size_3d=0.0009,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation="softplus",
    ),
)

MAX_VOXEL_LEVEL = 8


# -- Shared helpers ----------------------------------------------------------


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return (F.normalize(x.float(), dim=-1) * self.gamma.float() * self.scale).to(
            origin_dtype
        )


def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_rotated = torch.view_as_complex(
        hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2)
    )
    x_rotated = x_rotated * freqs
    x_out = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1)
    return x_out.type_as(hidden_states)


def clamp_mul(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    f_t = f.tanh()
    return x * f_t + x.detach() * (f - f_t)


def scaled_dot_product_attention(
    qkv: Optional[torch.Tensor] = None,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    kv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """(qkv [N, L, 3, H, C]) or (q [N, L, H, C], k, v [N, S, H, C]) or (q, kv [N, S, 2, H, C]) -> [N, L, H, C]."""
    if qkv is not None:
        q, k, v = qkv.unbind(dim=2)
    elif kv is not None:
        k, v = kv.unbind(dim=2)
    q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    return F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3)


# -- Positional embeddings ---------------------------------------------------


class RePo3DRotaryEmbedding(nn.Module):
    """Learned rotary position: each token's features predict a 3D offset per head, turned into RoPE angles."""

    def __init__(
        self,
        model_channels: int,
        num_heads: int,
        head_dim: int,
        repo_hidden_ratio: float = 0.125,
        max_freq: float = 16.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        freqs = [
            torch.linspace(1.0, float(max_freq), steps=d // 2, dtype=torch.float32)
            for d in (self.dim_0, self.dim_1, self.dim_2)
        ]
        self.freqs_0 = nn.Parameter(freqs[0])
        self.freqs_1 = nn.Parameter(freqs[1])
        self.freqs_2 = nn.Parameter(freqs[2])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = clamp_mul(delta_pos[..., 0].unsqueeze(-1), self.freqs_0) * torch.pi
        ang_1 = clamp_mul(delta_pos[..., 1].unsqueeze(-1), self.freqs_1) * torch.pi
        ang_2 = clamp_mul(delta_pos[..., 2].unsqueeze(-1), self.freqs_2) * torch.pi
        # fp32: torch.polar needs it to produce complex64
        ang = torch.cat([ang_0, ang_1, ang_2], dim=-1).float()
        return torch.polar(torch.ones_like(ang), ang).type(torch.complex64)


class PcdAbsolutePositionEmbedder(nn.Module):
    """Sinusoidal embedding of 3D positions in [0, 1] (the denoiser's latent token positions)."""

    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self, device) -> torch.Tensor:
        freqs_2exp = torch.arange(self.max_res, dtype=torch.float32, device=device)
        res_dim = max(0, self.freq_dim - self.max_res)
        freqs_res = (
            torch.arange(res_dim, dtype=torch.float32, device=device)
            / max(res_dim, 1)
            * self.max_res
            if res_dim > 0
            else torch.empty(0, device=device)
        )
        freqs = torch.cat([freqs_2exp, freqs_res], dim=0)[: self.freq_dim]
        return torch.pow(2.0, freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        *dims, _ = x.shape
        out = torch.outer(x.reshape(-1), self._freqs(x.device)) * 2 * torch.pi
        out = torch.cat([out.sin(), out.cos()], dim=-1).reshape(*dims, -1)
        if out.shape[-1] < self.channels:
            out = torch.cat(
                [
                    out,
                    torch.zeros(
                        *dims,
                        self.channels - out.shape[-1],
                        device=out.device,
                        dtype=out.dtype,
                    ),
                ],
                dim=-1,
            )
        return out.to(orig_dtype)


class PcdAbsolutePositionEmbedderV2(nn.Module):
    """Sinusoidal embedding of 3D points in [0, 1] (the decoders' query points)."""

    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self, device) -> torch.Tensor:
        logs = torch.linspace(
            0.0,
            float(self.max_res),
            steps=self.freq_dim,
            dtype=torch.float32,
            device=device,
        )
        return torch.pow(2.0, logs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        N, _ = x.shape
        ang = x.unsqueeze(-1) * self._freqs(x.device) * torch.pi
        embed = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(
                        N,
                        self.channels - embed.shape[1],
                        device=embed.device,
                        dtype=embed.dtype,
                    ),
                ],
                dim=-1,
            )
        return embed.to(orig_dtype)


# -- Transformer building blocks ---------------------------------------------


class FeedForwardNet(nn.Module):
    def __init__(
        self, channels: int, mlp_ratio: float = 4.0, channels_out: Optional[int] = None
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(
                int(channels * mlp_ratio),
                channels if channels_out is None else channels_out,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        inner_channels: int,
        channels_out: Optional[int] = None,
        mlp_layer_num: int = 2,
    ):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(
                nn.Linear(channels if i == 0 else inner_channels, inner_channels)
            )
            layers.append(nn.GELU(approximate="tanh"))
        layers.append(
            nn.Linear(
                inner_channels, channels if channels_out is None else channels_out
            )
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RopeMultiHeadAttention(nn.Module):
    """Self attention with rotary embeddings (the denoiser)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, rope_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.use_rope:
            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        h = scaled_dot_product_attention(q=q, k=k, v=v)
        return self.out(h.reshape(B, L, C))


class MultiHeadAttention(nn.Module):
    """Self or cross attention, optionally with RMS-normalised queries and keys (the decoders)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be a multiple of num_heads")
        if type not in ("self", "cross"):
            raise ValueError(f"type must be 'self' or 'cross', got {type!r}")
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1)
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = torch.stack([q, k, v], dim=2)
            h = scaled_dot_product_attention(qkv=qkv)
        else:
            Lkv = context.shape[1]
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            kv = self.to_kv(context).reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q=q, k=k, v=v)
            else:
                h = scaled_dot_product_attention(q=q, kv=kv)
        return self.to_out(h.reshape(B, L, -1))


class UnifiedTransformerBlock(nn.Module):
    """Denoiser block: RoPE self attention + FFN, modulated by the diffusion time (adaLN, shared or per block,
    plus this block's shift table); modulation=False for the condition refiner."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        modulation: bool = True,
        share_mod: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.modulation = modulation
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(
            channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.shift_table = None
        if modulation:
            if not share_mod:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
                )
            if use_shift_table:
                self.shift_table = nn.Parameter(
                    torch.randn(1, 6 * channels) / channels**0.5
                )

    def forward(
        self,
        x: torch.Tensor,
        mod: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            return x + self.mlp(self.norm2(x))
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        if self.shift_table is not None:
            mod = mod + self.shift_table.type(mod.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, rope_emb=rotary_emb)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.mlp(h) * gate_mlp.unsqueeze(1)


# -- Quasi-random sampling ---------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base: int, n: int) -> float:
    val = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim: int, n: int) -> list:
    return [radical_inverse(PRIMES[d], n) for d in range(dim)]


def hammersley_sequence(dim: int, n: int, num_samples: int) -> list:
    return [n / num_samples] + halton_sequence(dim - 1, n)


@torch.no_grad()
def sample_probs(
    probs: torch.Tensor,
    counts: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Split counts[...] items among the P bins of probs[..., P] by systematic resampling -> [..., P] counts."""
    batch_shape = counts.shape
    B = counts.numel()
    P = probs.size(-1)
    device = probs.device
    probs = probs.view(B, P)
    counts = counts.view(B)

    probs = probs.to(torch.float32).clamp_min_(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / P

    counts = counts.to(device=device, dtype=torch.long)
    out = torch.zeros(B, P, dtype=torch.long, device=device)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    unique_n, inv = counts.unique(sorted=False, return_inverse=True)
    for i, n in enumerate(unique_n.tolist()):
        if n == 0:
            continue
        rows = (inv == i).nonzero(as_tuple=False).squeeze(1)
        r = rows.numel()
        U0 = torch.rand(r, 1, device=device, generator=generator) / float(n)
        grid = torch.arange(n, device=device, dtype=torch.float32)[None, :] / float(n)
        us = (U0 + grid).clamp(max=1.0 - 1e-12)
        cdf_rows = cdf.index_select(0, rows)
        idx = torch.searchsorted(cdf_rows, us).clamp_max(probs.size(1) - 1)
        buf = torch.zeros(r, P, dtype=torch.float32, device=device)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, rows, buf.to(torch.long))

    return out.view(*batch_shape, P)


# -- Octree decoder ----------------------------------------------------------


class LevelEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 1024,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def level_embedding(
        t: torch.Tensor, dim: int, max_period: int = 1024
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None] * 2 * torch.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.level_embedding(t, self.frequency_embedding_size, self.max_period)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """Octree decoder block: cross attention to the latent + FFN, modulated by the octree level."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        qk_rms_norm_cross: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(
        self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.cross_attn(h, context) * gate_msa.unsqueeze(1)
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.mlp(h) * gate_mlp.unsqueeze(1)


class ModulatedCrossOnlyTransformerBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        additional_level_embed: bool = False,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm_cross = qk_rms_norm_cross

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.l_embedder = LevelEmbedder(model_channels)
        self.l_embedder2 = (
            LevelEmbedder(model_channels, max_period=100)
            if additional_level_embed
            else None
        )
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossOnlyBlock(
                    model_channels,
                    ctx_channels=cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    share_mod=self.share_mod,
                )
                for _ in range(num_blocks)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        x: torch.Tensor,
        l: torch.Tensor,
        cond: torch.Tensor,
        l2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_layer(x)
        l_emb = self.l_embedder(l)
        if self.l_embedder2 is not None and l2 is not None:
            l_emb = l_emb + self.l_embedder2(l2)
        if self.share_mod:
            l_emb = self.adaLN_modulation(l_emb)
        for block in self.blocks:
            h = block(h, l_emb, cond)
        return h


class OctreeProbabilityFixedlenDecoder(ModulatedCrossOnlyTransformerBase):
    """Predicts, for each octree cell, how its points split among the 8 children."""

    def __init__(
        self,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        additional_level_embed: bool = False,
        qk_rms_norm_cross: bool = True,
        *,
        no_norm: bool = False,
    ):
        super().__init__(
            in_channels=model_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            share_mod=share_mod,
            additional_level_embed=additional_level_embed,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.out_proj = nn.Linear(self.model_channels, 8)
        self.no_norm = no_norm
        self.in_proj = nn.Linear(3, self.model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderV2(
            channels=model_channels, in_channels=3
        )

    def forward(
        self,
        x: torch.Tensor,
        l: torch.Tensor,
        cond: torch.Tensor,
        l2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: cell centres [B, L, 3] in [0, 1]; l: the children's resolution [B]; cond: latent [B, Lc, C].
        Returns the logits [B, L, 8]."""
        d = self.dtype
        B, L, _ = x.shape
        h = self.in_proj(x.to(d)) + self.pos_embedder(x.reshape(-1, 3)).reshape(
            B, L, -1
        ).to(d)
        if l2 is not None:
            l2 = torch.log2(l2)
        h = super().forward(h, l, cond.to(d), l2)
        h = (
            F.layer_norm(h.float(), h.shape[-1:]).to(d)
            if not self.no_norm
            else h / (1 + 2 * self.num_blocks) ** 0.5
        )
        return self.out_proj(h)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_points: int,
        level: int,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Place num_points anchor points [B, num_points, 3] in [0, 1]^3, descending the octree to ``level``."""
        B = cond.shape[0]
        device = cond.device
        child_offset = torch.tensor(
            [[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]],
            dtype=torch.long,
            device=device,
        )
        prev_coords_int = torch.zeros(B, 1, 3, dtype=torch.long, device=device)
        prev_counts = torch.full((B, 1), num_points, dtype=torch.long, device=device)
        batch_indices_range = torch.arange(B, device=device).unsqueeze(1)

        for lv in range(1, level + 1):
            res_p = 1 << (lv - 1)
            res = 1 << lv
            parent_coords_norm = (prev_coords_int.to(torch.float32) + 0.5) / res_p
            res_tensor = torch.full((B,), res, dtype=torch.long, device=device)
            pred_logits = self(parent_coords_norm, res_tensor, cond) / temperature
            pred_probs = torch.softmax(pred_logits, dim=-1)
            sampled = sample_probs(pred_probs, prev_counts, generator).flatten(1, 2)
            child_coords_int = (
                prev_coords_int[:, :, None, :] * 2 + child_offset[None, None, :, :]
            ).flatten(1, 2)
            # keep the children that received points, packed to the front
            mask = sampled > 0
            max_valid = mask.sum(dim=1).max().item()
            valid_scatter_indices = (mask.cumsum(dim=1) - 1)[mask]
            valid_batch_indices = batch_indices_range.expand_as(mask)[mask]
            next_coords = torch.zeros(
                B, max_valid, 3, dtype=child_coords_int.dtype, device=device
            )
            next_coords[valid_batch_indices, valid_scatter_indices] = child_coords_int[
                mask
            ]
            next_counts = torch.zeros(B, max_valid, dtype=sampled.dtype, device=device)
            next_counts[valid_batch_indices, valid_scatter_indices] = sampled[mask]
            prev_coords_int = next_coords
            prev_counts = next_counts

        res = 1 << level
        coords_int = torch.repeat_interleave(
            prev_coords_int.flatten(0, 1), prev_counts.flatten(0, 1), dim=0
        ).reshape(B, num_points, -1)
        # each point lands at a uniformly random position inside its leaf cell
        jitter = torch.rand(
            coords_int.shape, dtype=torch.float32, device=device, generator=generator
        )
        return (coords_int.to(torch.float32) + jitter) / res


# -- Gaussian decoder --------------------------------------------------------


class TransformerCrossBlock(nn.Module):
    """Gaussian decoder block: self attention + cross attention to the latent + FFN."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class TransformerBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                TransformerCrossBlock(
                    model_channels,
                    ctx_channels=cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h, cond)
        return h


class FixedlenDecoder(TransformerBase):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__(
            in_channels=model_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderV2(
            channels=model_channels, in_channels=3
        )

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        d = self.dtype
        B, L, _ = points.shape
        h = self.in_proj(points.to(d)) + self.pos_embedder(
            points.reshape(-1, 3)
        ).reshape(B, L, -1).to(d)
        return super().forward(h, cond.to(d))


class ElasticGaussianFixedlenDecoder(FixedlenDecoder):
    """Turns each anchor point into ``num_gaussians`` Gaussians (offsets, colour, scale, rotation, opacity)."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4.0,
        *,
        no_norm: bool = False,
        representation_config: Optional[dict] = None,
        use_learned_offset_scale: bool = True,
        use_per_offset: bool = True,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        self.rep_config = representation_config
        self.use_learned_offset_scale = use_learned_offset_scale
        self.use_per_offset = use_per_offset
        self.out_channels = self._calc_layout()
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            cond_channels=cond_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            qk_rms_norm=qk_rms_norm,
            qk_rms_norm_cross=qk_rms_norm_cross,
        )
        self.out_proj = nn.Linear(model_channels, self.out_channels)
        self.no_norm = no_norm
        self._build_perturbation()

    def _calc_layout(self) -> int:
        ng = self.rep_config["num_gaussians"]
        self.layout = {
            "_xyz": {"shape": (ng, 3), "size": ng * 3},
            "_features_dc": {"shape": (ng, 1, 3), "size": ng * 3},
            "_scaling": {"shape": (ng, 3), "size": ng * 3},
            "_rotation": {"shape": (ng, 4), "size": ng * 4},
            "_opacity": {"shape": (ng, 1), "size": ng},
        }
        if self.use_learned_offset_scale and self.use_per_offset:
            self.layout["_offset_scale"] = {"shape": (ng, 1), "size": ng}
        start = 0
        for v in self.layout.values():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        return start

    def _build_perturbation(self) -> None:
        ng = self.rep_config["num_gaussians"]
        perturbation = torch.tensor(
            [hammersley_sequence(3, i, ng) for i in range(ng)]
        ).float()
        perturbation = torch.atanh(
            (perturbation * 2 - 1) / self.rep_config["perturbe_size"]
        )
        self.register_buffer("points_offset_perturbation", perturbation)
        if self.use_learned_offset_scale:
            base = torch.tensor(self.rep_config["offset_scale"])
            self.register_buffer("base_offset_scale", torch.log(torch.exp(base) - 1.0))

    def _get_offset(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        if self.use_learned_offset_scale:
            r = self.layout["_offset_scale"]["range"]
            offset_scale = F.softplus(
                h[:, :, r[0] : r[1]].reshape(
                    B, -1, *self.layout["_offset_scale"]["shape"]
                )
                + self.base_offset_scale
            )
        r = self.layout["_xyz"]["range"]
        offset = h[:, :, r[0] : r[1]].reshape(B, -1, *self.layout["_xyz"]["shape"])
        offset = offset * self.rep_config["lr"]["_xyz"]
        if self.rep_config["perturb_offset"]:
            offset = offset + self.points_offset_perturbation
        offset = torch.tanh(offset) * 0.5 * self.rep_config["perturbe_size"]
        return offset * (
            offset_scale
            if self.use_learned_offset_scale
            else self.rep_config["offset_scale"]
        )

    def forward(self, points: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """points: anchors [B, L, 3]; cond: latent [B, Lc, C] -> per-anchor features [B, L, out_channels]"""
        h = super().forward(points, cond)
        h = (
            F.layer_norm(h.float(), h.shape[-1:]).to(h.dtype)
            if not self.no_norm
            else h / (1 + 3 * self.num_blocks) ** 0.5
        )
        return self.out_proj(h)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class OctreeGaussianDecoder(nn.Module):
    """The octree decoder and the Gaussian decoder, as one checkpoint (``octree.*`` and ``gs.*``)."""

    def __init__(self, octree_args: dict, gs_args: dict):
        super().__init__()
        self.octree = OctreeProbabilityFixedlenDecoder(**octree_args)
        self.gs = ElasticGaussianFixedlenDecoder(**gs_args)

    @property
    def gaussians_per_point(self) -> int:
        return self.gs.rep_config["num_gaussians"]

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        num_gaussians: int,
        level: int = MAX_VOXEL_LEVEL,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Activated Gaussians of the first object in the batch (see ``to_gaussians``)."""
        num_points = max(1, num_gaussians // self.gaussians_per_point)
        points = self.octree.sample(
            latent, num_points=num_points, level=level, generator=generator
        )
        return self.to_gaussians(points, self.gs(points, latent))

    def to_gaussians(
        self, points: torch.Tensor, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Upstream's ``Gaussian`` holder, activated: xyz [N, 3] (z-up, inside [-0.5, 0.5]^3), features_dc
        [N, 1, 3], scaling [N, 3], rotation [N, 4] unit (w, x, y, z), opacity [N, 1]."""
        gs = self.gs
        config = gs.rep_config
        offset = gs._get_offset(features)
        raw_xyz = (offset[0] + points[0, :, None, :]).flatten(0, 1)
        raw = {}
        for name in ("_features_dc", "_scaling", "_rotation", "_opacity"):
            spec = gs.layout[name]
            raw[name] = (
                features[0][:, spec["range"][0] : spec["range"][1]]
                .reshape(-1, *spec["shape"])
                .flatten(0, 1)
                * config["lr"][name]
            )
        device = raw_xyz.device
        aabb = torch.tensor(
            [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], dtype=torch.float32, device=device
        )
        if config["scaling_activation"] == "exp":
            activation, inverse = torch.exp, torch.log
        else:
            activation, inverse = F.softplus, inverse_softplus
        scale_bias = inverse(torch.tensor(config["scaling_bias"])).to(device)
        opacity_bias = inverse_sigmoid(torch.tensor(config["opacity_bias"])).to(device)
        rotation_bias = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        # The activations run in the features' precision (fp16 on GPU), as
        # upstream's Gaussian holder does; the payload is float32.
        scaling = activation(raw["_scaling"] + scale_bias)
        kernel = config["filter_kernel_size_3d"]
        return {
            "xyz": raw_xyz.float() * aabb[None, 3:] + aabb[None, :3],
            "features_dc": raw["_features_dc"].float(),
            "scaling": torch.sqrt(torch.square(scaling) + kernel**2).float(),
            # upstream writes the raw biased quaternion; viewers normalise it,
            # and the Gaussians payload asks for unit quaternions
            "rotation": F.normalize(
                raw["_rotation"].float() + rotation_bias[None, :], dim=-1
            ),
            "opacity": torch.sigmoid(raw["_opacity"] + opacity_bias).float(),
        }


# -- Flow matching denoiser --------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class LatentSeqMMFlowModel(nn.Module):
    """The denoiser: a rectified-flow transformer over the object latent (8192 tokens) and the input camera
    (1 token), conditioned on the image tokens. Joint attention over [latent, image, camera] with learned
    rotary positions. The latent token positions are a fixed scrambled Sobol sequence (seed 123), as upstream."""

    def __init__(
        self,
        q_token_length: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_refiner_blocks: int = 2,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        cam_channels: Optional[int] = None,
        cond2_channels: Optional[int] = None,
        mlp_ratio: float = 4,
        share_mod: bool = True,
        qk_rms_norm: bool = False,
        use_shift_table: bool = False,
    ):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond2_channels = cond2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.use_shift_table = use_shift_table

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = (
            nn.Linear(cond2_channels, model_channels)
            if cond2_channels is not None
            else None
        )

        # A plain attribute, as upstream: a buffer would be cast to fp16 with
        # the weights, and the position embedding needs the fp32 positions.
        self.pos_pe = (
            torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123)
            .draw(q_token_length)
            .unsqueeze(0)
        )
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)

        def rope_layers(count):
            return nn.ModuleList(
                [
                    RePo3DRotaryEmbedding(
                        model_channels,
                        num_heads=self.num_heads,
                        head_dim=num_head_channels,
                    )
                    for _ in range(count)
                ]
            )

        self.noise_repo_layers = rope_layers(num_refiner_blocks)
        self.context_repo_layers = rope_layers(num_refiner_blocks)
        self.repo_layers = rope_layers(num_blocks)

        block_kwargs = dict(
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            use_rope=True,
            qk_rms_norm=self.qk_rms_norm,
            use_shift_table=self.use_shift_table,
        )
        self.noise_refiner = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    model_channels,
                    modulation=True,
                    share_mod=self.share_mod,
                    **block_kwargs,
                )
                for _ in range(num_refiner_blocks)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    model_channels, modulation=False, **block_kwargs
                )
                for _ in range(num_refiner_blocks)
            ]
        )
        if self.cam_channels is not None:
            self.cam_refiner = MLP(
                self.cam_channels,
                model_channels,
                model_channels,
                mlp_layer_num=num_refiner_blocks,
            )
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    model_channels,
                    modulation=True,
                    share_mod=self.share_mod,
                    **block_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.shift_table = (
            nn.Parameter(torch.randn(1, 2, model_channels) / model_channels**0.5)
            if use_shift_table
            else None
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        if cam_channels is not None:
            self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        latent: torch.Tensor,
        camera: Optional[torch.Tensor],
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ):
        """Velocity of the object latent [B, 8192, 16] and camera [B, 1, 5] at time t [B] in [0, 1000], given the
        image tokens cond {'feature1': [B, L_c, 1280] DINOv3, 'feature2': [B, L_c, 128] FLUX.2 VAE}."""
        d = self.dtype
        z = latent.to(d)
        feat1 = cond["feature1"].to(d)
        feat2 = cond["feature2"].to(d) if self.cond_embedder2 is not None else None

        h_x = self.input_layer(z)
        h_cond = self.cond_embedder(feat1)
        if feat2 is not None:
            h_cond = h_cond + self.cond_embedder2(feat2)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb

        self.pos_pe = self.pos_pe.to(z.device)
        h_x = h_x + self.pos_embedder(self.pos_pe).to(d)

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x))

        for i, block in enumerate(self.context_refiner):
            h_cond = block(
                h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond)
            )

        parts = [h_x, h_cond]
        if self.cam_channels is not None:
            cam = camera.to(d)
            parts.append(self.cam_refiner(cam))
        h = torch.cat(parts, dim=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h))

        h_x = F.layer_norm(h[:, : z.shape[1]].float(), h.shape[-1:]).type(d)
        if self.cam_channels is not None:
            h_cam = F.layer_norm(h[:, -cam.shape[1] :].float(), h.shape[-1:]).type(d)

        if self.use_shift_table:
            shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
            h_x = h_x * (1 + scale) + shift
            if self.cam_channels is not None:
                h_cam = h_cam * (1 + scale) + shift

        out_latent = self.out_layer(h_x)
        if self.cam_channels is None:
            return out_latent, None
        return out_latent, self.cam_out_layer(h_cam)
