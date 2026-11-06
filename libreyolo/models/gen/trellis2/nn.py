# Adapted from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2, commit
# 75fbf0183001ed9876c8dbb35de6b68552ee08bd), MIT, Copyright (c) Microsoft
# Corporation: ``trellis2/models`` and ``trellis2/modules``.
"""TRELLIS.2 networks for image to 3D mesh.

The geometry path of upstream's ``TRELLIS.2-4B`` pipeline:

- ``SparseStructureFlowModel``: a dense 1.3B transformer with 3D rotary
  positions that denoises a 16^3 occupancy latent, conditioned on DINOv3
  image tokens. The latent is decoded by TRELLIS's ``SparseStructureDecoder``
  (imported from the TRELLIS family; TRELLIS.2 reuses that network).
- ``SLatFlowModel``: a sparse 1.3B transformer with 3D rotary positions that
  denoises 32 latent channels on the occupied voxels (the shape latent).
- ``FlexiDualGridVaeDecoder``: a sparse ConvNeXt U-Net decoder that upsamples
  the latent 16 times, predicting at each step which of the 8 children of a
  voxel exist, and outputs a dual vertex, three edge-intersection flags and a
  quad split weight per voxel. ``mesh.py`` turns those into triangles.

Module and parameter names mirror upstream so the released safetensors load
with ``strict=True``. Each network keeps upstream's precision: the torso in
the published dtype (bf16 transformers, fp16 decoders), norms and the
outer layers in fp32. Sparse operations are the PyTorch ones of
``trellis/sparse.py``; the 3D rotary embedding is written here.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trellis.nn import (
    FeedForwardNet,
    LayerNorm32,
    MultiHeadRMSNorm,
    SparseStructureDecoder,
    TimestepEmbedder,
    _dense_attention,
)
from ..trellis.sparse import SparseTensor, SubMConv3d, _sdpa, sparse_self_attention

__all__ = [
    "FlexiDualGridVaeDecoder",
    "SLatFlowModel",
    "SparseStructureDecoder",
    "SparseStructureFlowModel",
    "SparseUnetVaeDecoder",
]

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def convert_module_to(module: nn.Module, dtype: torch.dtype) -> None:
    """Upstream's torso conversion: Linear and convolution parameters to ``dtype``; norms keep theirs."""
    if isinstance(module, (nn.Linear, nn.Conv3d, SubMConv3d)):
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype)


# -- 3D rotary position embedding ---------------------------------------------


def rope_phases(
    positions: torch.Tensor, head_dim: int, freq: Tuple[float, float] = (1.0, 10000.0)
) -> torch.Tensor:
    """Unit complex phases ``[N, head_dim // 2]`` of integer 3D ``positions`` ``[N, 3]``.

    Each axis gets ``head_dim // 6`` frequencies; the leftover pairs rotate by
    zero, as upstream pads them.
    """
    freq_dim = head_dim // 2 // 3
    freqs = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
    freqs = (freq[0] / (freq[1] ** freqs)).to(positions.device)
    angles = torch.outer(positions.reshape(-1), freqs)
    phases = torch.polar(torch.ones_like(angles), angles)
    phases = phases.reshape(*positions.shape[:-1], -1)
    pad = head_dim // 2 - phases.shape[-1]
    if pad > 0:
        phases = torch.cat(
            [
                phases,
                torch.polar(
                    torch.ones(*phases.shape[:-1], pad, device=phases.device),
                    torch.zeros(*phases.shape[:-1], pad, device=phases.device),
                ),
            ],
            dim=-1,
        )
    return phases


def apply_rope(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs of ``x`` ``[..., L, H, C]`` by ``phases`` ``[L, C // 2]`` (complex).

    Upstream multiplies in complex64; so does this on CUDA and CPU. MPS has no
    complex multiply, so there the same product is written out in real
    arithmetic (equal up to fp32 rounding).
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    phases = phases.unsqueeze(-2)
    if x.device.type == "mps":
        cos, sin = phases.real, phases.imag
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
        return rotated.reshape(x.shape).to(x.dtype)
    rotated = torch.view_as_complex(pairs) * phases
    return torch.view_as_real(rotated).reshape(x.shape).to(x.dtype)


# -- Dense transformer (sparse-structure flow) ---------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels or channels, channels * 2)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, length, _ = x.shape
        if self._type == "self":
            q, k, v = self.to_qkv(x).reshape(b, length, 3, self.num_heads, -1).unbind(2)
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
            if phases is not None:
                q, k = apply_rope(q, phases), apply_rope(k, phases)
        else:
            q = self.to_q(x).reshape(b, length, self.num_heads, -1)
            kv = self.to_kv(context).reshape(b, context.shape[1], 2, self.num_heads, -1)
            k, v = kv.unbind(2)
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        h = _dense_attention(q, k, v).reshape(b, length, -1)
        return self.to_out(h)


def _modulation(block: nn.Module, mod: torch.Tensor):
    if block.share_mod:
        return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)
    return block.adaLN_modulation(mod).chunk(6, dim=1)


class ModulatedTransformerCrossBlock(nn.Module):
    """Self attention with RoPE, cross attention to the image tokens, MLP; adaLN modulation."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels, num_heads, type="self", qk_rms_norm=qk_rms_norm
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            num_heads,
            ctx_channels=ctx_channels,
            type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(torch.zeros(6 * channels))
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _modulation(
            self, mod
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        x = x + self.cross_attn(h, context)
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return x + h * gate_mlp.unsqueeze(1)


def _check_pe_mode(pe_mode: str) -> None:
    if pe_mode != "rope":
        raise ValueError(
            f"TRELLIS.2's released flow models use pe_mode='rope', got {pe_mode!r}."
        )


class SparseStructureFlowModel(nn.Module):
    """Dense flow transformer over the 16^3 occupancy latent."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
        pe_mode: str = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_: Any,
    ):
        super().__init__()
        _check_pe_mode(pe_mode)
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads or model_channels // num_head_channels
        self.share_mod = share_mod
        self.dtype = _DTYPES[dtype]
        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        # Upstream computes the phases once on the CPU at construction.
        grid = torch.meshgrid(*[torch.arange(resolution)] * 3, indexing="ij")
        positions = torch.stack(grid, dim=-1).reshape(-1, 3)
        self.register_buffer(
            "rope_phases",
            rope_phases(positions, model_channels // self.num_heads, rope_freq),
            persistent=False,
        )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.convert_to(self.dtype)

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        return h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3)


# -- Sparse transformer (shape latent flow) ------------------------------------


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = type
        self.use_rope = use_rope
        self.rope_freq = tuple(rope_freq)
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels or channels, channels * 2)
        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def _phases(self, x: SparseTensor) -> torch.Tensor:
        key = ("rope_phases", self.head_dim, self.rope_freq)
        phases = x.cache.get(key)
        if phases is None:
            phases = rope_phases(x.coords[:, 1:], self.head_dim, self.rope_freq)
            x.cache[key] = phases
        return phases

    def forward(
        self, x: SparseTensor, context: Optional[torch.Tensor] = None
    ) -> SparseTensor:
        heads = self.num_heads
        if self._type == "self":
            qkv = self.to_qkv(x.feats).reshape(-1, 3, heads, self.head_dim)
            if self.qk_rms_norm or self.use_rope:
                q, k, v = qkv.unbind(dim=1)
                if self.qk_rms_norm:
                    q, k = self.q_rms_norm(q), self.k_rms_norm(k)
                if self.use_rope:
                    phases = self._phases(x)
                    q, k = apply_rope(q, phases), apply_rope(k, phases)
                qkv = torch.stack([q, k, v], dim=1)
            h = sparse_self_attention(x.replace(qkv)).feats
        else:
            q = self.to_q(x.feats).reshape(-1, heads, self.head_dim)
            kv = self.to_kv(context)
            k, v = kv.reshape(*kv.shape[:2], 2, heads, self.head_dim).unbind(2)
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
            parts = [
                _sdpa(q[s][None], k[i : i + 1], v[i : i + 1])[0]
                for i, s in enumerate(x.layout)
            ]
            h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return x.replace(self.to_out(h.reshape(h.shape[0], -1)))


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.mlp(x.feats))


class ModulatedSparseTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            type="self",
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            ctx_channels=ctx_channels,
            type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(torch.zeros(6 * channels))
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _modulation(
            self, mod
        )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        x = x + h * gate_msa
        h = x.replace(self.norm2(x.feats))
        x = x + self.cross_attn(h, context)
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return x + h * gate_mlp


class SLatFlowModel(nn.Module):
    """Sparse flow transformer over the latent of the occupied voxels."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
        pe_mode: str = "rope",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = "float32",
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_: Any,
    ):
        super().__init__()
        _check_pe_mode(pe_mode)
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads or model_channels // num_head_channels
        self.share_mod = share_mod
        self.dtype = _DTYPES[dtype]
        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_rope=True,
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.convert_to(self.dtype)

    def convert_to(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def forward(
        self,
        x: SparseTensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        concat_cond: Optional[SparseTensor] = None,
    ) -> SparseTensor:
        if concat_cond is not None:
            x = x.replace(torch.cat([x.feats, concat_cond.feats], dim=-1))
        h = x.replace(self.input_layer(x.feats).type(self.dtype))
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        feats = h.feats.type(x.dtype)
        feats = F.layer_norm(feats, feats.shape[-1:])
        return x.replace(self.out_layer(feats))


# -- Sparse U-Net decoder ------------------------------------------------------


def channel_to_spatial(
    x: SparseTensor, subdivision: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upstream's ``SparseChannel2Spatial(2)`` index map: (child coords, flat source rows).

    ``subdivision`` ``[N, 8]`` bool says which of each voxel's 8 children
    exist; child ``i`` sits at offset ``(i & 1, i >> 1 & 1, i >> 2 & 1)``.
    Children are ordered by parent, then by child index.
    """
    counts = subdivision.sum(dim=-1)
    child = subdivision.nonzero()[:, -1]
    coords = x.coords.clone()
    coords[:, 1:] *= 2
    coords = torch.repeat_interleave(coords, counts, dim=0, output_size=child.shape[0])
    for axis in range(3):
        coords[:, axis + 1] += (child // 2**axis % 2).to(coords.dtype)
    parent = torch.repeat_interleave(
        torch.arange(x.coords.shape[0], device=x.device),
        counts,
        dim=0,
        output_size=child.shape[0],
    )
    return coords, parent * 8 + child


class SparseConvNeXtBlock3d(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = SubMConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.SiLU(),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.conv(x)
        h = h.replace(self.mlp(self.norm(h.feats)))
        return h + x


class SparseResBlockC2S3d(nn.Module):
    """Residual block that doubles the resolution, moving channels to child voxels."""

    def __init__(
        self, channels: int, out_channels: Optional[int] = None, pred_subdiv: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.pred_subdiv = pred_subdiv
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SubMConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = SubMConv3d(self.out_channels, self.out_channels, 3)
        if pred_subdiv:
            self.to_subdiv = nn.Linear(channels, 8)

    def forward(
        self, x: SparseTensor, subdiv: Optional[SparseTensor] = None
    ) -> Tuple[SparseTensor, Optional[SparseTensor]]:
        if self.pred_subdiv:
            subdiv = x.replace(self.to_subdiv(x.feats))
        h = F.silu(self.norm1(x.feats))
        h = self.conv1(x.replace(h)).feats
        coords, rows = channel_to_spatial(x, subdiv.feats > 0)
        h = h.reshape(h.shape[0] * 8, -1)[rows]
        skip = x.feats.reshape(x.feats.shape[0] * 8, -1)[rows]
        skip = skip.repeat_interleave(self.out_channels // (self.channels // 8), dim=1)
        h = SparseTensor(F.silu(self.norm2(h)), coords, batch_size=x.batch_size)
        h = self.conv2(h)
        return h.replace(h.feats + skip), subdiv


class SparseUnetVaeDecoder(nn.Module):
    """Latent -> features at 16x the latent resolution, through 4 C2S upsamplings."""

    def __init__(
        self,
        out_channels: int,
        model_channels: Sequence[int],
        latent_channels: int,
        num_blocks: Sequence[int],
        block_type: Sequence[str],
        up_block_type: Sequence[str],
        block_args: Sequence[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        for name in block_type:
            if name != "SparseConvNeXtBlock3d":
                raise ValueError(f"Unsupported decoder block {name!r}.")
        for name in up_block_type:
            if name != "SparseResBlockC2S3d":
                raise ValueError(f"Unsupported decoder upsampling block {name!r}.")
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.output_layer = nn.Linear(model_channels[-1], out_channels)
        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.blocks = nn.ModuleList()
        for i, count in enumerate(num_blocks):
            stage = nn.ModuleList(
                [
                    SparseConvNeXtBlock3d(model_channels[i], **block_args[i])
                    for _ in range(count)
                ]
            )
            if i < len(num_blocks) - 1:
                stage.append(
                    SparseResBlockC2S3d(
                        model_channels[i],
                        model_channels[i + 1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
            self.blocks.append(stage)
        if use_fp16:
            self.blocks.apply(partial(convert_module_to, dtype=torch.float16))

    def _run(
        self,
        x: SparseTensor,
        guide_subs: Optional[List[SparseTensor]],
        stop_at: Optional[int] = None,
    ):
        h = x.replace(self.from_latent(x.feats).type(self.dtype))
        subs = []
        for i, stage in enumerate(self.blocks):
            if i == stop_at:
                return h, subs
            for j, block in enumerate(stage):
                if i < len(self.blocks) - 1 and j == len(stage) - 1:
                    guide = guide_subs[i] if guide_subs is not None else None
                    h, sub = block(h, guide)
                    subs.append(sub)
                else:
                    h = block(h)
        return h, subs

    def forward(
        self,
        x: SparseTensor,
        guide_subs: Optional[List[SparseTensor]] = None,
        return_subs: bool = False,
    ):
        if guide_subs is not None and self.pred_subdiv:
            raise ValueError("guide_subs needs a decoder with pred_subdiv=False.")
        h, subs = self._run(x, guide_subs)
        feats = h.feats.type(x.dtype)
        feats = F.layer_norm(feats, feats.shape[-1:])
        h = h.replace(self.output_layer(feats))
        return (h, subs) if return_subs else h

    def upsample(self, x: SparseTensor, upsample_times: int) -> torch.Tensor:
        """Coordinates of the voxels after ``upsample_times`` predicted subdivisions."""
        h, _ = self._run(x, None, stop_at=upsample_times)
        return h.coords


class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    """Shape latent -> per-voxel dual vertex, edge intersections and quad split weight."""

    def __init__(
        self,
        resolution: int,
        model_channels: Sequence[int],
        latent_channels: int,
        num_blocks: Sequence[int],
        block_type: Sequence[str],
        up_block_type: Sequence[str],
        block_args: Sequence[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def decode_fields(self, x: SparseTensor) -> Tuple[Dict[str, Any], List[SparseTensor]]:
        """The decoder's mesh fields per voxel, and the subdivisions the texture decoder follows."""
        h, subs = self.forward(x, return_subs=True)
        feats = h.feats
        margin = self.voxel_margin
        fields = {
            "coords": h.coords,
            "layout": h.layout,
            "dual_vertices": (1 + 2 * margin) * torch.sigmoid(feats[..., 0:3]) - margin,
            "intersected": feats[..., 3:6] > 0,
            "split_weight": F.softplus(feats[..., 6:7]),
        }
        return fields, subs
