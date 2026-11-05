# Adapted from TRELLIS (https://github.com/microsoft/TRELLIS, commit
# 193e0502b5cb3eb5997ec457d5340e95d09b158e), MIT, Copyright (c) Microsoft
# Corporation: ``trellis/models`` and ``trellis/modules``.
"""TRELLIS networks for image to 3D Gaussians.

Four networks, as in upstream's ``TRELLIS-image-large`` pipeline:

- ``SparseStructureFlowModel``: a dense transformer that denoises a 16^3
  latent of the object's occupancy, conditioned on DINOv2 image tokens.
- ``SparseStructureDecoder``: a 3D CNN that turns that latent into a 64^3
  occupancy grid; its active voxels are the object's structure.
- ``SLatFlowModel``: a sparse transformer that denoises 8 latent channels on
  those voxels (the structured latent), with submanifold-convolution ResBlocks
  around it.
- ``SLatGaussianDecoder``: a shifted-window sparse transformer that turns each
  voxel's latent into 32 Gaussians.

Module and parameter names mirror upstream so the released safetensors load
with ``strict=True``; the torso of each network is fp16 when ``use_fp16`` is
set, as upstream converts it, and each forward keeps upstream's precision
casts. The sparse operations are the PyTorch ones in ``sparse.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import (
    SparseTensor,
    SubMConv3d,
    sparse_cross_attention,
    sparse_downsample,
    sparse_self_attention,
    sparse_upsample,
    sparse_windowed_self_attention,
)

# -- Precision helpers -------------------------------------------------------

_FP16_TYPES = (nn.Conv3d, nn.Linear, SubMConv3d)


def convert_module_to_f16(module: nn.Module) -> None:
    """Upstream's torso conversion: Linear and convolution parameters to fp16; norms stay fp32."""
    if isinstance(module, _FP16_TYPES):
        for parameter in module.parameters():
            parameter.data = parameter.data.half()


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    """LayerNorm over the channel axis of a [B, C, *spatial] tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.dim()
        x = x.permute(0, *range(2, dims), 1).contiguous()
        x = super().forward(x)
        return x.permute(0, dims - 1, *range(1, dims - 1)).contiguous()


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


# -- Embeddings --------------------------------------------------------------


class AbsolutePositionEmbedder(nn.Module):
    """Sinusoidal embedding of integer 3D positions (no parameters)."""

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x.reshape(-1).float(), self.freqs)
        embed = torch.cat([torch.sin(out), torch.cos(out)], dim=-1).reshape(n, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(n, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


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
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


# -- Dense attention and blocks (sparse-structure flow) ----------------------


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[N, L, H, C] in and out (upstream's ``sdpa`` backend)."""
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    )
    return out.permute(0, 2, 1, 3)


class MultiHeadAttention(nn.Module):
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
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if self._type == "self":
            q, k, v = self.to_qkv(x).reshape(B, L, 3, self.num_heads, -1).unbind(dim=2)
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        else:
            q = self.to_q(x).reshape(B, L, self.num_heads, -1)
            k, v = (
                self.to_kv(context)
                .reshape(B, context.shape[1], 2, self.num_heads, -1)
                .unbind(dim=2)
            )
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        return self.to_out(_dense_attention(q, k, v).reshape(B, L, -1))


class ModulatedTransformerCrossBlock(nn.Module):
    """Self attention + cross attention + FFN, adaLN-modulated by the diffusion time."""

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
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        x = x + self.cross_attn(h, context)
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.mlp(h) * gate_mlp.unsqueeze(1)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    dims = x.dim() - 2
    x = x.reshape(
        *x.shape[:2],
        *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, dims + 2)], []),
    )
    x = x.permute(
        0, 1, *([2 * i + 3 for i in range(dims)] + [2 * i + 2 for i in range(dims)])
    )
    return x.reshape(x.shape[0], x.shape[1] * (patch_size**dims), *(x.shape[-dims:]))


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    dims = x.dim() - 2
    x = x.reshape(
        x.shape[0],
        x.shape[1] // (patch_size**dims),
        *([patch_size] * dims),
        *(x.shape[-dims:]),
    )
    x = x.permute(0, 1, *(sum([[2 + dims + i, 2 + i] for i in range(dims)], [])))
    return x.reshape(
        x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(dims)]
    )


class SparseStructureFlowModel(nn.Module):
    """Denoiser of the 16^3 occupancy latent (a dense DiT with cross attention to the image tokens)."""

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
        patch_size: int = 2,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_unused,
    ):
        super().__init__()
        if pe_mode != "ape":
            raise ValueError(
                "Only the absolute position embedding of the released models is supported."
            )
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32
        num_heads = num_heads or model_channels // num_head_channels

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        side = resolution // patch_size
        grid = torch.meshgrid(*[torch.arange(side)] * 3, indexing="ij")
        coords = torch.stack(grid, dim=-1).reshape(-1, 3)
        self.register_buffer(
            "pos_emb", AbsolutePositionEmbedder(model_channels, 3)(coords)
        )
        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)
        if use_fp16:
            self.blocks.apply(convert_module_to_f16)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        side = self.resolution // self.patch_size
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], side, side, side)
        return unpatchify(h, self.patch_size).contiguous()


# -- Sparse-structure decoder (dense 3D CNN) ---------------------------------


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    c = C // scale_factor**3
    x = x.reshape(B, c, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, c, H * scale_factor, W * scale_factor, D * scale_factor)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """16^3 occupancy latent -> 64^3 occupancy logits."""

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: Sequence[int],
        num_res_blocks_middle: int = 2,
        norm_type: str = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        if norm_type != "layer":
            raise ValueError(
                "Only the layer-norm decoder of the released models is supported."
            )
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[
                ResBlock3d(channels[0], channels[0])
                for _ in range(num_res_blocks_middle)
            ]
        )
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )
        if use_fp16:
            self.blocks.apply(convert_module_to_f16)
            self.middle_block.apply(convert_module_to_f16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x).type(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        return self.out_layer(h.type(x.dtype))


# -- Sparse attention and blocks ---------------------------------------------


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: SparseTensor) -> SparseTensor:
        dtype = x.dtype
        feats = F.normalize(x.feats.float(), dim=-1)
        return x.replace((feats * self.gamma * self.scale).to(dtype))


class SparseMultiHeadAttention(nn.Module):
    """Self attention (full or shifted-window) over voxels, or cross attention from voxels to image tokens."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        attn_mode: str = "full",
        window_size: Optional[int] = None,
        shift_window: int = 0,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        if type == "cross" and qk_rms_norm:
            raise ValueError(
                "Cross attention with RMS-normalised keys is not used by the released models."
            )
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self, x: SparseTensor, context: Optional[torch.Tensor] = None
    ) -> SparseTensor:
        if self._type == "self":
            qkv = x.replace(
                self.to_qkv(x.feats).reshape(x.feats.shape[0], 3, self.num_heads, -1)
            )
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_self_attention(qkv)
            else:
                h = sparse_windowed_self_attention(
                    qkv, self.window_size, self.shift_window
                )
        else:
            q = x.replace(
                self.to_q(x.feats).reshape(x.feats.shape[0], self.num_heads, -1)
            )
            kv = self.to_kv(context)
            kv = kv.reshape(*kv.shape[:2], 2, self.num_heads, -1)
            h = sparse_cross_attention(q, kv)
        return h.replace(self.to_out(h.feats.reshape(h.feats.shape[0], -1)))


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


class SparseTransformerBlock(nn.Module):
    """Self attention + FFN (the Gaussian decoder)."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: str = "full",
        window_size: Optional[int] = None,
        shift_window: int = 0,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = x + self.attn(x.replace(self.norm1(x.feats)))
        return x + self.mlp(x.replace(self.norm2(x.feats)))


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Self attention + cross attention + FFN over voxels, adaLN-modulated by the diffusion time."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
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
            num_heads=num_heads,
            type="self",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        if not self.share_mod:
            mod = self.adaLN_modulation(mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
            6, dim=1
        )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        return x + h


# -- Structured-latent flow --------------------------------------------------


class SparseConv3d(nn.Module):
    """Upstream's wrapper name around the submanifold convolution (keeps the ``.conv.`` parameter prefix)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = SubMConv3d(in_channels, out_channels, kernel_size)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.conv(x)


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels, bias=True)
        )
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: SparseTensor, emb: torch.Tensor) -> SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        if self.downsample:
            x = sparse_downsample(x, 2)
        elif self.upsample:
            x = sparse_upsample(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        skip = (
            x.feats
            if isinstance(self.skip_connection, nn.Identity)
            else self.skip_connection(x.feats)
        )
        return h + x.replace(skip)


class SLatFlowModel(nn.Module):
    """Denoiser of the structured latent: 8 channels on each active voxel of the 64^3 structure."""

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
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: Optional[List[int]] = None,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_unused,
    ):
        super().__init__()
        if pe_mode != "ape" or io_block_channels is None:
            raise ValueError("Only the released SLat flow configuration is supported.")
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_mod = share_mod
        self.use_skip_connection = use_skip_connection
        self.dtype = torch.float16 if use_fp16 else torch.float32
        num_heads = num_heads or model_channels // num_head_channels

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels, io_block_channels[0])
        self.input_blocks = nn.ModuleList([])
        for chs, next_chs in zip(
            io_block_channels, io_block_channels[1:] + [model_channels]
        ):
            self.input_blocks.extend(
                [
                    SparseResBlock3d(chs, model_channels, out_channels=chs)
                    for _ in range(num_io_res_blocks - 1)
                ]
            )
            self.input_blocks.append(
                SparseResBlock3d(
                    chs, model_channels, out_channels=next_chs, downsample=True
                )
            )
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_blocks = nn.ModuleList([])
        for chs, prev_chs in zip(
            reversed(io_block_channels),
            [model_channels] + list(reversed(io_block_channels[1:])),
        ):
            self.out_blocks.append(
                SparseResBlock3d(
                    prev_chs * 2 if use_skip_connection else prev_chs,
                    model_channels,
                    out_channels=chs,
                    upsample=True,
                )
            )
            self.out_blocks.extend(
                [
                    SparseResBlock3d(
                        chs * 2 if use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks - 1)
                ]
            )
        self.out_layer = nn.Linear(io_block_channels[0], out_channels)
        if use_fp16:
            self.input_blocks.apply(convert_module_to_f16)
            self.blocks.apply(convert_module_to_f16)
            self.out_blocks.apply(convert_module_to_f16)

    def forward(
        self, x: SparseTensor, t: torch.Tensor, cond: torch.Tensor
    ) -> SparseTensor:
        h = x.replace(self.input_layer(x.feats)).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)
        skips = []
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = h.type(x.dtype)
        return h.replace(self.out_layer(h.feats))


# -- Gaussian decoder --------------------------------------------------------


def _radical_inverse(base: int, n: int) -> float:
    value, inverse, scale = 0.0, 1.0 / base, 1.0 / base
    while n > 0:
        value += (n % base) * scale
        n //= base
        scale *= inverse
    return value


def hammersley(dim: int, n: int, count: int) -> List[float]:
    """Point ``n`` of a ``count``-point Hammersley set in [0, 1]^dim (first coordinate n / count, then Halton)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    return [n / count] + [_radical_inverse(primes[d], n) for d in range(dim - 1)]


class SLatGaussianDecoder(nn.Module):
    """Structured latent -> 32 Gaussians per voxel (shifted-window sparse transformer)."""

    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
        attn_mode: str = "swin",
        window_size: int = 8,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        qk_rms_norm: bool = False,
        representation_config: Optional[dict] = None,
        **_unused,
    ):
        super().__init__()
        if attn_mode != "swin" or pe_mode != "ape":
            raise ValueError(
                "Only the released shifted-window Gaussian decoder is supported."
            )
        self.resolution = resolution
        self.rep_config = representation_config
        self.dtype = torch.float16 if use_fp16 else torch.float32
        num_heads = num_heads or model_channels // num_head_channels
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(latent_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="windowed",
                    window_size=window_size,
                    shift_window=window_size // 2 * (i % 2),
                    qk_rms_norm=qk_rms_norm,
                )
                for i in range(num_blocks)
            ]
        )
        count = representation_config["num_gaussians"]
        self.layout: Dict[str, Tuple[Tuple[int, ...], Tuple[int, int]]] = {}
        start = 0
        for name, shape in (
            ("_xyz", (count, 3)),
            ("_features_dc", (count, 1, 3)),
            ("_scaling", (count, 3)),
            ("_rotation", (count, 4)),
            ("_opacity", (count, 1)),
        ):
            size = int(np.prod(shape))
            self.layout[name] = (shape, (start, start + size))
            start += size
        self.out_layer = nn.Linear(model_channels, start)
        points = (
            torch.tensor([hammersley(3, i, count) for i in range(count)]).float() * 2
            - 1
        )
        self.register_buffer(
            "offset_perturbation",
            torch.atanh(points / representation_config["voxel_size"]),
        )
        if use_fp16:
            self.blocks.apply(convert_module_to_f16)

    def forward(self, x: SparseTensor) -> SparseTensor:
        """The raw per-voxel outputs [T, 32 x 14]; ``to_gaussians`` activates them."""
        h = x.replace(self.input_layer(x.feats))
        h = h + self.pos_embedder(x.coords[:, 1:])
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return h.replace(self.out_layer(h.feats))

    def to_gaussians(
        self, out: SparseTensor, index: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Batch element ``index`` of the decoder output as activated Gaussians in the object frame.

        Returns ``xyz`` [N, 3] (z up, centred in [-0.5, 0.5]^3), ``sh`` [N, 1, 3]
        (degree-0 colour), ``scales`` [N, 3] (linear), ``rotations`` [N, 4]
        (unit, w first) and ``opacities`` [N].
        """
        config = self.rep_config
        rows = out.layout[index]
        feats = out.feats[rows]
        centres = (out.coords[rows][:, 1:].float() + 0.5) / self.resolution

        def take(name: str) -> torch.Tensor:
            shape, (start, stop) = self.layout[name]
            return feats[:, start:stop].reshape(-1, *shape) * config["lr"][name]

        offsets = take("_xyz")
        if config["perturb_offset"]:
            offsets = offsets + self.offset_perturbation
        offsets = torch.tanh(offsets) / self.resolution * 0.5 * config["voxel_size"]
        xyz = (centres.unsqueeze(1) + offsets).flatten(0, 1) - 0.5

        def flat(name: str) -> torch.Tensor:
            return take(name).flatten(0, 1)

        def softplus_inverse(value: float) -> torch.Tensor:
            value = torch.tensor(value)
            return value + torch.log(-torch.expm1(-value))

        if config["scaling_activation"] == "softplus":
            scales = F.softplus(
                flat("_scaling")
                + softplus_inverse(config["scaling_bias"]).to(feats.device)
            )
        else:
            scales = torch.exp(
                flat("_scaling")
                + torch.log(torch.tensor(config["scaling_bias"])).to(feats.device)
            )
        scales = torch.sqrt(torch.square(scales) + config["3d_filter_kernel_size"] ** 2)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=feats.device)
        rotations = F.normalize(flat("_rotation") + identity[None, :], dim=-1)
        bias = torch.tensor(config["opacity_bias"])
        opacity_bias = torch.log(bias / (1 - bias)).to(feats.device)
        opacities = torch.sigmoid(flat("_opacity") + opacity_bias).reshape(-1)
        return {
            "xyz": xyz.float(),
            "sh": flat("_features_dc").float(),
            "scales": scales.float(),
            "rotations": rotations.float(),
            "opacities": opacities.float(),
        }
