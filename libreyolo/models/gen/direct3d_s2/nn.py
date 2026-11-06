# Adapted from Direct3D-S2 (https://github.com/DreamTechAI/Direct3D-S2, commit
# a1cf235b2881cff04a91900060a9546b40e7ee5d), MIT, Copyright (c) 2025
# DreamTechAI: ``direct3d_s2/models`` and ``direct3d_s2/modules``.
"""Direct3D-S2 networks for image to mesh.

Six networks, as in upstream's ``direct3d-s2-v-1-1`` release:

- ``DenseDiT``: a dense flow transformer that denoises a 16^3 latent of the
  object's occupancy, conditioned on DINOv2 image tokens.
- ``DenseShapeVAE`` (decoder only): a 3D CNN that turns it into a 64^3
  occupancy grid; its active voxels index the sparse latent.
- ``SparseDiT``: a sparse flow transformer with spatial sparse attention
  (SSA) that denoises a 16-channel latent on those voxels.
- ``SparseSDFVAE`` (decoder only): a shifted-window sparse transformer plus
  three subdivision blocks that turn the latent into a sparse signed distance
  field at 8x the latent resolution.
- ``VoxelRefiner`` / ``VoxelSignRefiner``: dense 3D UNets that fill in the
  SDF away from the surface (512^3), and the inside/outside sign (1024^3).

Direct3D-S2 shares TRELLIS's module code (MIT), so the dense transformer, the
dense decoder and the shifted-window sparse blocks are the TRELLIS port's
classes. Upstream converts every Linear and convolution of a network to fp16
(not only its torso as TRELLIS does) and its timestep embedder casts to the
weight dtype; both are reproduced here. Module and parameter names mirror
upstream so the released checkpoints load with ``strict=True``.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..trellis.nn import AbsolutePositionEmbedder as _TrellisPositionEmbedder
from ..trellis.nn import (
    LayerNorm32,
    SparseFeedForwardNet,
    SparseMultiHeadAttention,
    SparseStructureDecoder,
    SparseStructureFlowModel,
    SparseTransformerBlock,
)
from ..trellis.nn import TimestepEmbedder as _TrellisTimestepEmbedder
from .sparse import (
    SparseConv3d,
    SparseTensor,
    sparse_mean_pool,
    sparse_subdivide,
    spatial_sparse_attention,
)

_FP16_TYPES = (
    nn.Conv3d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def convert_module_to_f16(module: nn.Module) -> None:
    """Upstream's conversion: Linear and convolution parameters to fp16; norms stay fp32."""
    from .sparse import _TorchsparseConv

    if isinstance(module, _FP16_TYPES + (_TorchsparseConv,)):
        for parameter in module.parameters():
            parameter.data = parameter.data.half()


class TimestepEmbedder(_TrellisTimestepEmbedder):
    """Direct3D-S2's embedder casts the sinusoid to the MLP's dtype (fp16 when converted)."""

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(frequencies.to(self.mlp[0].weight.dtype))


class AbsolutePositionEmbedder(_TrellisPositionEmbedder):
    """TRELLIS's sinusoidal embedder with upstream's optional coordinate ``factor``."""

    def forward(self, x: torch.Tensor, factor: Optional[float] = None) -> torch.Tensor:
        if factor is not None:
            x = x * factor
        return super().forward(x)


# -- Dense stage -------------------------------------------------------------


class DenseDiT(SparseStructureFlowModel):
    """Denoiser of the 16^3 occupancy latent: TRELLIS's dense DiT, every layer fp16."""

    def __init__(self, *, latent_shape: Sequence[int] = (8, 16, 16, 16), use_fp16: bool = False, **kwargs):
        super().__init__(use_fp16=use_fp16, **kwargs)
        self.latent_shape = list(latent_shape)
        self.t_embedder = TimestepEmbedder(self.t_embedder.mlp[0].out_features)
        if use_fp16:
            self.apply(convert_module_to_f16)


class DenseShapeVAE(nn.Module):
    """The decoder of upstream's ``DenseShapeVAE``: 16^3 latent -> 64^3 occupancy logits."""

    def __init__(
        self,
        *,
        embed_dim: int,
        model_channels_decoder: Sequence[int],
        num_res_blocks_decoder: int,
        num_res_blocks_middle_decoder: int,
        out_channels: int = 1,
        use_fp16: bool = False,
        latents_scale: float = 1.0,
        latents_shift: float = 0.0,
        **_encoder_args,
    ):
        super().__init__()
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift
        self.decoder = SparseStructureDecoder(
            out_channels=out_channels,
            latent_channels=embed_dim,
            num_res_blocks=num_res_blocks_decoder,
            channels=list(model_channels_decoder),
            num_res_blocks_middle=num_res_blocks_middle_decoder,
            use_fp16=use_fp16,
        )
        if use_fp16:
            self.decoder.apply(convert_module_to_f16)

    @torch.no_grad()
    def occupied_voxels(self, latents: torch.Tensor, threshold: float) -> List[torch.Tensor]:
        """Per batch element, the ``(0, x, y, z)`` rows where sigmoid(occupancy) >= threshold."""
        logits = self.decoder(latents)
        return [
            (logits[i].sigmoid() >= threshold).float().squeeze(0).unsqueeze(0).nonzero()
            for i in range(len(logits))
        ]


# -- Sparse flow transformer with spatial sparse attention -------------------


class SparseLinear(nn.Linear):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class _CompressionBlock(nn.Module):
    """Upstream's ``SparseDownBlock3d_v1``: 1x1x1 conv, SiLU, mean pool by ``factor``."""

    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.act_layers = nn.Sequential(SparseConv3d(channels, channels, 1, padding=0))

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.act_layers(x)
        return sparse_mean_pool(h.replace(F.silu(h.feats)), self.factor)


class SpatialSparseAttention(nn.Module):
    """Upstream's SSA: compressed, selected and windowed attention, blended by a learned gate."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compression_block_size: int,
        selection_block_size: int,
        topk: int,
        window_size: int,
        shift_window: int,
        compression_version: str = "v1",
    ):
        super().__init__()
        if compression_version != "v1":
            raise ValueError(
                "Only the v1 compression block of the released models is supported."
            )
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size
        self.topk = topk
        self.window_size = window_size
        self.shift_window = shift_window
        kv = num_kv_heads * head_dim
        self.proj_q = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.proj_k = nn.Linear(hidden_size, kv, bias=False)
        self.proj_v = nn.Linear(hidden_size, kv, bias=False)
        self.proj_o = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
        self.compression_key = _CompressionBlock(kv, compression_block_size)
        self.compression_value = _CompressionBlock(kv, compression_block_size)
        size = compression_block_size
        self.intra_block_pe = nn.Parameter(torch.zeros(size, size, size, kv))
        self.gate = nn.Sequential(nn.Linear(hidden_size, 3, bias=False), nn.Sigmoid())

    def forward(self, x: SparseTensor) -> SparseTensor:
        feats = x.feats
        count = feats.shape[0]
        q = self.proj_q(feats).view(count, self.num_q_heads, self.head_dim)
        k = self.proj_k(feats)
        v = self.proj_v(feats)
        inner = x.coords[:, 1:].long() % self.compression_block_size
        position = self.intra_block_pe[inner[:, 0], inner[:, 1], inner[:, 2]].to(k.dtype)
        compressed_k = self.compression_key(x.replace(k + position))
        compressed_v = self.compression_value(x.replace(v))
        kv_shape = (-1, self.num_kv_heads, self.head_dim)
        compressed_k = compressed_k.replace(compressed_k.feats.view(*kv_shape))
        compressed_v = compressed_v.replace(compressed_v.feats.view(*kv_shape))
        compressed, selected, windowed = spatial_sparse_attention(
            x,
            q,
            k.view(*kv_shape),
            v.view(*kv_shape),
            compressed_k,
            compressed_v,
            topk=self.topk,
            selection_block_size=self.selection_block_size,
            compression_block_size=self.compression_block_size,
            window_size=self.window_size,
            shift=self.shift_window,
        )
        gate = self.gate(feats)
        out = (
            gate[:, 0:1, None] * compressed
            + gate[:, 1:2, None] * selected
            + gate[:, 2:3, None] * windowed
        )
        return x.replace(self.proj_o(out.reshape(count, -1)))


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """SSA self attention + cross attention to the image tokens + FFN, adaLN-modulated by the time."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        num_kv_heads: int,
        compression_block_size: int,
        selection_block_size: int,
        topk: int,
        window_size: int,
        shift_window: int,
        mlp_ratio: float = 4.0,
        compression_version: str = "v1",
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SpatialSparseAttention(
            channels,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=channels // num_heads,
            compression_block_size=compression_block_size,
            selection_block_size=selection_block_size,
            topk=topk,
            window_size=window_size,
            shift_window=shift_window,
            compression_version=compression_version,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, num_heads=num_heads, ctx_channels=ctx_channels, type="cross"
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
        )

    def forward(
        self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor
    ) -> SparseTensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(mod).chunk(6, dim=1)
        )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        x = x + h * gate_msa
        h = x.replace(self.norm2(x.feats))
        x = x + self.cross_attn(h, context)
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(h) * gate_mlp


class SparseDiT(nn.Module):
    """Denoiser of the 16-channel sparse latent on the occupied voxels."""

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
        num_kv_heads: int = 2,
        compression_block_size: int = 4,
        selection_block_size: int = 8,
        topk: int = 8,
        compression_version: str = "v2",
        mlp_ratio: float = 4,
        pe_mode: str = "ape",
        use_fp16: bool = False,
        sparse_conditions: bool = False,
        factor: float = 1.0,
        window_size: int = 8,
        use_shift: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        **_unused,
    ):
        super().__init__()
        if pe_mode != "ape" or sparse_conditions or share_mod or qk_rms_norm_cross:
            raise ValueError(
                "Only the configuration of the released Direct3D-S2 models is supported."
            )
        del qk_rms_norm  # SSA has no RMS norm; upstream ignores the flag too.
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        self.selection_block_size = selection_block_size
        self.sparse_conditions = sparse_conditions
        self.dtype = torch.float16 if use_fp16 else torch.float32
        num_heads = num_heads or model_channels // num_head_channels
        self.t_embedder = TimestepEmbedder(model_channels)
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    compression_block_size=compression_block_size,
                    selection_block_size=selection_block_size,
                    topk=topk,
                    window_size=window_size,
                    shift_window=window_size // 2 * (i % 2)
                    if use_shift
                    else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    compression_version=compression_version,
                )
                for i in range(num_blocks)
            ]
        )
        self.out_layer = SparseLinear(model_channels, out_channels)
        if use_fp16:
            self.apply(convert_module_to_f16)

    def forward(
        self, x: SparseTensor, t: torch.Tensor, cond: torch.Tensor
    ) -> SparseTensor:
        h = self.input_layer(x).to(self.dtype)
        t_emb = self.t_embedder(t).type(self.dtype)
        cond = cond.type(self.dtype)
        h = h + self.pos_embedder(h.coords[:, 1:], factor=self.factor).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h.to(x.dtype))


# -- Sparse SDF decoder ------------------------------------------------------


class SparseSubdivideBlock3d(nn.Module):
    """Conv + SiLU, subdivide every voxel into 8, conv + SiLU."""

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.act_layers = nn.Sequential(SparseConv3d(channels, out_channels, 3, padding=1))
        self.out_layers = nn.Sequential(
            SparseConv3d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = self.act_layers(x)
        h = sparse_subdivide(h.replace(F.silu(h.feats)))
        h = self.out_layers(h)
        return h.replace(F.silu(h.feats))


class SparseSDFDecoder(nn.Module):
    """Latent on the 64^3 (or 128^3) voxels -> SDF on the 8x finer voxels."""

    # Upstream hard-codes the decoder's 2^3 upsampling ratio for chunking.
    _UPSAMPLE_RATIO = 8

    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: int = 64,
        mlp_ratio: float = 4,
        window_size: int = 8,
        out_channels: int = 1,
        use_fp16: bool = False,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.chunk_size = chunk_size
        self.dtype = torch.float16 if use_fp16 else torch.float32
        num_heads = num_heads or model_channels // num_head_channels
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = SparseLinear(latent_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="windowed",
                    window_size=window_size,
                    shift_window=window_size // 2 * (i % 2),
                )
                for i in range(num_blocks)
            ]
        )
        self.upsample = nn.ModuleList(
            [
                SparseSubdivideBlock3d(model_channels, model_channels // 4),
                SparseSubdivideBlock3d(model_channels // 4, model_channels // 8),
                SparseSubdivideBlock3d(model_channels // 8, model_channels // 16),
            ]
        )
        self.out_layer = SparseLinear(model_channels // 16, out_channels)
        if use_fp16:
            self.apply(convert_module_to_f16)

    def _upsample(self, h: SparseTensor, dtype) -> Tuple[SparseTensor, SparseTensor]:
        for block in self.upsample:
            h = block(h)
        h = h.to(dtype)
        out = self.out_layer(h)
        return out.replace(torch.tanh(out.feats)), h

    def _chunks(self, h: SparseTensor, padding: int = 4):
        """Upstream's ``split_for_meshing``: overlapping sub-volumes with their crop bounds."""
        sub = self.resolution // self.chunk_size
        ratio = self._UPSAMPLE_RATIO
        coords = h.coords
        for i, j, k in itertools.product(range(self.chunk_size), repeat=3):
            starts = [max(0, c * sub - padding) for c in (i, j, k)]
            ends = [min((c + 1) * sub + padding, self.resolution) for c in (i, j, k)]
            mask = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
            for axis in range(3):
                mask &= (coords[:, axis + 1] >= starts[axis]) & (
                    coords[:, axis + 1] < ends[axis]
                )
            if not bool(mask.any()):
                continue
            local = coords[mask].clone()
            local[:, 1:] -= torch.tensor(starts, device=coords.device, dtype=local.dtype)
            original = []
            for c in (i, j, k):
                original += [c * sub * ratio, (c + 1) * sub * ratio + (ratio - 1)]
            offsets = [s * ratio for s in starts]
            yield SparseTensor(h.feats[mask], local), original, offsets

    def forward(
        self, x: SparseTensor, factor: Optional[float] = None, return_feat: bool = False
    ):
        h = self.input_layer(x)
        h = h + self.pos_embedder(x.coords[:, 1:], factor)
        h = h.to(self.dtype)
        for block in self.blocks:
            h = block(h)
        if self.chunk_size <= 1:
            out, feat = self._upsample(h, x.dtype)
            return (out, feat) if return_feat else out
        # Chunked decoding returns the SDF only, as upstream does.
        all_coords, all_feats = [], []
        for chunk, bounds, offsets in self._chunks(h):
            out, _ = self._upsample(chunk, chunk.dtype)
            for b in range(x.batch_size):
                rows = torch.nonzero(out.coords[:, 0] == b).squeeze(-1)
                if rows.numel() == 0:
                    continue
                coords = out.coords[rows].clone()
                coords[:, 1:] += torch.tensor(offsets, device=coords.device, dtype=coords.dtype)
                inside = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
                for axis in range(3):
                    inside &= (coords[:, axis + 1] >= bounds[2 * axis]) & (
                        coords[:, axis + 1] < bounds[2 * axis + 1]
                    )
                if bool(inside.any()):
                    all_coords.append(coords[inside])
                    all_feats.append(out.feats[rows][inside])
        return SparseTensor(torch.cat(all_feats), torch.cat(all_coords))


class SparseSDFVAE(nn.Module):
    """The decoder of upstream's ``SparseSDFVAE``."""

    def __init__(
        self,
        *,
        embed_dim: int,
        resolution: int,
        model_channels_decoder: int,
        num_blocks_decoder: int,
        num_heads_decoder: int,
        num_head_channels_decoder: int = 64,
        out_channels: int = 1,
        use_fp16: bool = False,
        chunk_size: int = 1,
        latents_scale: float = 1.0,
        latents_shift: float = 0.0,
        **_encoder_args,
    ):
        super().__init__()
        self.resolution = resolution
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift
        self.decoder = SparseSDFDecoder(
            resolution=resolution,
            model_channels=model_channels_decoder,
            latent_channels=embed_dim,
            num_blocks=num_blocks_decoder,
            num_heads=num_heads_decoder,
            num_head_channels=num_head_channels_decoder,
            out_channels=out_channels,
            use_fp16=use_fp16,
            chunk_size=chunk_size,
        )


# -- Dense refiners ----------------------------------------------------------


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, eps: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (
            nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = F.silu(self.norm1(x.float()).to(dtype))
        h = self.conv1(h)
        h = F.silu(self.norm2(h.float()).to(dtype))
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class _Resampler(nn.Module):
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, groups, eps, add_downsample):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock3D(in_channels if i == 0 else out_channels, out_channels, groups, eps)
                for i in range(num_layers)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([_Resampler(nn.Conv3d(out_channels, out_channels, 2, stride=2))])
            if add_downsample
            else None
        )

    def forward(self, x: torch.Tensor):
        states = ()
        for resnet in self.resnets:
            x = resnet(x)
            states += (x,)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            states += (x,)
        return x, states


class UpBlock3D(nn.Module):
    def __init__(
        self, in_channels, prev_output_channel, out_channels, num_layers, groups, eps, add_upsample
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            skip = in_channels if i == num_layers - 1 else out_channels
            first = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock3D(first + skip, out_channels, groups, eps))
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = (
            nn.ModuleList(
                [_Resampler(nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2))]
            )
            if add_upsample
            else None
        )

    def forward(self, x: torch.Tensor, skips: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(torch.cat([x, skips[-1]], dim=1))
            skips = skips[:-1]
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels, num_layers, groups, eps):
        super().__init__()
        self.resnets = nn.ModuleList(
            [ResnetBlock3D(channels, channels, groups, eps) for _ in range(num_layers + 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return x


class UNet3DModel(nn.Module):
    """Upstream's refiner UNet: four down blocks, a middle block, four up blocks, no output conv."""

    def __init__(
        self,
        in_channels: int,
        block_out_channels: Sequence[int] = (8, 32, 128, 512),
        layers_per_block: int = 2,
        layers_mid_block: int = 2,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        channels = list(block_out_channels)
        self.conv_in = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        output = channels[0]
        for i, width in enumerate(channels):
            self.down_blocks.append(
                DownBlock3D(
                    output, width, layers_per_block, norm_num_groups, norm_eps,
                    add_downsample=i < len(channels) - 1,
                )
            )
            output = width
        self.mid_block = UNetMidBlock3D(channels[-1], layers_mid_block, norm_num_groups, norm_eps)
        reversed_channels = channels[::-1]
        self.up_blocks = nn.ModuleList()
        output = reversed_channels[0]
        for i, width in enumerate(reversed_channels):
            previous = output
            skip = reversed_channels[min(i + 1, len(channels) - 1)]
            self.up_blocks.append(
                UpBlock3D(
                    skip, previous, width, layers_per_block + 1, norm_num_groups, norm_eps,
                    add_upsample=i < len(channels) - 1,
                )
            )
            output = width
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, channels[0], eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        skips: Tuple[torch.Tensor, ...] = (x,)
        for block in self.down_blocks:
            x, states = block(x)
            skips += states
        x = self.mid_block(x)
        for block in self.up_blocks:
            count = len(block.resnets)
            x = block(x, skips[-count:])
            skips = skips[:-count]
        dtype = x.dtype
        return F.silu(self.conv_norm_out(x.float()).to(dtype))


class GeoDecoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(n_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _adaptive_conv(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-voxel 3x3x3 filter whose 27 weights come from ``weights`` ``[B, 27, ...]``."""
    padded = F.pad(inputs, (1, 1, 1, 1, 1, 1), mode="constant", value=0)
    size = inputs.shape[-1]
    output = torch.zeros_like(inputs)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                index = i * 9 + j * 3 + k
                output = (
                    output
                    + padded[:, :, i : i + size, j : j + size, k : k + size]
                    * weights[:, index : index + 1]
                )
    return output


def _adaptive_block(inputs: torch.Tensor, conv: nn.Module, source=None) -> torch.Tensor:
    weights = F.normalize(conv(inputs if source is None else source), dim=1, p=1)
    for _ in range(3):
        inputs = _adaptive_conv(inputs, weights)
    return inputs


def _adaptive_branch() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv3d(8, 27, 3, padding=1, bias=False)
    )


class VoxelRefiner(nn.Module):
    """Upstream's ``Voxel_RefinerXL``: the 512^3 SDF far from the surface, from the sparse SDF and features."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        layers_per_block: int = 2,
        layers_mid_block: int = 2,
        patch_size: int = 192,
        res: int = 512,
        use_fp16: bool = False,
        **_unused,
    ):
        super().__init__()
        self.unet3d1 = UNet3DModel(16, layers_per_block=layers_per_block, layers_mid_block=layers_mid_block)
        self.conv_in = nn.Conv3d(in_channels, 8, 3, padding=1)
        self.latent_mlp = GeoDecoder(32)
        self.adaptive_conv1 = _adaptive_branch()
        self.adaptive_conv2 = _adaptive_branch()
        self.adaptive_conv3 = _adaptive_branch()
        self.mid_conv = nn.Conv3d(8, 8, 3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, 3, padding=1)
        self.patch_size = patch_size
        self.res = res
        if use_fp16:
            self.apply(convert_module_to_f16)

    def _patch(self, sdf: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        inputs = self.conv_in(sdf)
        feats = self.latent_mlp(feats.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        mid = self.unet3d1(torch.cat([inputs, feats], dim=1))
        mid = _adaptive_block(mid, self.adaptive_conv1)
        mid = self.mid_conv(mid)
        mid = _adaptive_block(mid, self.adaptive_conv2)
        final = self.conv_out(mid)
        final = _adaptive_block(final, self.adaptive_conv3, source=mid)
        return torch.tanh(final)

    @torch.no_grad()
    def refine(self, sdf: SparseTensor, feat: SparseTensor) -> torch.Tensor:
        """Sparse SDF and decoder features at 512^3 -> dense SDF ``[B, 1, 512, 512, 512]``.

        Upstream densifies the 32-channel features over the whole grid; here
        each 192^3 patch is filled from the sparse features directly, which
        gives the same values with a fraction of the memory.
        """
        res, patch, stride, steps = self.res, self.patch_size, 160, 3
        if patch != 192 or res != 512:
            raise ValueError("The 512 refiner tiles a 512^3 grid with 192^3 patches.")
        values, coords = sdf.feats, sdf.coords.long()
        device, dtype = values.device, values.dtype
        batch = sdf.batch_size
        grid = torch.ones(batch, 1, res, res, res, device=device, dtype=dtype)
        grid[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = values[:, 0]
        features = feat.feats.to(dtype)
        patches = []
        for i, j, k in itertools.product(range(steps), repeat=3):
            lo = torch.tensor([stride * i, stride * j, stride * k], device=device)
            inside = ((coords[:, 1:] >= lo) & (coords[:, 1:] < lo + patch)).all(dim=1)
            local = coords[inside, 1:] - lo
            crop = torch.zeros(batch, features.shape[1], patch, patch, patch, device=device, dtype=dtype)
            crop[coords[inside, 0], :, local[:, 0], local[:, 1], local[:, 2]] = features[inside]
            window = grid[
                :, :, stride * i : stride * i + patch, stride * j : stride * j + patch,
                stride * k : stride * k + patch,
            ]
            patches.append(self._patch(window, crop))
        return _blend_patches(patches, batch, res, device, dtype)


def _blend_patches(patches, batch, res, device, dtype) -> torch.Tensor:
    """Upstream's stitching of 3x3x3 patches of 192 at stride 160, blending 32-voxel overlaps linearly."""
    weights = torch.linspace(0, 1, steps=32, device=device, dtype=dtype)

    def join(parts, shape, dim):
        out = torch.ones(shape, dtype=dtype, device=device) * 2
        w = weights.view([1] * dim + [-1] + [1] * (4 - dim))

        def cut(t, a, b):
            index = [slice(None)] * 5
            index[dim] = slice(a, b)
            return t[tuple(index)]

        def put(a, b, value):
            index = [slice(None)] * 5
            index[dim] = slice(a, b)
            out[tuple(index)] = value

        first, second, third = parts
        put(0, 160, cut(first, 0, 160))
        put(192, 320, cut(second, 32, 160))
        put(352, res, cut(third, 32, None))
        put(160, 192, cut(first, 160, None) * (1 - w) + cut(second, 0, 32) * w)
        put(320, 352, cut(second, 160, None) * (1 - w) + cut(third, 0, 32) * w)
        return out

    lines = [
        join(patches[3 * n : 3 * n + 3], [batch, 1, 192, 192, res], 4) for n in range(9)
    ]
    layers = [join(lines[3 * n : 3 * n + 3], [batch, 1, 192, res, res], 3) for n in range(3)]
    return join(layers, [batch, 1, res, res, res], 2)


class VoxelSignRefiner(nn.Module):
    """Upstream's ``Voxel_RefinerXL_sign``: the inside/outside sign of the 1024^3 SDF, predicted at 512^3."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        layers_per_block: int = 2,
        layers_mid_block: int = 2,
        patch_size: int = 192,
        res: int = 512,
        use_fp16: bool = False,
        **_unused,
    ):
        super().__init__()
        self.unet3d1 = UNet3DModel(8, layers_per_block=layers_per_block, layers_mid_block=layers_mid_block)
        self.conv_in = nn.Conv3d(in_channels, 8, 3, padding=1)
        self.conv_out = nn.Conv3d(8, out_channels, 3, padding=1)
        self.patch_size = patch_size
        if use_fp16:
            self.apply(convert_module_to_f16)

    @torch.no_grad()
    def refine(self, sdf: SparseTensor) -> torch.Tensor:
        """Sparse SDF at 1024^3 -> dense signed SDF ``[B, 1, 1024, 1024, 1024]``."""
        full, coarse_res, stride, steps = 1024, 512, 128, 3
        values, coords = sdf.feats, sdf.coords.long()
        device, dtype = values.device, values.dtype
        batch = sdf.batch_size
        dense = torch.ones(batch, 1, full, full, full, device=device, dtype=dtype)
        dense[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = values[:, 0]
        coarse = sparse_mean_pool(sdf, 2)
        cc = coarse.coords.long()
        grid = torch.ones(batch, 1, coarse_res, coarse_res, coarse_res, device=device, dtype=dtype)
        grid[cc[:, 0], 0, cc[:, 1], cc[:, 2], cc[:, 3]] = coarse.feats[:, 0]
        sign = torch.ones_like(grid)
        patch = self.patch_size
        for i, j, k in itertools.product(range(steps), repeat=3):
            view = (
                slice(None), slice(None),
                slice(stride * i, stride * i + patch),
                slice(stride * j, stride * j + patch),
                slice(stride * k, stride * k + patch),
            )
            out = torch.sigmoid(self.conv_out(self.unet3d1(self.conv_in(grid[view]))))
            out[out >= 0.5] = 1
            out[out < 0.5] = -1
            sign[view] = out
        del grid
        signed = dense.abs()
        for axis in (2, 3, 4):
            sign = sign.repeat_interleave(2, dim=axis)
        signed.mul_(sign)
        del sign
        signed[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = values[:, 0]
        return signed


__all__ = [
    "DenseDiT",
    "DenseShapeVAE",
    "SparseDiT",
    "SparseSDFVAE",
    "VoxelRefiner",
    "VoxelSignRefiner",
]
