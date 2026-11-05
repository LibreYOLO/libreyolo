"""Sparse 3D UNet refiner for MoGe-3, in plain PyTorch.

The module layout and parameter names follow ``Sparse3DUNet`` and its blocks
from Microsoft MoGe at commit 74fbce054ebed49800de42d0ad0e83495065719a (MIT).
Upstream runs the submanifold convolution, strided mean pooling, and nearest
upsampling through FlexGEMM GPU kernels (MIT, commit
b2fadb29d41846c7981ade6801ffc689fae119cf). This file re-expresses the same
operator semantics with standard PyTorch ops (sort, searchsorted, unique,
gather, index_add, matmul) so the refiner runs on CPU, CUDA, and MPS without a
compiled extension:

- submanifold 3x3x3 convolution: output sites equal input sites; tap ``v`` of
  the kernel reads the input site at ``coord + offset_v`` when it exists, with
  ``offset_v`` enumerated row-major over ``(i, j, z)`` in ``{-1, 0, 1}`` and the
  weight laid out as ``(C_out, 3, 3, 3, C_in)``;
- strided mean pooling with kernel = stride = ``f``: each output site is
  ``floor(coord / f)`` over the spatial axes and averages its children;
- nearest upsampling back to the pre-pooling sites copies the parent value.

Accumulation order differs from the fused kernels, so outputs agree to
floating-point rounding rather than bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Row-major (i, j, z) enumeration of the 27 taps of a 3x3x3 kernel.
_TAP_OFFSETS = tuple(
    (di, dj, dz) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dz in (-1, 0, 1)
)
_CENTER_TAP = _TAP_OFFSETS.index((0, 0, 0))


@dataclass
class SparseLevel:
    """Active voxel sites at one resolution.

    ``coords`` is ``(M, 4)`` int64 with columns ``(batch, i, j, z)``; ``size``
    is the dense extent ``(B, H, W, Z)``. ``neighbors`` caches the
    submanifold neighbor table, shared by every convolution at this level.
    """

    coords: torch.Tensor
    size: tuple[int, int, int, int]
    neighbors: torch.Tensor | None = None


def _linear_keys(coords: torch.Tensor, size: tuple[int, int, int, int]) -> torch.Tensor:
    _, height, width, depth = size
    batch, i, j, z = coords.unbind(dim=1)
    return ((batch * height + i) * width + j) * depth + z


def build_neighbor_table(level: SparseLevel) -> torch.Tensor:
    """Return ``(27, M)`` int64 input indices per tap, ``M`` where absent."""
    coords = level.coords
    count = int(coords.shape[0])
    _, height, width, depth = level.size
    keys = _linear_keys(coords, level.size)
    sorted_keys, order = torch.sort(keys)
    bounds = torch.tensor((height, width, depth), device=coords.device)
    table = []
    for di, dj, dz in _TAP_OFFSETS:
        if (di, dj, dz) == (0, 0, 0):
            table.append(torch.arange(count, device=coords.device))
            continue
        delta = torch.tensor((0, di, dj, dz), device=coords.device)
        query = coords + delta
        spatial = query[:, 1:]
        inside = ((spatial >= 0) & (spatial < bounds)).all(dim=1)
        query_keys = _linear_keys(query, level.size)
        position = torch.searchsorted(sorted_keys, query_keys).clamp_max(count - 1)
        found = inside & (sorted_keys[position] == query_keys)
        table.append(torch.where(found, order[position], count))
    return torch.stack(table, dim=0)


def _neighbors(level: SparseLevel) -> torch.Tensor:
    if level.neighbors is None:
        level.neighbors = build_neighbor_table(level)
    return level.neighbors


class SubmanifoldConv3d(nn.Module):
    """3x3x3 submanifold convolution with FlexGEMM's weight layout."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_channels * 27)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feats: torch.Tensor, level: SparseLevel) -> torch.Tensor:
        table = _neighbors(level)
        padded = torch.cat((feats, feats.new_zeros(1, feats.shape[1])), dim=0)
        taps = self.weight.flatten(1, 3)
        out = F.linear(feats, taps[:, _CENTER_TAP], self.bias)
        for tap in range(len(_TAP_OFFSETS)):
            if tap == _CENTER_TAP:
                continue
            out = out + padded[table[tap]] @ taps[:, tap].transpose(0, 1)
        return out


def sparse_mean_pool(
    feats: torch.Tensor,
    level: SparseLevel,
    factor: int,
) -> tuple[torch.Tensor, SparseLevel, torch.Tensor]:
    """Average children into ``floor(coord / factor)`` parents.

    Returns the pooled features, the parent level, and the child-to-parent
    index used by the matching nearest upsample.
    """
    batch, height, width, depth = level.size
    size = (
        batch,
        -(-height // factor),
        -(-width // factor),
        -(-depth // factor),
    )
    parents = level.coords.clone()
    parents[:, 1:] = torch.div(parents[:, 1:], factor, rounding_mode="floor")
    keys = _linear_keys(parents, size)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    num_parents = int(unique_keys.shape[0])
    sums = feats.new_zeros(num_parents, feats.shape[1]).index_add_(0, inverse, feats)
    counts = torch.zeros(num_parents, device=feats.device, dtype=feats.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=feats.dtype))
    pooled = sums / counts.unsqueeze(1)

    coords = torch.empty(num_parents, 4, dtype=parents.dtype, device=parents.device)
    coords[inverse] = parents
    return pooled, SparseLevel(coords, size), inverse


class SparseResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6)
        self.conv1 = SubmanifoldConv3d(channels, channels)
        self.conv2 = SubmanifoldConv3d(channels, channels)
        self.skip_connection = nn.Identity()

    def forward(self, feats: torch.Tensor, level: SparseLevel) -> torch.Tensor:
        hidden = F.silu(self.norm1(feats).type_as(feats))
        hidden = self.conv1(hidden, level)
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden, level)
        return hidden + self.skip_connection(feats)


class PoolDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, feats: torch.Tensor, level: SparseLevel):
        feats, parent, inverse = sparse_mean_pool(feats, level, self.factor)
        return self.linear(feats), parent, inverse


class NearestUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, feats: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        return self.linear(feats)[inverse]


class Sparse3DUNet(nn.Module):
    """Sparse 3D UNet conditioned on the dense encoder feature at its bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: int,
        model_channels: list[int],
        encoder_blocks_per_level: int = 1,
        decoder_blocks_per_level: int = 1,
        bottleneck_blocks: int = 1,
        downsample_factors: list[int] | None = None,
        encoder_downsample: int = 16,
    ):
        super().__init__()
        levels = len(model_channels)
        if downsample_factors is None:
            downsample_factors = [2] * (levels - 1)
        if math.prod(downsample_factors) != encoder_downsample:
            raise ValueError(
                "MoGe-3 refiner encoder_downsample must equal the product of "
                "downsample_factors."
            )
        self.encoder_downsample = encoder_downsample
        self.downsample_factors = list(downsample_factors)

        self.input_proj = nn.Linear(in_channels, model_channels[0])
        self.encoder_fuse = nn.Linear(encoder_channels, model_channels[-1])
        bottleneck = model_channels[-1]
        self.fuse_proj = nn.Sequential(
            nn.Linear(bottleneck * 2, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.down_stages = nn.ModuleList(
            [
                self._stage(channels, encoder_blocks_per_level)
                for channels in model_channels
            ]
        )
        self.downsample_blocks = nn.ModuleList(
            [
                PoolDown(model_channels[index], model_channels[index + 1], factor)
                for index, factor in enumerate(self.downsample_factors)
            ]
        )
        self.bottleneck_stage = self._stage(bottleneck, bottleneck_blocks)
        self.upsample_blocks = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        for index in range(levels - 1):
            source = levels - 1 - index
            target = source - 1
            self.upsample_blocks.append(
                NearestUp(
                    model_channels[source],
                    model_channels[target],
                    self.downsample_factors[target],
                )
            )
            self.up_stages.append(
                self._stage(model_channels[target], decoder_blocks_per_level)
            )
        self.out_proj = nn.Linear(model_channels[0], out_channels)

    @staticmethod
    def _stage(channels: int, blocks: int) -> nn.ModuleList:
        return nn.ModuleList([SparseResBlock3d(channels) for _ in range(blocks)])

    def forward(
        self,
        feats: torch.Tensor,
        level: SparseLevel,
        encoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        _, height, width, _ = level.size
        if (
            height != encoder_feature.shape[2] * self.encoder_downsample
            or width != encoder_feature.shape[3] * self.encoder_downsample
        ):
            raise ValueError(
                "MoGe-3 refiner point grid must be the encoder grid times "
                f"{self.encoder_downsample}."
            )

        feats = self.input_proj(feats)
        levels = [level]
        inverses = []
        skips = []
        for index, stage in enumerate(self.down_stages):
            for block in stage:
                feats = block(feats, levels[-1])
            if index < len(self.downsample_blocks):
                skips.append(feats)
                feats, parent, inverse = self.downsample_blocks[index](
                    feats, levels[-1]
                )
                levels.append(parent)
                inverses.append(inverse)

        bottom = levels[-1].coords
        sampled = encoder_feature[bottom[:, 0], :, bottom[:, 1], bottom[:, 2]]
        sampled = self.encoder_fuse(sampled)
        feats = self.fuse_proj(torch.cat((feats, sampled), dim=-1))
        for block in self.bottleneck_stage:
            feats = block(feats, levels[-1])

        for index, (upsample, stage) in enumerate(
            zip(self.upsample_blocks, self.up_stages)
        ):
            target = len(levels) - 2 - index
            feats = upsample(feats, inverses[target]) + skips[target]
            for block in stage:
                feats = block(feats, levels[target])
        return self.out_proj(feats)


__all__ = [
    "NearestUp",
    "PoolDown",
    "Sparse3DUNet",
    "SparseLevel",
    "SparseResBlock3d",
    "SubmanifoldConv3d",
    "build_neighbor_table",
    "sparse_mean_pool",
]
