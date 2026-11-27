"""LoopedViT backbone for LoopDVR.

Ported from Intellindust-AI-Lab/LoopDVR (commit f2aba1f122029e4db757b69cd37fc5a4909db34e):
    loopdvr/models/looped_vit.py
Copyright (c) 2026 The LoopDVR Authors (modifications).
SPDX-License-Identifier: Apache-2.0 AND MIT

Modified from Raptor (https://github.com/KempnerInstitute/raptor)
Copyright (c) 2025 mozesjacobs.

Modified from EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.

2D rotary position embedding (`RopePositionEmbedding` and `rope_apply`) is
imported from `libreyolo/models/deimv2/engine/backbone/dinov3/` rather than
re-vendored here; see root NOTICE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..deimv2.engine.backbone.dinov3.layers.attention import rope_apply
from ..deimv2.engine.backbone.dinov3.layers.rope_position_encoding import (
    RopePositionEmbedding,
)
from .encoder import ConvNormLayer_fuse
from .loop_adapter import MLP, LoopAdapter


class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        if patch_size != 16:
            raise ValueError(
                f"ConvPyramidPatchEmbed requires patch_size=16, got {patch_size}"
            )
        ratios = [2**index for index in range(int(math.log2(patch_size)) - 1, 0, -1)]
        channels = [embed_dim // ratio for ratio in ratios]
        if any(channel <= 0 for channel in channels):
            raise ValueError(f"embed_dim is too small for the patch pyramid: {embed_dim}")
        self.convs = nn.ModuleList(
            ConvNormLayer_fuse(ch_in, ch_out, 3, 2, act="relu")
            for ch_in, ch_out in zip([3, *channels[:-1]], channels)
        )
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convs:
            x = layer(x)
        return self.proj(x)


class RopeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, dim // self.num_heads
        )
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(2))
        sin, cos = rope
        # The register token precedes the spatial tokens and has no rotary embedding.
        query = torch.cat(
            (query[:, :, :1], rope_apply(query[:, :, 1:], sin, cos)), dim=2
        )
        key = torch.cat((key[:, :, :1], rope_apply(key[:, :, 1:], sin, cos)), dim=2)
        output = F.scaled_dot_product_attention(query, key, value)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, dim))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class LoopedBlock(nn.Module):
    """One shared Transformer block and an independent Loop Adapter for each layer in the group."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float,
        adapter_hidden_dim: int,
        group_size: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RopeAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim)
        self.loop_adapters = nn.ModuleList(
            LoopAdapter(dim, adapter_hidden_dim) for _ in range(group_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        layer_in_group: int,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return self.loop_adapters[layer_in_group](x)


class LoopedViT(nn.Module):
    """Apply the shared Transformer block of each group recurrently, average selected token maps and project to three scales."""

    def __init__(
        self,
        *,
        group_boundaries: Sequence[Sequence[int]],
        interaction_indexes: Sequence[int],
        embed_dim: int,
        num_heads: int,
        ffn_ratio: float,
        proj_dim: int | None,
        adapter_hidden_dim: int,
    ):
        super().__init__()
        self.patch_embed = ConvPyramidPatchEmbed(embed_dim)
        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.rope_embed = RopePositionEmbedding(embed_dim, num_heads=num_heads)
        self.blocks = nn.ModuleList()
        depth = 0
        for start, end in group_boundaries:
            if start != depth or end <= start:
                raise ValueError(
                    f"Invalid contiguous group [{start}, {end}) after {depth} layers"
                )
            self.blocks.append(
                LoopedBlock(
                    embed_dim,
                    num_heads,
                    ffn_ratio,
                    adapter_hidden_dim,
                    end - start,
                )
            )
            depth = end
        indexes = tuple(interaction_indexes)
        if (
            not indexes
            or tuple(sorted(set(indexes))) != indexes
            or indexes[0] < 0
            or indexes[-1] >= depth
        ):
            raise ValueError(
                "interaction_indexes must be sorted, unique indexes within the loop depth"
            )
        self.interaction_indexes = indexes
        # S and M project only the coarsest map. L and X project all three maps.
        dimensions = [embed_dim] if proj_dim is None else [proj_dim] * 3
        self.projector = nn.ModuleList(
            ConvNormLayer_fuse(embed_dim, width, 1, 1) for width in dimensions
        )

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        if images.ndim != 4 or any(size % 32 for size in images.shape[-2:]):
            raise ValueError(
                "Expected a batch of images with height and width divisible by 32"
            )
        patches = self.patch_embed(images)
        batch, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        x = torch.cat((self.register_token.expand(batch, -1, -1), tokens), dim=1)
        sin, cos = self.rope_embed(H=height, W=width)
        rope = (
            sin.to(dtype=x.dtype)[None, None],
            cos.to(dtype=x.dtype)[None, None],
        )
        selected = []
        index = 0
        for block in self.blocks:
            for layer_in_group in range(len(block.loop_adapters)):
                x = block(x, rope, layer_in_group)
                if index in self.interaction_indexes:
                    selected.append(x[:, 1:])
                index += 1
        fused = torch.stack(selected).mean(dim=0).transpose(1, 2).contiguous()
        fused = fused.view(batch, channels, height, width)
        features = [
            F.interpolate(fused, size=size, mode="bilinear", align_corners=False)
            for size in (
                (height * 2, width * 2),
                (height, width),
                (height // 2, width // 2),
            )
        ]
        if len(self.projector) == 1:
            features[-1] = self.projector[0](features[-1])
        else:
            features = [
                project(feature) for project, feature in zip(self.projector, features)
            ]
        return features
