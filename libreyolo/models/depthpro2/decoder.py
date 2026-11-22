#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DPTDecoder(nn.Module):
    """Decoder fusing a 4-level feature pyramid from low to high resolution."""

    def __init__(self, dims_reassemble: Sequence[int], dim_decoder: int):
        """Initialize the decoder.

        Args:
            dims_reassemble: Channel counts of the reassembled pyramid, highest resolution first.
            dim_decoder: Channel count of every decoder stage.
        """
        super().__init__()
        self.dim_out = dim_decoder
        num_levels = len(dims_reassemble)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(dim, dim_decoder, kernel_size=3, padding=1, bias=False)
                for dim in dims_reassemble
            ]
        )
        self.fusions = nn.ModuleList(
            [
                FeatureFusionBlock2d(dim_decoder, with_skip=i < num_levels - 1)
                for i in range(num_levels)
            ]
        )

    def forward(self, encodings: Sequence[torch.Tensor]) -> torch.Tensor:
        """Decode the pyramid, highest resolution first, into features at 1/2 of the input."""
        features = self.fusions[-1](self.convs[-1](encodings[-1]))
        for i in range(len(encodings) - 2, -1, -1):
            features = self.fusions[i](features, self.convs[i](encodings[i]))
        return features


class ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module):
        super().__init__()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class FeatureFusionBlock2d(nn.Module):
    """DPT feature fusion: add the skip features, refine, and upsample 2x."""

    def __init__(self, dim: int, with_skip: bool):
        """Initialize the feature fusion block.

        Args:
            dim: Channel count of the input, the skip input and the output.
            with_skip: Whether the block fuses a same-resolution skip input.
        """
        super().__init__()
        if with_skip:
            self.resnet1 = self._residual_block(dim)
        self.resnet2 = self._residual_block(dim)
        self.deconv = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=False)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.resnet1(skip)
        x = self.resnet2(x)
        x = self.deconv(x)
        return self.out_conv(x)

    @staticmethod
    def _residual_block(dim: int) -> ResidualBlock:
        return ResidualBlock(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
        )


def create_inverse_depth_head(dim_in: int) -> nn.Sequential:
    """Create the head mapping 1/2-resolution decoder features to full-resolution inverse depth."""
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_in // 2, kernel_size=3, padding=1),
        nn.ConvTranspose2d(dim_in // 2, dim_in // 2, kernel_size=2, stride=2),
        nn.Conv2d(dim_in // 2, 32, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, kernel_size=1),
        nn.ReLU(),
    )
