"""Native QuickSRNet modules for compact super-resolution.

The architecture is adapted from Qualcomm's QuickSRNet implementation at
``quic/aimet-model-zoo`` (BSD-3-Clause). Module names intentionally match the
official checkpoint so its tensor state can be loaded without remapping. See
the family ``NOTICE`` for pinned provenance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuickSRNet(nn.Module):
    """QuickSRNet convolutional backbone followed by pixel shuffle.

    Args:
        scale: integer output upscale factor.
        num_channels: feature width of the convolutional trunk.
        num_intermediate_layers: number of 3x3 feature-to-feature convolutions.
    """

    def __init__(
        self,
        scale: int = 2,
        num_channels: int = 32,
        num_intermediate_layers: int = 5,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be positive, got {scale}.")

        layers: list[nn.Module] = [
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.Hardtanh(min_val=0.0, max_val=1.0),
        ]
        for _ in range(num_intermediate_layers):
            layers.extend(
                (
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Hardtanh(min_val=0.0, max_val=1.0),
                )
            )

        self.cnn = nn.Sequential(*layers)
        self.conv_last = nn.Conv2d(
            num_channels,
            3 * scale * scale,
            kernel_size=3,
            padding=1,
        )
        self.clip_output = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.depth_to_space = nn.PixelShuffle(scale)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        restored = self.cnn(image)
        restored = self.clip_output(self.conv_last(restored))
        return self.depth_to_space(restored)


__all__ = ["QuickSRNet"]
