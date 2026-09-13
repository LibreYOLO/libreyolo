"""ConvNeXt V2 classifiers, adapted from Meta's MIT implementation.

Source: facebookresearch/ConvNeXt-V2 at
2553895753323c6fe0b2bf390683f5ea358a42b9 (models/convnextv2.py and
the dense LayerNorm/GRN in models/utils.py). Copyright (c) Meta Platforms,
Inc. and affiliates. See NOTICE. Official tensor names are preserved.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCH_DEFS = {
    "atto": ((2, 2, 6, 2), (40, 80, 160, 320)),
    "femto": ((2, 2, 6, 2), (48, 96, 192, 384)),
    "pico": ((2, 2, 6, 2), (64, 128, 256, 512)),
    "n": ((2, 2, 8, 2), (80, 160, 320, 640)),
    "t": ((3, 3, 9, 3), (96, 192, 384, 768)),
    "b": ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "l": ((3, 3, 27, 3), (192, 384, 768, 1536)),
    "h": ((3, 3, 27, 3), (352, 704, 1408, 2816)),
}


class LayerNorm(nn.Module):
    """Upstream channel-first arithmetic and channel-last LayerNorm."""

    def __init__(self, dim: int, channels_first: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.channels_first = channels_first
        self.normalized_shape = (dim,)

    def forward(self, x):
        if not self.channels_first:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, 1e-6)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + 1e-6)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization on NHWC activations."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        response = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * normalized) + self.beta + x


class Block(nn.Module):
    """Depthwise convolution and channel MLP with GRN."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return residual + x.permute(0, 3, 1, 2)


class ConvNeXtV2(nn.Module):
    """Dense supervised classifier; FCMAE pretraining is not implemented."""

    def __init__(self, size: str = "atto", num_classes: int = 1000):
        super().__init__()
        depths, dims = ARCH_DEFS[size]
        self.num_features = dims[-1]
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm(dims[0], True))
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], True), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)
            ))
        self.stages = nn.ModuleList([
            nn.Sequential(*(Block(dim) for _ in range(depth)))
            for depth, dim in zip(depths, dims)
        ])
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward_features(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return self.norm(x.mean([-2, -1]))

    def reset_classifier(self, num_classes: int):
        weight = self.head.weight
        self.head = nn.Linear(self.num_features, num_classes).to(weight)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.forward_features(x))
