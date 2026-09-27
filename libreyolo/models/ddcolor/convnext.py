"""ConvNeXt backbone used by the DDColor encoder.

Adapted for inference from ``piddnad/DDColor`` at commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13`` (Apache-2.0), whose backbone is
adapted from ``facebookresearch/ConvNeXt`` at commit
``048efcea897d999aed302f2639b6270aedf8d4c8`` (MIT). Module names intentionally
match DDColor's official checkpoint. This file was modified to remove the
optional timm runtime dependency; the shipped configurations use zero drop
path and PyTorch provides the required truncated-normal initializer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth, inactive in released DDColor configs."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    """ConvNeXt depthwise-convolution block with channels-last MLP."""

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class ConvNeXt(nn.Module):
    """DDColor's checkpoint-compatible ConvNeXt feature encoder."""

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("DDColor ConvNeXt expects four stages.")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(
                        dims[index],
                        eps=1e-6,
                        data_format="channels_first",
                    ),
                    nn.Conv2d(
                        dims[index],
                        dims[index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )

        rates = [
            value.item() for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        self.stages = nn.ModuleList()
        cursor = 0
        for stage_index in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(
                            dim=int(dims[stage_index]),
                            drop_path=rates[cursor + block_index],
                            layer_scale_init_value=layer_scale_init_value,
                        )
                        for block_index in range(int(depths[stage_index]))
                    ]
                )
            )
            cursor += int(depths[stage_index])

        # DDColor consumes these explicit normalization layers as decoder skips.
        for index in range(4):
            self.add_module(
                f"norm{index}",
                LayerNorm(
                    dims[index],
                    eps=1e-6,
                    data_format="channels_first",
                ),
            )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        intermediates = []
        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
            intermediates.append(getattr(self, f"norm{index}")(x))
        pooled = self.norm(x.mean((-2, -1)))
        if return_intermediates:
            return pooled, tuple(intermediates)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class LayerNorm(nn.Module):
    """LayerNorm supporting the two data layouts used by ConvNeXt."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"Unsupported LayerNorm data format: {data_format!r}.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = float(eps)
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


__all__ = ["Block", "ConvNeXt", "LayerNorm"]
