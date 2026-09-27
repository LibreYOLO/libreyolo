"""UNet decoding utilities used by DDColor.

Adapted from the Apache-2.0 DDColor source pinned in this family's ``NOTICE``.
The dynamic-UNet lineage is ColorFormer (permissive three-clause license) and
fastai (Apache-2.0). Only DDColor's inference path is included. In particular,
this file contains none of ColorFormer's unrelated DFDNet/CC-NC components.
Modified for static channel bookkeeping so construction does not execute a
large random image through the encoder; parameter names and numerics are
unchanged.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn


NormType = Enum("NormType", "Batch BatchZero Weight Spectral")


def batchnorm_2d(channels: int, norm_type: NormType = NormType.Batch) -> nn.BatchNorm2d:
    layer = nn.BatchNorm2d(channels)
    with torch.no_grad():
        layer.bias.fill_(1e-3)
        layer.weight.fill_(0.0 if norm_type == NormType.BatchZero else 1.0)
    return layer


def init_default(
    module: nn.Module,
    initializer=nn.init.kaiming_normal_,
) -> nn.Module:
    if initializer and hasattr(module, "weight"):
        initializer(module.weight)
    bias = getattr(module, "bias", None)
    if bias is not None and hasattr(bias, "data"):
        bias.data.fill_(0.0)
    return module


def icnr(
    weight: torch.Tensor,
    scale: int = 2,
    initializer=nn.init.kaiming_normal_,
) -> None:
    out_channels, in_channels, height, width = weight.shape
    reduced_channels = int(out_channels / (scale**2))
    kernel = initializer(
        torch.zeros(reduced_channels, in_channels, height, width)
    ).transpose(0, 1)
    kernel = kernel.contiguous().view(reduced_channels, in_channels, -1)
    kernel = kernel.repeat(1, 1, scale**2)
    kernel = (
        kernel.contiguous()
        .view(
            in_channels,
            out_channels,
            height,
            width,
        )
        .transpose(0, 1)
    )
    weight.data.copy_(kernel)


def custom_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    bias: bool | None = None,
    norm_type: NormType = NormType.Batch,
    use_activation: bool = True,
    initializer=nn.init.kaiming_normal_,
    extra_bn: bool = False,
) -> nn.Sequential:
    """Build DDColor's convolution, activation, and optional BatchNorm stack."""

    if padding is None:
        padding = (kernel_size - 1) // 2
    use_bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn
    if bias is None:
        bias = not use_bn
    convolution = init_default(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        initializer,
    )
    if norm_type == NormType.Weight:
        convolution = nn.utils.weight_norm(convolution)
    elif norm_type == NormType.Spectral:
        convolution = nn.utils.spectral_norm(convolution)

    layers: list[nn.Module] = [convolution]
    if use_activation:
        layers.append(nn.ReLU(True))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class CustomPixelShuffleICNR(nn.Module):
    """ICNR convolution followed by pixel shuffle and optional blur."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 2,
        blur: bool = True,
        norm_type: NormType = NormType.Spectral,
        extra_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = custom_conv_layer(
            in_channels,
            out_channels * (scale**2),
            kernel_size=1,
            use_activation=False,
            norm_type=norm_type,
            extra_bn=extra_bn,
        )
        icnr(self.conv[0].weight, scale=scale)
        self.shuf = nn.PixelShuffle(scale)
        self.do_blur = bool(blur)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


class UnetBlockWide(nn.Module):
    """One DDColor wide UNet skip/upsampling block."""

    def __init__(
        self,
        up_in_channels: int,
        skip_channels: int,
        out_channels: int,
        blur: bool = False,
        norm_type: NormType = NormType.Spectral,
    ) -> None:
        super().__init__()
        self.shuf = CustomPixelShuffleICNR(
            up_in_channels,
            out_channels,
            blur=blur,
            norm_type=norm_type,
            extra_bn=True,
        )
        self.bn = batchnorm_2d(skip_channels)
        self.conv = custom_conv_layer(
            out_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            extra_bn=True,
        )
        self.relu = nn.ReLU()

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        merged = self.relu(torch.cat((up_out, self.bn(skip)), dim=1))
        return self.conv(merged)


__all__ = [
    "CustomPixelShuffleICNR",
    "NormType",
    "UnetBlockWide",
    "custom_conv_layer",
]
