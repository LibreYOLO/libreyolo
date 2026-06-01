"""Native PyTorch LibreFOMO architecture."""

from __future__ import annotations

import math
from typing import ClassVar, Tuple

import torch
import torch.nn as nn


CONFIGS = {
    "s": {"alpha": 0.35, "imgsz": 96, "head_in_channels": 96},
    "m": {"alpha": 0.50, "imgsz": 192, "head_in_channels": 96},
    "l": {"alpha": 1.00, "imgsz": 224, "head_in_channels": 192},
}


def _make_divisible(v: float, divisor: int = 8, min_value=None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _same_pad_1d(input_size: int, kernel: int, stride: int, dilation: int = 1):
    out_size = math.ceil(float(input_size) / float(stride))
    effective_kernel = (kernel - 1) * dilation + 1
    total_pad = max((out_size - 1) * stride + effective_kernel - input_size, 0)
    before = total_pad // 2
    return before, total_pad - before


def _same_pad_2d(input_hw: Tuple[int, int], kernel_hw, stride_hw, dilation_hw=(1, 1)):
    ih, iw = input_hw
    kh, kw = kernel_hw
    sh, sw = stride_hw
    dh, dw = dilation_hw
    top, bottom = _same_pad_1d(ih, kh, sh, dh)
    left, right = _same_pad_1d(iw, kw, sw, dw)
    return left, right, top, bottom


class StaticSamePadConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        groups: int = 1,
        bias: bool = False,
        input_hw: Tuple[int, int] | None = None,
    ):
        super().__init__()
        if input_hw is None:
            raise ValueError("StaticSamePadConv2d requires fixed input_hw")
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.pad = nn.ZeroPad2d(_same_pad_2d(input_hw, kernel_size, stride))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, input_hw):
        super().__init__(
            StaticSamePadConv2d(in_channels, out_channels, kernel_size, stride, input_hw=input_hw),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.ReLU6(inplace=True),
        )


class DepthwiseConvBNReLU6(nn.Sequential):
    def __init__(self, channels, kernel_size, stride, input_hw):
        super().__init__(
            StaticSamePadConv2d(
                channels,
                channels,
                kernel_size,
                stride,
                groups=channels,
                input_hw=input_hw,
            ),
            nn.BatchNorm2d(channels, eps=1e-3),
            nn.ReLU6(inplace=True),
        )


class ProjectConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, input_hw, use_residual):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = use_residual
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_channels, hidden_channels, 1, 1, input_hw))
        layers.append(DepthwiseConvBNReLU6(hidden_channels, 3, stride, input_hw))
        layers.append(ProjectConvBN(hidden_channels, out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class LibreFOMOBackbone(nn.Module):
    def __init__(self, size: str):
        super().__init__()
        if size not in CONFIGS:
            raise ValueError(f"Unsupported LibreFOMO size: {size!r}")
        cfg = CONFIGS[size]
        alpha = cfg["alpha"]
        imgsz = cfg["imgsz"]
        c0 = _make_divisible(32 * alpha, 8)
        c1 = _make_divisible(16 * alpha, 8)
        c2 = _make_divisible(24 * alpha, 8)
        c3 = _make_divisible(32 * alpha, 8)

        self.conv1 = ConvBNReLU6(3, c0, 3, 2, (imgsz, imgsz))
        hw = math.ceil(imgsz / 2)
        self.expanded_conv = InvertedResidual(c0, c1, 1, 1, (hw, hw), False)
        self.block_1 = InvertedResidual(c1, c2, 2, 6, (hw, hw), False)
        hw = math.ceil(hw / 2)
        self.block_2 = InvertedResidual(c2, c2, 1, 6, (hw, hw), True)
        self.block_3 = InvertedResidual(c2, c3, 2, 6, (hw, hw), False)
        hw = math.ceil(hw / 2)
        self.block_4 = InvertedResidual(c3, c3, 1, 6, (hw, hw), True)
        self.block_5 = InvertedResidual(c3, c3, 1, 6, (hw, hw), True)
        self.block_6_expand = ConvBNReLU6(c3, int(round(c3 * 6)), 1, 1, (hw, hw))

    def forward(self, x):
        x = self.conv1(x)
        x = self.expanded_conv(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return self.block_6_expand(x)


class LibreFOMOModel(nn.Module):
    CONFIGS: ClassVar[dict] = CONFIGS

    def __init__(self, size: str = "m", nc: int = 1, head_channels: int | None = None):
        super().__init__()
        if size not in CONFIGS:
            raise ValueError(f"Unsupported LibreFOMO size: {size!r}")
        self.size = size
        self.nc = nc
        self.head_channels = head_channels if head_channels is not None else nc + 1
        self.imgsz = CONFIGS[size]["imgsz"]
        self.backbone = LibreFOMOBackbone(size)
        self.head = nn.Conv2d(CONFIGS[size]["head_in_channels"], self.head_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def detect_size_from_state_dict(state_dict: dict) -> str | None:
    weight = state_dict.get("head.weight")
    if weight is None:
        return None
    in_ch = int(weight.shape[1])
    if in_ch == 192:
        return "l"
    if in_ch == 96:
        conv1 = state_dict.get("backbone.conv1.1.weight")
        if conv1 is not None:
            return {16: "s", 24: "m"}.get(int(conv1.shape[0]))
    return None
