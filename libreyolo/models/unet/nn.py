"""Native mmseg UNet-S5-D16 + FCN head, with upstream state-dict keys.

Derived from open-mmlab/mmsegmentation ``mmseg/models/backbones/unet.py``,
``mmseg/models/utils/up_conv_block.py``, and ``mmseg/models/decode_heads/fcn_head.py``
at commit ``b040e147adfa`` (Apache-2.0). Module names follow upstream so the
official Cityscapes checkpoint loads strictly. This is the same-padded 2D
S5-D16 graph with an FCN head, not the 2015 Caffe valid-convolution U-Net.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

STRIDE = 16
IGNORE_INDEX = 255

# ImageNet mean/std as stored in the mmseg SegDataPreProcessor (0-255 RGB).
_PIXEL_MEAN = (123.675, 116.28, 103.53)
_PIXEL_STD = (58.395, 57.12, 57.375)

# ``imgsz`` is the evaluation canvas: the mmseg Cityscapes test pipeline is
# ``Resize(scale=(2048, 1024), keep_ratio=True)`` + ``test_cfg(mode='whole')``,
# so the published 69.10 mIoU is measured on full 1024x2048 frames. The
# ``512x1024`` in the checkpoint name is the training crop, taken from the
# source frame rescaled by a factor in ``rescale_range`` (mmseg RandomResize
# ratio_range 0.5..2.0 of the 2048x1024 scale, then RandomCrop).
SIZE_CONFIGS = {
    "s": {
        "base_channels": 64,
        "imgsz": (1024, 2048),
        "train_crop": (512, 1024),
        "rescale_range": (0.5, 2.0),
    },
}


class ConvBNReLU(nn.Module):
    """mmcv ``ConvModule`` with conv / BN / ReLU and ``.conv`` / ``.bn`` names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class BasicConvBlock(nn.Module):
    """Two 3x3 convs; optional stride on the first layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_convs):
            layers.append(
                ConvBNReLU(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    stride=stride if index == 0 else 1,
                    dilation=1 if index == 0 else dilation,
                    padding=1 if index == 0 else dilation,
                )
            )
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class _BilinearUpsample(nn.Module):
    """Parameter-free 2x bilinear upsample (mmseg ``Upsample`` slot)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class InterpConv(nn.Module):
    """Upsample, then 1x1 conv. Sequential index 1 is the conv, matching upstream keys."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.interp_upsample = nn.Sequential(
            _BilinearUpsample(),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.interp_upsample(x)


class UpConvBlock(nn.Module):
    """Upsample the deep map, concat the skip, then two 3x3 convs."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_convs: int = 2,
    ) -> None:
        super().__init__()
        self.upsample = InterpConv(in_channels, skip_channels)
        self.conv_block = BasicConvBlock(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
        )

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(torch.cat([skip, self.upsample(x)], dim=1))


class UNetBackbone(nn.Module):
    """Five-stage encoder / four-stage decoder (S5-D16)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        num_stages = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        stage_in = in_channels
        for index in range(num_stages):
            out_channels = base_channels * (2**index)
            block: list[nn.Module] = []
            if index != 0:
                block.append(nn.MaxPool2d(kernel_size=2))
                self.decoder.append(
                    UpConvBlock(
                        in_channels=base_channels * (2**index),
                        skip_channels=base_channels * (2 ** (index - 1)),
                        out_channels=base_channels * (2 ** (index - 1)),
                    )
                )
            block.append(BasicConvBlock(stage_in, out_channels, stride=1))
            self.encoder.append(nn.Sequential(*block))
            stage_in = out_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        enc_outs: list[torch.Tensor] = []
        for encoder in self.encoder:
            x = encoder(x)
            enc_outs.append(x)
        dec_outs = [x]
        for index in reversed(range(len(self.decoder))):
            x = self.decoder[index](enc_outs[index], x)
            dec_outs.append(x)
        return dec_outs


class FCNHead(nn.Module):
    """Single-conv FCN head used by the mmseg UNet Cityscapes config."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        in_index: int,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_index = int(in_index)
        self.convs = nn.Sequential(
            ConvBNReLU(in_channels, channels, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        feats = self.convs(inputs[self.in_index])
        if self.dropout is not None:
            feats = self.dropout(feats)
        return self.conv_seg(feats)


class LibreUNetNet(nn.Module):
    """UNet-S5-D16 backbone plus primary and auxiliary FCN heads."""

    IGNORE_INDEX = IGNORE_INDEX

    def __init__(
        self,
        size: str = "s",
        num_classes: int = 19,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown U-Net size {size!r}; choose from {tuple(SIZE_CONFIGS)}.")
        self.size = size
        self.num_classes = int(num_classes)
        self.normalize_input = bool(normalize_input)
        base = int(SIZE_CONFIGS[size]["base_channels"])
        self.backbone = UNetBackbone(in_channels=3, base_channels=base)
        self.decode_head = FCNHead(
            in_channels=base,
            channels=base,
            num_classes=self.num_classes,
            in_index=4,
        )
        self.auxiliary_head = FCNHead(
            in_channels=base * 2,
            channels=base,
            num_classes=self.num_classes,
            in_index=3,
        )
        mean = torch.tensor(_PIXEL_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_PIXEL_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_input:
            return x
        # Incoming tensors are RGB floats in [0, 1]; mmseg subtracted 0-255 stats.
        return (x * 255.0 - self._mean) / self._std

    def replace_num_classes(self, num_classes: int) -> None:
        num_classes = int(num_classes)
        self.num_classes = num_classes
        in_main = self.decode_head.conv_seg.in_channels
        in_aux = self.auxiliary_head.conv_seg.in_channels
        self.decode_head.conv_seg = nn.Conv2d(in_main, num_classes, kernel_size=1)
        self.auxiliary_head.conv_seg = nn.Conv2d(in_aux, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_shape = x.shape[-2:]
        feats = self.backbone(self._normalize(x))
        main = F.interpolate(
            self.decode_head(feats),
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )
        if self.training:
            aux = F.interpolate(
                self.auxiliary_head(feats),
                size=input_shape,
                mode="bilinear",
                align_corners=False,
            )
            return main, aux
        return main


__all__ = [
    "IGNORE_INDEX",
    "SIZE_CONFIGS",
    "STRIDE",
    "FCNHead",
    "LibreUNetNet",
    "UNetBackbone",
]
