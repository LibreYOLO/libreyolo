# Copyright (c) OpenMMLab. All rights reserved.
"""FCOS3D ResNet101-DCN inference network.

Adapted from OpenMMLab's Apache-2.0 implementations; see NOTICE for pins.
LibreYOLO removes framework registries and uses torchvision's deformable op.
State-dict names match the official nuScenes checkpoint.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


class ModulatedConv(nn.Conv2d):
    """DCNv2 with a learned offset/mask convolution."""

    def __init__(self, channels, *, bias=False):
        super().__init__(channels, channels, 3, padding=1, bias=bias)
        self.conv_offset = nn.Conv2d(channels, 27, 3, padding=1)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        return deform_conv2d(
            x,
            offset_mask[:, :18],
            self.weight,
            self.bias,
            padding=(1, 1),
            mask=offset_mask[:, 18:].sigmoid(),
        )


class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, deform=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = (
            ModulatedConv(channels)
            if deform
            else nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.downsample = None
        if stride != 1 or in_channels != channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.relu(self.bn3(self.conv3(x)) + identity)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        in_channels = 64
        for i, (channels, count) in enumerate(zip((64, 128, 256, 512), (3, 4, 23, 3))):
            blocks = []
            for j in range(count):
                blocks.append(
                    Bottleneck(in_channels, channels, 2 if i and j == 0 else 1, i >= 2)
                )
                in_channels = channels * 4
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, stride=2, padding=1)
        outputs = []
        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)
            outputs.append(x)
        return outputs


class ConvModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, norm=False, deform=False
    ):
        super().__init__()
        self.conv = (
            ModulatedConv(in_channels, bias=True)
            if deform
            else nn.Conv2d(
                in_channels, out_channels, kernel, stride=stride, padding=kernel // 2
            )
        )
        if norm:
            self.gn = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(self.gn(x)) if hasattr(self, "gn") else x


class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [ConvModule(c, 256, 1) for c in (512, 1024, 2048)]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvModule(256, 256) for _ in range(3)]
            + [ConvModule(256, 256, stride=2) for _ in range(2)]
        )

    def forward(self, features):
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, features[1:])]
        for i in (2, 1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        outputs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]
        outputs.append(self.fpn_convs[3](outputs[-1]))
        outputs.append(self.fpn_convs[4](F.relu(outputs[-1])))
        return outputs


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


def _branch(out_channels=256):
    return nn.ModuleList([ConvModule(256, out_channels, norm=True)])


def _run(layers, x):
    if layers is not None:
        for layer in layers:
            x = layer(x)
    return x


class Head(nn.Module):
    strides = (8, 16, 32, 64, 128)

    def __init__(self):
        super().__init__()
        self.cls_convs = nn.ModuleList(
            [ConvModule(256, 256, norm=True, deform=i == 1) for i in range(2)]
        )
        self.reg_convs = nn.ModuleList(
            [ConvModule(256, 256, norm=True, deform=i == 1) for i in range(2)]
        )
        self.conv_cls_prev = _branch()
        self.conv_cls = nn.Conv2d(256, 10, 1)
        self.conv_reg_prevs = nn.ModuleList([_branch() for _ in range(4)] + [None])
        self.conv_regs = nn.ModuleList([nn.Conv2d(256, n, 1) for n in (2, 1, 3, 1, 2)])
        self.conv_dir_cls_prev = _branch()
        self.conv_dir_cls = nn.Conv2d(256, 2, 1)
        self.conv_attr_prev = _branch()
        self.conv_attr = nn.Conv2d(256, 9, 1)
        self.conv_centerness_prev = _branch(64)
        self.conv_centerness = nn.Conv2d(64, 1, 1)
        self.scales = nn.ModuleList(
            [nn.ModuleList([Scale() for _ in range(3)]) for _ in self.strides]
        )

    def forward(self, features):
        outputs = []
        for x, stride, scales in zip(features, self.strides, self.scales):
            cls = _run(self.cls_convs, x)
            reg = _run(self.reg_convs, x)
            bbox = torch.cat(
                [
                    conv(_run(prev, reg))
                    for prev, conv in zip(self.conv_reg_prevs, self.conv_regs)
                ],
                1,
            )
            bbox = torch.cat(
                (
                    scales[0](bbox[:, :2]).float() * stride,
                    scales[1](bbox[:, 2:3]).float().exp(),
                    scales[2](bbox[:, 3:6]).float().exp(),
                    bbox[:, 6:],
                ),
                1,
            )
            outputs.append(
                (
                    self.conv_cls(_run(self.conv_cls_prev, cls)),
                    bbox,
                    self.conv_dir_cls(_run(self.conv_dir_cls_prev, reg)),
                    self.conv_attr(_run(self.conv_attr_prev, cls)),
                    self.conv_centerness(_run(self.conv_centerness_prev, reg)),
                )
            )
        return tuple(map(list, zip(*outputs)))


class FCOS3DNetwork(nn.Module):
    """Inference architecture for the official R101 nuScenes checkpoint."""

    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = FPN()
        self.bbox_head = Head()

    def forward(self, x):
        return self.bbox_head(self.neck(self.backbone(x)))
