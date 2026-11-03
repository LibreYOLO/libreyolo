# Copyright (c) 2025, Princeton Vision & Learning Lab. BSD-3-Clause, see LICENSE.
# Adapted from WAFT model/backbone/head.py (the DPT head on frozen encoders,
# itself derived from DPT and Depth Anything V2). ``nn.quantized.FloatFunctional``
# adds are plain additions; they hold no parameters.
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, bn=False, align_corners=True, size=None):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn, align_corners=True, size=size)


class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False,
                 out_channels=(256, 512, 1024, 1024), lvl=-2):
        super().__init__()
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels, oc, 1, 1, 0) for oc in out_channels]
        )
        resize = []
        for i in range(len(self.projects)):
            if i + lvl < 0:
                k = 2 ** (-i - lvl)
                resize.append(nn.ConvTranspose2d(out_channels[i], out_channels[i], k, k, 0))
            else:
                k = 2 ** (i + lvl)
                resize.append(nn.Conv2d(out_channels[i], out_channels[i], k, k, 0))
        self.resize_layers = nn.ModuleList(resize)
        self.scratch = nn.ModuleList(
            [nn.Conv2d(oc, features, 3, 1, 1, bias=False) for oc in out_channels]
        )
        self.refine = nn.ModuleList(
            [_make_fusion_block(features, use_bn) for _ in range(len(out_channels))]
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        for i in range(1, len(out_rn) + 1):
            if i == 1:
                out_rn[-i] = self.refine[-i](out_rn[-i], size=out_rn[-i].shape[2:])
            else:
                up = F.interpolate(out_rn[-i + 1], scale_factor=2, mode="bilinear", align_corners=True)
                out_rn[-i] = self.refine[-i](out_rn[-i], up, size=out_rn[-i].shape[2:])
        return out_rn
