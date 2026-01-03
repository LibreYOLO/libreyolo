"""
Neural network architecture for LibreYOLO v7.

All layers are self-contained within this module.
YOLOv7 uses anchor-based detection with ImplicitA/ImplicitM layers.

Architecture matches the official YOLO v7.yaml config exactly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """
    Reparameterizable Convolution block for YOLOv7.

    During training: 3x3 conv + 1x1 conv
    During inference: Single fused 3x3 conv

    Matches the official YOLO RepConv with conv1 and conv2 naming.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, deploy=False):
        super().__init__()
        assert k == 3 and d == 1
        self.deploy = deploy
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # Use Conv wrappers with original naming (conv1, conv2)
        self.conv1 = Conv(c1, c2, k, s, g=g, act=False)  # 3x3 branch
        self.conv2 = Conv(c1, c2, 1, s, g=g, act=False)  # 1x1 branch

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(x))

        return self.act(self.conv1(x) + self.conv2(x))

    def fuse_convs(self):
        """Fuse parallel convolutions into single conv."""
        if hasattr(self, 'rbr_reparam'):
            return

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])

        self.rbr_reparam = nn.Conv2d(
            self.c1, self.c2, 3, 1, 1, groups=self.g, bias=True
        )
        self.rbr_reparam.weight.data = kernel3x3 + kernel1x1
        self.rbr_reparam.bias.data = bias3x3 + bias1x1

        for para in self.parameters():
            para.detach_()

        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def _fuse_bn_tensor(self, branch):
        """Fuse batch norm into conv weights."""
        if branch is None:
            return 0, 0

        # branch is a Conv wrapper with .conv and .bn
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class MP(nn.Module):
    """MaxPool with stride 2."""

    def __init__(self, k=2):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k, padding=0)

    def forward(self, x):
        return self.m(x)


class SPPCSPC(nn.Module):
    """SPP with CSP for YOLOv7.

    Matches the official YOLO SPPCSPConv implementation exactly.
    """

    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)  # neck_channels
        # Pre-processing branch: 3 convs
        self.pre_conv = nn.Sequential(
            Conv(c1, c_, 1, 1),
            Conv(c_, c_, 3, 1),
            Conv(c_, c_, 1, 1),
        )
        # Short-cut branch
        self.short_conv = Conv(c1, c_, 1, 1)
        # SPP pooling
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # Post SPP: 2 convs (input is 4*c_ from concat of [pre_conv_out, pool1, pool2, pool3])
        self.post_conv = nn.Sequential(
            Conv(4 * c_, c_, 1, 1),
            Conv(c_, c_, 3, 1),
        )
        # Final merge
        self.merge_conv = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        # Pre-conv branch with SPP
        features = [self.pre_conv(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        features = torch.cat(features, dim=1)
        y1 = self.post_conv(features)
        # Short-cut branch
        y2 = self.short_conv(x)
        # Merge
        return self.merge_conv(torch.cat((y1, y2), dim=1))


class ImplicitA(nn.Module):
    """Implicit addition layer for YOLOv7."""

    def __init__(self, channel, mean=0., std=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return x + self.implicit


class ImplicitM(nn.Module):
    """Implicit multiplication layer for YOLOv7."""

    def __init__(self, channel, mean=1., std=0.02):
        super().__init__()
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=mean, std=std)

    def forward(self, x):
        return x * self.implicit


class IDetect(nn.Module):
    """
    Anchor-based detection head with implicit layers for YOLOv7.
    """

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]

        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.register_buffer('stride', torch.tensor([8., 16., 32.]))

        # Detection convs with implicit layers
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

        self._init_biases()

    def _init_biases(self):
        for mi, s in zip(self.m, [8, 16, 32]):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.im[i](self.m[i](self.ia[i](x[i])))
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].split((2, 2, self.nc + 1), 4)
                xy = (xy.sigmoid() * 2 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf.sigmoid()), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            return x
        return torch.cat(z, 1), x

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        yv, xv = torch.meshgrid(torch.arange(ny, device=d, dtype=t),
                                torch.arange(nx, device=d, dtype=t), indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(1, self.na, ny, nx, 2)
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(1, self.na, ny, nx, 2)
        return grid, anchor_grid


# =============================================================================
# YOLOv7 Base Architecture (matches v7.yaml exactly)
# =============================================================================

V7_ANCHORS = [
    [12, 16, 19, 36, 40, 28],
    [36, 75, 76, 55, 72, 146],
    [142, 110, 192, 243, 459, 401],
]


class Backbone7(nn.Module):
    """YOLOv7 Backbone matching the official v7.yaml structure exactly.

    Architecture (layer indices from v7.yaml):
    - 0-3: Stem convs
    - 4-11: E-ELAN block 1
    - 12-24: Transition + E-ELAN block 2 (B3)
    - 25-37: Transition + E-ELAN block 3 (B4)
    - 38-50: Transition + E-ELAN block 4 (B5)
    """

    def __init__(self):
        super().__init__()

        # Stem: layers 0-3
        self.conv0 = Conv(3, 32, 3, 1)        # 0: 3->32
        self.conv1 = Conv(32, 64, 3, 2)       # 1: 32->64, stride 2
        self.conv2 = Conv(64, 64, 3, 1)       # 2: 64->64
        self.conv3 = Conv(64, 128, 3, 2)      # 3: 64->128, stride 2

        # E-ELAN 1: layers 4-11
        self.conv4 = Conv(128, 64, 1, 1)      # 4: 128->64
        self.conv5 = Conv(128, 64, 1, 1)      # 5: 128->64 (from 3)
        self.conv6 = Conv(64, 64, 3, 1)       # 6: 64->64
        self.conv7 = Conv(64, 64, 3, 1)       # 7: 64->64
        self.conv8 = Conv(64, 64, 3, 1)       # 8: 64->64
        self.conv9 = Conv(64, 64, 3, 1)       # 9: 64->64
        # 10: Concat [9,7,5,4] = 256
        self.conv11 = Conv(256, 256, 1, 1)    # 11: 256->256

        # Transition 1 + E-ELAN 2: layers 12-24
        # 12: MP
        self.conv13 = Conv(256, 128, 1, 1)    # 13: 256->128
        self.conv14 = Conv(256, 128, 1, 1)    # 14: 256->128 (from 11)
        self.conv15 = Conv(128, 128, 3, 2)    # 15: 128->128, stride 2
        # 16: Concat [15,13] = 256
        self.conv17 = Conv(256, 128, 1, 1)    # 17: 256->128
        self.conv18 = Conv(256, 128, 1, 1)    # 18: 256->128 (from 16)
        self.conv19 = Conv(128, 128, 3, 1)    # 19: 128->128
        self.conv20 = Conv(128, 128, 3, 1)    # 20: 128->128
        self.conv21 = Conv(128, 128, 3, 1)    # 21: 128->128
        self.conv22 = Conv(128, 128, 3, 1)    # 22: 128->128
        # 23: Concat [22,20,18,17] = 512, tagged B3
        self.conv24 = Conv(512, 512, 1, 1)    # 24: 512->512

        # Transition 2 + E-ELAN 3: layers 25-37
        # 25: MP
        self.conv26 = Conv(512, 256, 1, 1)    # 26: 512->256
        self.conv27 = Conv(512, 256, 1, 1)    # 27: 512->256 (from 24)
        self.conv28 = Conv(256, 256, 3, 2)    # 28: 256->256, stride 2
        # 29: Concat [28,26] = 512
        self.conv30 = Conv(512, 256, 1, 1)    # 30: 512->256
        self.conv31 = Conv(512, 256, 1, 1)    # 31: 512->256 (from 29)
        self.conv32 = Conv(256, 256, 3, 1)    # 32: 256->256
        self.conv33 = Conv(256, 256, 3, 1)    # 33: 256->256
        self.conv34 = Conv(256, 256, 3, 1)    # 34: 256->256
        self.conv35 = Conv(256, 256, 3, 1)    # 35: 256->256
        # 36: Concat [35,33,31,30] = 1024
        self.conv37 = Conv(1024, 1024, 1, 1)  # 37: 1024->1024, tagged B4

        # Transition 3 + E-ELAN 4: layers 38-50
        # 38: MP
        self.conv39 = Conv(1024, 512, 1, 1)   # 39: 1024->512
        self.conv40 = Conv(1024, 512, 1, 1)   # 40: 1024->512 (from 37)
        self.conv41 = Conv(512, 512, 3, 2)    # 41: 512->512, stride 2
        # 42: Concat [41,39] = 1024
        self.conv43 = Conv(1024, 256, 1, 1)   # 43: 1024->256
        self.conv44 = Conv(1024, 256, 1, 1)   # 44: 1024->256 (from 42)
        self.conv45 = Conv(256, 256, 3, 1)    # 45: 256->256
        self.conv46 = Conv(256, 256, 3, 1)    # 46: 256->256
        self.conv47 = Conv(256, 256, 3, 1)    # 47: 256->256
        self.conv48 = Conv(256, 256, 3, 1)    # 48: 256->256
        # 49: Concat [48,46,44,43] = 1024
        self.conv50 = Conv(1024, 1024, 1, 1)  # 50: 1024->1024, tagged B5

        self.mp = MP(2)

    def forward(self, x):
        # Stem
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)

        # E-ELAN 1
        x4 = self.conv4(x3)
        x5 = self.conv5(x3)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = torch.cat([x9, x7, x5, x4], 1)  # 256 channels
        x11 = self.conv11(x10)

        # Transition 1 + E-ELAN 2
        x12 = self.mp(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x11)
        x15 = self.conv15(x14)
        x16 = torch.cat([x15, x13], 1)  # 256 channels
        x17 = self.conv17(x16)
        x18 = self.conv18(x16)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        x21 = self.conv21(x20)
        x22 = self.conv22(x21)
        x23 = torch.cat([x22, x20, x18, x17], 1)  # 512 channels (B3)
        b3 = self.conv24(x23)

        # Transition 2 + E-ELAN 3
        x25 = self.mp(b3)
        x26 = self.conv26(x25)
        x27 = self.conv27(b3)
        x28 = self.conv28(x27)
        x29 = torch.cat([x28, x26], 1)  # 512 channels
        x30 = self.conv30(x29)
        x31 = self.conv31(x29)
        x32 = self.conv32(x31)
        x33 = self.conv33(x32)
        x34 = self.conv34(x33)
        x35 = self.conv35(x34)
        x36 = torch.cat([x35, x33, x31, x30], 1)  # 1024 channels
        b4 = self.conv37(x36)

        # Transition 3 + E-ELAN 4
        x38 = self.mp(b4)
        x39 = self.conv39(x38)
        x40 = self.conv40(b4)
        x41 = self.conv41(x40)
        x42 = torch.cat([x41, x39], 1)  # 1024 channels
        x43 = self.conv43(x42)
        x44 = self.conv44(x42)
        x45 = self.conv45(x44)
        x46 = self.conv46(x45)
        x47 = self.conv47(x46)
        x48 = self.conv48(x47)
        x49 = torch.cat([x48, x46, x44, x43], 1)  # 1024 channels
        b5 = self.conv50(x49)

        return b3, b4, b5


class Neck7(nn.Module):
    """YOLOv7 PANet Neck matching the official v7.yaml structure.

    Architecture (layer indices from v7.yaml head):
    - 51: SPPCSPC (N3)
    - 52-63: Top-down path to N2
    - 64-75: Top-down path to P3
    - 76-88: Bottom-up path to P4
    - 89-101: Bottom-up path to P5
    """

    def __init__(self):
        super().__init__()

        # SPPCSPC: layer 51
        self.sppcspc = SPPCSPC(1024, 512)

        # Top-down to N2: layers 52-63
        self.conv52 = Conv(512, 256, 1, 1)    # 52: 512->256
        # 53: Upsample
        self.conv54 = Conv(1024, 256, 1, 1)   # 54: from B4, 1024->256
        # 55: Concat [54,53_up] = 512
        self.conv56 = Conv(512, 256, 1, 1)    # 56: 512->256
        self.conv57 = Conv(512, 256, 1, 1)    # 57: 512->256 (from 55)
        self.conv58 = Conv(256, 128, 3, 1)    # 58: 256->128
        self.conv59 = Conv(128, 128, 3, 1)    # 59: 128->128
        self.conv60 = Conv(128, 128, 3, 1)    # 60: 128->128
        self.conv61 = Conv(128, 128, 3, 1)    # 61: 128->128
        # 62: Concat [61,60,59,58,57,56] = 256+256+128+128+128+128 = 1024
        self.conv63 = Conv(1024, 256, 1, 1)   # 63: 1024->256 (N2)

        # Top-down to P3: layers 64-75
        self.conv64 = Conv(256, 128, 1, 1)    # 64: 256->128
        # 65: Upsample
        self.conv66 = Conv(512, 128, 1, 1)    # 66: from B3, 512->128
        # 67: Concat [66,65_up] = 256
        self.conv68 = Conv(256, 128, 1, 1)    # 68: 256->128
        self.conv69 = Conv(256, 128, 1, 1)    # 69: 256->128 (from 67)
        self.conv70 = Conv(128, 64, 3, 1)     # 70: 128->64
        self.conv71 = Conv(64, 64, 3, 1)      # 71: 64->64
        self.conv72 = Conv(64, 64, 3, 1)      # 72: 64->64
        self.conv73 = Conv(64, 64, 3, 1)      # 73: 64->64
        # 74: Concat [73,72,71,70,69,68] = 128+128+64+64+64+64 = 512
        self.conv75 = Conv(512, 128, 1, 1)    # 75: 512->128 (P3)

        # Bottom-up to P4: layers 76-88
        # 76: MP
        self.conv77 = Conv(128, 128, 1, 1)    # 77: 128->128
        self.conv78 = Conv(128, 128, 1, 1)    # 78: 128->128 (from P3)
        self.conv79 = Conv(128, 128, 3, 2)    # 79: 128->128, stride 2
        # 80: Concat [79,77,N2] = 128+128+256 = 512
        self.conv81 = Conv(512, 256, 1, 1)    # 81: 512->256
        self.conv82 = Conv(512, 256, 1, 1)    # 82: 512->256 (from 80)
        self.conv83 = Conv(256, 128, 3, 1)    # 83: 256->128
        self.conv84 = Conv(128, 128, 3, 1)    # 84: 128->128
        self.conv85 = Conv(128, 128, 3, 1)    # 85: 128->128
        self.conv86 = Conv(128, 128, 3, 1)    # 86: 128->128
        # 87: Concat [86,85,84,83,82,81] = 256+256+128+128+128+128 = 1024
        self.conv88 = Conv(1024, 256, 1, 1)   # 88: 1024->256 (P4)

        # Bottom-up to P5: layers 89-101
        # 89: MP
        self.conv90 = Conv(256, 256, 1, 1)    # 90: 256->256
        self.conv91 = Conv(256, 256, 1, 1)    # 91: 256->256 (from P4)
        self.conv92 = Conv(256, 256, 3, 2)    # 92: 256->256, stride 2
        # 93: Concat [92,90,N3] = 256+256+512 = 1024
        self.conv94 = Conv(1024, 512, 1, 1)   # 94: 1024->512
        self.conv95 = Conv(1024, 512, 1, 1)   # 95: 1024->512 (from 93)
        self.conv96 = Conv(512, 256, 3, 1)    # 96: 512->256
        self.conv97 = Conv(256, 256, 3, 1)    # 97: 256->256
        self.conv98 = Conv(256, 256, 3, 1)    # 98: 256->256
        self.conv99 = Conv(256, 256, 3, 1)    # 99: 256->256
        # 100: Concat [99,98,97,96,95,94] = 512+512+256+256+256+256 = 2048
        self.conv101 = Conv(2048, 512, 1, 1)  # 101: 2048->512 (P5)

        self.mp = MP(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, b3, b4, b5):
        # SPPCSPC
        n3 = self.sppcspc(b5)  # 512 channels

        # Top-down to N2
        x52 = self.conv52(n3)
        x53 = self.up(x52)
        x54 = self.conv54(b4)
        x55 = torch.cat([x54, x53], 1)
        x56 = self.conv56(x55)
        x57 = self.conv57(x55)
        x58 = self.conv58(x57)
        x59 = self.conv59(x58)
        x60 = self.conv60(x59)
        x61 = self.conv61(x60)
        x62 = torch.cat([x61, x60, x59, x58, x57, x56], 1)
        n2 = self.conv63(x62)

        # Top-down to P3
        x64 = self.conv64(n2)
        x65 = self.up(x64)
        x66 = self.conv66(b3)
        x67 = torch.cat([x66, x65], 1)
        x68 = self.conv68(x67)
        x69 = self.conv69(x67)
        x70 = self.conv70(x69)
        x71 = self.conv71(x70)
        x72 = self.conv72(x71)
        x73 = self.conv73(x72)
        x74 = torch.cat([x73, x72, x71, x70, x69, x68], 1)
        p3 = self.conv75(x74)

        # Bottom-up to P4
        x76 = self.mp(p3)
        x77 = self.conv77(x76)
        x78 = self.conv78(p3)
        x79 = self.conv79(x78)
        x80 = torch.cat([x79, x77, n2], 1)
        x81 = self.conv81(x80)
        x82 = self.conv82(x80)
        x83 = self.conv83(x82)
        x84 = self.conv84(x83)
        x85 = self.conv85(x84)
        x86 = self.conv86(x85)
        x87 = torch.cat([x86, x85, x84, x83, x82, x81], 1)
        p4 = self.conv88(x87)

        # Bottom-up to P5
        x89 = self.mp(p4)
        x90 = self.conv90(x89)
        x91 = self.conv91(p4)
        x92 = self.conv92(x91)
        x93 = torch.cat([x92, x90, n3], 1)
        x94 = self.conv94(x93)
        x95 = self.conv95(x93)
        x96 = self.conv96(x95)
        x97 = self.conv97(x96)
        x98 = self.conv98(x97)
        x99 = self.conv99(x98)
        x100 = torch.cat([x99, x98, x97, x96, x95, x94], 1)
        p5 = self.conv101(x100)

        return p3, p4, p5


class LibreYOLO7Model(nn.Module):
    """
    Complete LibreYOLO7 model matching the official v7.yaml.
    """

    def __init__(self, config='base', nb_classes=80, img_size=640):
        super().__init__()

        if config != 'base':
            raise ValueError(f"Only 'base' config is currently supported")

        self.config = config
        self.nc = nb_classes
        self.img_size = img_size

        self.backbone = Backbone7()
        self.neck = Neck7()

        # RepConv layers: 102, 103, 104
        self.rep102 = RepConv(128, 256, 3, 1)   # P3: 128->256
        self.rep103 = RepConv(256, 512, 3, 1)   # P4: 256->512
        self.rep104 = RepConv(512, 1024, 3, 1)  # P5: 512->1024

        # Detection head: layer 105
        self.detect = IDetect(
            nc=nb_classes,
            anchors=V7_ANCHORS,
            ch=(256, 512, 1024)
        )

    def forward(self, x):
        b3, b4, b5 = self.backbone(x)
        p3, p4, p5 = self.neck(b3, b4, b5)

        d3 = self.rep102(p3)
        d4 = self.rep103(p4)
        d5 = self.rep104(p5)

        output = self.detect([d3, d4, d5])

        if self.training:
            return output

        predictions, raw_outputs = output
        return {
            'predictions': predictions,
            'raw_outputs': raw_outputs,
            'x8': {'features': p3},
            'x16': {'features': p4},
            'x32': {'features': p5}
        }

    def fuse(self):
        """Fuse RepConv layers for faster inference."""
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_convs()
        return self
