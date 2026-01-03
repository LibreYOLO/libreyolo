"""
Neural network architecture for LibreYOLO-RD (Regional Diversity).

YOLO-RD extends YOLOv9-c architecture with Dynamic Convolution (DConv)
for enhanced regional feature diversity.

All layers are self-contained within this module.
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class RepConvN(nn.Module):
    """RepConv block for neural network re-parameterization."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(c2) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNBottleneck(nn.Module):
    """Bottleneck with RepConvN."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    """CSP Bottleneck with RepConvN."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN(nn.Module):
    """CSP-ELAN block with RepConvN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c2 // 2, c3, n), Conv(c3, c3, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c3, c3, n), Conv(c3, c3, 3, 1))
        self.cv4 = Conv(c2 + 2 * c3, c4, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        return self.cv4(torch.cat(y, 1))


class PONO(nn.Module):
    """
    Position-wise Channel Normalization.

    Normalizes features position-wise along the channel dimension.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Normalize along channel dimension for each spatial position
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, H, W)
        return (x - mean) / std


class DConv(nn.Module):
    """
    Dynamic Convolution with Position-wise Channel Normalization (PONO).

    This is the key component of YOLO-RD that enables regional diversity.

    Components:
    - CG: Channel Group projection (in -> atoms) with activation
    - GIE: Group Interaction Extraction (depthwise conv, k=5) without activation
    - PONO: Position-wise channel normalization (method, not layer)
    - D: Decoding projection (atoms -> in) without activation

    Output: alpha * D(PONO(GIE(CG(x)))) + (1-alpha) * x

    Matches the original YOLO DConv implementation exactly.
    """

    def __init__(self, in_channels=512, atoms=512, alpha=0.8):
        """
        Args:
            in_channels: Number of input channels
            atoms: Number of atoms for dynamic convolution (intermediate channels)
            alpha: Blending factor for residual connection
        """
        super().__init__()
        self.alpha = alpha

        # CG: Channel Group projection (with activation)
        self.CG = Conv(in_channels, atoms, 1, act=True)

        # GIE: Group Interaction Extraction (depthwise conv, no activation)
        self.GIE = Conv(atoms, atoms, 5, g=atoms, act=False)

        # D: Decoding projection (no activation)
        self.D = Conv(atoms, in_channels, 1, act=False)

    def PONO(self, x):
        """Position-wise channel normalization."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + 1e-5)

    def forward(self, r):
        x = self.CG(r)
        x = self.GIE(x)
        x = self.PONO(x)
        x = self.D(x)
        return self.alpha * x + (1 - self.alpha) * r


class RepNCSPELAND(nn.Module):
    """
    RepNCSPELAN + DConv for Regional Diversity.

    This is the core building block unique to YOLO-RD.
    Matches the original YOLO RepNCSPELAND implementation.
    """

    def __init__(self, c1, c2, c3, c4, n=1, atoms=512, dconv_alpha=0.8):
        """
        Args:
            c1: Input channels
            c2: Intermediate channels 1 (part_channels)
            c3: Intermediate channels 2 (part_channels // 2)
            c4: Output channels
            n: Number of RepNCSP blocks
            atoms: Atoms for DConv (intermediate channels)
            dconv_alpha: Alpha for DConv blending
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c2 // 2, c3, n), Conv(c3, c3, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c3, c3, n), Conv(c3, c3, 3, 1))
        self.cv4 = Conv(c2 + 2 * c3, c4, 1, 1)

        # DConv for regional diversity - output channels is c4
        self.dconv = DConv(in_channels=c4, atoms=atoms, alpha=dconv_alpha)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        out = self.cv4(torch.cat(y, 1))

        # Apply DConv for regional diversity
        return self.dconv(out)


class ADown(nn.Module):
    """Advanced dual-path downsampling block."""

    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP + ELAN block for global context.

    Matches the original YOLO SPPELAN implementation.
    """

    def __init__(self, c1, c2, neck_channels=None, k=5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            neck_channels: Intermediate channels (default: c2 // 2)
            k: Pool kernel size
        """
        super().__init__()
        neck_channels = neck_channels or c2 // 2
        self.cv1 = Conv(c1, neck_channels, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for _ in range(3)])
        self.cv5 = Conv(4 * neck_channels, c2, 1, 1)

    def forward(self, x):
        features = [self.cv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.cv5(torch.cat(features, 1))


class Concat(nn.Module):
    """Concatenate tensors along dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """Distribution Focal Loss module."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, 4*reg_max, anchors
        # Reshape to (b, 4, reg_max, a) and transpose to (b, reg_max, 4, a)
        x = x.view(b, 4, self.c1, a).transpose(1, 2)
        # Apply softmax over reg_max dimension (now dim 1)
        x = F.softmax(x, dim=1)
        # Apply weighted sum via conv: reshape to (b, reg_max, 4*a) for Conv2d
        x = self.conv(x.reshape(b, self.c1, 4, a))
        # Reshape back to (b, 4, a)
        return x.view(b, 4, a)


def round_up(x, divisor):
    """Round up x to be divisible by divisor."""
    return int((x + divisor - 1) // divisor) * divisor


class DDetect(nn.Module):
    """Detection Head for YOLO-RD.

    Matches the original YOLO MultiheadDetection + Detection structure.
    Uses grouped convolutions in the anchor branch.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), reg_max=16, stride=(), use_group=True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.tensor(stride) if stride else torch.zeros(self.nl)

        # Following original Detection class structure
        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max  # 64

        first_neck = ch[0]  # First channel count (P3 = 256)

        # Anchor branch uses grouped convolutions
        anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)  # 64
        # Class branch
        class_neck = max(first_neck, min(nc * 2, 128))  # 256

        # cv2 = anchor_conv (box regression)
        self.cv2 = nn.ModuleList()
        for x in ch:
            self.cv2.append(nn.Sequential(
                Conv(x, anchor_neck, 3),
                Conv(anchor_neck, anchor_neck, 3, g=groups),
                nn.Conv2d(anchor_neck, anchor_channels, 1, groups=groups),
            ))

        # cv3 = class_conv (classification)
        self.cv3 = nn.ModuleList()
        for x in ch:
            self.cv3.append(nn.Sequential(
                Conv(x, class_neck, 3),
                Conv(class_neck, class_neck, 3),
                nn.Conv2d(class_neck, nc, 1),
            ))

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self._init_bias()

    def _init_bias(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)

    def forward(self, x):
        shape = x[0].shape

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self._make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self._decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y, x

    def _make_anchors(self, feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device

        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def _decode_bboxes(self, bboxes, anchors):
        return self._dist2bbox(bboxes, anchors, xywh=False)

    def _dist2bbox(self, distance, anchor_points, xywh=True, dim=1):
        """Transform distance(ltrb) to box(xywh or xyxy).

        Args:
            distance: (batch, 4, anchors) - l, t, r, b distances from anchor
            anchor_points: (1, 2, anchors) - anchor center coordinates
            xywh: Return xywh format if True, else xyxy
            dim: Dimension to split/concat on (should be 1 for coordinates)
        """
        lt, rb = distance.chunk(2, dim)  # Each (batch, 2, anchors)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)
        return torch.cat((x1y1, x2y2), dim)


# =============================================================================
# YOLO-RD Model Architecture
# =============================================================================


class BackboneRD(nn.Module):
    """YOLO-RD Backbone (based on v9-c with DConv at B3).

    Matches the official rd-9c.yaml architecture exactly.

    Args:
        atoms: Number of atoms for DConv (512 for rd-9c, 4096 for rd-9c-4096)
    """

    def __init__(self, atoms=512):
        super().__init__()

        # rd-9c configuration from YAML:
        # Conv: out_channels: 64
        # Conv: out_channels: 128
        # RepNCSPELAN: out_channels: 256, part_channels: 128
        # ADown: out_channels: 256
        # RepNCSPELAND: out_channels: 512, part_channels: 256, atoms: 512 or 4096
        # ADown: out_channels: 512
        # RepNCSPELAN: out_channels: 512, part_channels: 512
        # ADown: out_channels: 512
        # RepNCSPELAN: out_channels: 512, part_channels: 512
        # SPPELAN: out_channels: 512

        # Stem
        self.conv0 = Conv(3, 64, 3, 2)
        self.conv1 = Conv(64, 128, 3, 2)

        # Stage 1: RepNCSPELAN(in=128, part=128, part//2=64, out=256)
        self.elan1 = RepNCSPELAN(128, 128, 64, 256, 1)

        # Stage 2 - B3 with DConv for Regional Diversity
        self.down2 = ADown(256, 256)
        # RepNCSPELAND(in=256, part=256, part//2=128, out=512, atoms=atoms)
        self.elan2 = RepNCSPELAND(256, 256, 128, 512, 1, atoms=atoms)

        # Stage 3 - B4
        self.down3 = ADown(512, 512)
        # RepNCSPELAN(in=512, part=512, part//2=256, out=512)
        self.elan3 = RepNCSPELAN(512, 512, 256, 512, 1)

        # Stage 4 - B5
        self.down4 = ADown(512, 512)
        # RepNCSPELAN(in=512, part=512, part//2=256, out=512)
        self.elan4 = RepNCSPELAN(512, 512, 256, 512, 1)

        # SPP: SPPELAN(in=512, out=512)
        self.spp = SPPELAN(512, 512)

    def forward(self, x):
        # Stem
        x = self.conv0(x)
        x = self.conv1(x)

        # Stage 1
        x = self.elan1(x)

        # Stage 2 - B3 with DConv
        x = self.down2(x)
        p3 = self.elan2(x)

        # Stage 3 - B4
        x = self.down3(p3)
        p4 = self.elan3(x)

        # Stage 4 - B5
        x = self.down4(p4)
        x = self.elan4(x)
        p5 = self.spp(x)

        return p3, p4, p5


class NeckRD(nn.Module):
    """YOLO-RD PANet Neck.

    Matches the official rd-9c.yaml architecture exactly.

    Flow:
    - Backbone: B3=512, B4=512, N3(spp)=512
    - Top-down: up(N3)+B4=1024 -> N4=512
    - Head: up(N4)+B3=1024 -> P3=256
    - Bottom-up: down(P3)+N4=768 -> P4=512
    - Bottom-up: down(P4)+N3=1024 -> P5=512
    """

    def __init__(self):
        super().__init__()

        # Top-down path (neck)
        # up(spp)+B4 = 512+512=1024 -> N4=512
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.elan_up1 = RepNCSPELAN(1024, 512, 256, 512, 1)  # N4

        # Head section
        # up(N4)+B3 = 512+512=1024 -> P3=256
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.elan_up2 = RepNCSPELAN(1024, 256, 128, 256, 1)  # P3

        # Bottom-up path
        # down(P3)+N4 = 256+512=768 -> P4=512
        self.down1 = ADown(256, 256)
        self.elan_down1 = RepNCSPELAN(768, 512, 256, 512, 1)  # P4

        # down(P4)+N3 = 512+512=1024 -> P5=512
        self.down2 = ADown(512, 512)
        self.elan_down2 = RepNCSPELAN(1024, 512, 256, 512, 1)  # P5

    def forward(self, b3, b4, n3):
        """
        Args:
            b3: B3 output from backbone (512 channels)
            b4: B4 output from backbone (512 channels)
            n3: N3/spp output from backbone (512 channels)
        """
        # Top-down: up(n3) + b4 -> N4
        x = self.up1(n3)
        x = torch.cat([x, b4], 1)
        n4 = self.elan_up1(x)

        # up(N4) + B3 -> P3
        x = self.up2(n4)
        x = torch.cat([x, b3], 1)
        p3 = self.elan_up2(x)

        # Bottom-up: down(P3) + N4 -> P4
        x = self.down1(p3)
        x = torch.cat([x, n4], 1)
        p4 = self.elan_down1(x)

        # down(P4) + N3 -> P5
        x = self.down2(p4)
        x = torch.cat([x, n3], 1)
        p5 = self.elan_down2(x)

        return p3, p4, p5


class LibreYOLORDModel(nn.Module):
    """
    Complete LibreYOLO-RD (Regional Diversity) model.

    Based on YOLOv9-c architecture with DConv at B3 position
    for enhanced regional feature diversity.
    """

    def __init__(self, config='c', reg_max=16, nb_classes=80, img_size=640, atoms=512):
        """
        Initialize YOLO-RD model.

        Args:
            config: Model configuration ('c' for rd-9c)
            reg_max: Regression max value for DFL
            nb_classes: Number of classes
            img_size: Input image size
            atoms: DConv atoms (512 for rd-9c, 4096 for rd-9c-4096)
        """
        super().__init__()

        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max
        self.img_size = img_size
        self.atoms = atoms

        c3 = 256
        c4 = 512

        self.backbone = BackboneRD(atoms=atoms)
        self.neck = NeckRD()

        # Detection head
        self.detect = DDetect(
            nc=nb_classes,
            ch=(c3, c4, c4),
            reg_max=reg_max,
            stride=(8, 16, 32)
        )

    def forward(self, x):
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(p3, p4, p5)

        # Detection head
        output = self.detect([n3, n4, n5])

        if self.training:
            return output

        # Inference mode
        y, x_list = output

        return {
            'predictions': y,
            'raw_outputs': x_list,
            'x8': {'features': n3},
            'x16': {'features': n4},
            'x32': {'features': n5}
        }

    def fuse(self):
        """Fuse Conv+BN for faster inference."""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = self._fuse_conv_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self

    def _fuse_conv_bn(self, conv, bn):
        """Fuse Conv2d and BatchNorm2d."""
        fusedconv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True
        ).requires_grad_(False).to(conv.weight.device)

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
