"""
Neural network architecture for LibreYOLO v9.

All layers are self-contained within this module to allow scientists to modify
a single version without affecting others.

Layer naming follows the official YOLO repo convention for direct weight loading:
- model.0., model.1., ... etc.

Supports v9-t (tiny), v9-s (small), v9-m (medium), and v9-c (compact/largest).
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

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer.

        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
            d: Dilation
            act: Activation (True for default, False for none, or nn.Module)
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Forward pass for fused Conv (inference only)."""
        return self.act(self.conv(x))


class RepConvN(nn.Module):
    """
    RepConv block for neural network re-parameterization.

    During training: 3x3 conv + 1x1 conv (+ identity if c1==c2)
    During inference: Single fused 3x3 conv
    """

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
        """Forward pass with parallel paths."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_fuse(self, x):
        """Forward pass for fused RepConv."""
        return self.act(self.conv(x))

    def fuse_convs(self):
        """Fuse parallel convolutions into single conv for inference."""
        if hasattr(self, 'conv'):
            return

        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)

        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])

        self.conv = nn.Conv2d(self.c1, self.c2, 3, 1, 1, groups=self.g, bias=True)
        self.conv.weight.data = kernel3x3 + kernel1x1 + kernelid
        self.conv.bias.data = bias3x3 + bias1x1 + biasid

        for para in self.parameters():
            para.detach_()

        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

    def _fuse_bn_tensor(self, branch):
        """Fuse batch norm into conv weights."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = torch.zeros((self.c1, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        else:
            raise NotImplementedError

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Add shortcut connection
            g: Groups for 3x3 conv
            k: Kernel sizes for the two convs
            e: Expansion ratio for hidden channels
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
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
    """CSP Bottleneck with RepConvN (3 convolutions)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of bottleneck blocks
            shortcut: Use shortcut connections in bottlenecks
            g: Groups
            e: Expansion ratio
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class ELAN(nn.Module):
    """
    Efficient Layer Aggregation Network block.
    Used in v9-t and v9-s variants.

    Architecture:
    - cv1: input -> part_channels (c2), then split in half
    - cv2: takes half of cv1 output, outputs c3
    - cv3: takes cv2 output, outputs c3
    - cv4: concatenates [half1, half2, cv2_out, cv3_out] -> output
    """

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Args:
            c1: Input channels
            c2: cv1 output channels (part_channels, gets split in half)
            c3: cv2/cv3 output channels (part_channels // 2)
            c4: Output channels
            n: Number of additional conv blocks after cv3
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2 // 2, c3, 3, 1)
        self.cv3 = Conv(c3, c3, 3, 1)
        # cv4 input = c2/2 + c2/2 + c3 + c3*(n) = c2 + c3*(1+n)
        # For n=1 (default): c2 + 2*c3
        self.cv4 = Conv(c2 + c3 * (1 + n), c4, 1, 1)
        self.m = nn.ModuleList(Conv(c3, c3, 3, 1) for _ in range(n - 1))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.append(self.cv2(y[-1]))
        y.append(self.cv3(y[-1]))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN(nn.Module):
    """
    CSP-ELAN block with RepConvN.
    Used in v9-m and v9-c variants.
    """

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Args:
            c1: Input channels
            c2: Intermediate channels 1
            c3: Intermediate channels 2
            c4: Output channels
            n: Number of RepNCSP blocks
        """
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


class AConv(nn.Module):
    """Asymmetric convolution for downsampling."""

    def __init__(self, c1, c2):
        super().__init__()
        self.cv = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv(x)


class ADown(nn.Module):
    """
    Advanced dual-path downsampling block.
    Used in v9-c variant.
    """

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

    Architecture matches official YOLO SPPELAN:
    - conv1: in_channels -> neck_channels
    - pools: 3x MaxPool2d (no weights)
    - conv5: 4*neck_channels -> out_channels
    """

    def __init__(self, c1, c2, c3, k=5):
        """
        Args:
            c1: Input channels
            c2: Neck channels (intermediate)
            c3: Output channels
            k: Max pool kernel size
        """
        super().__init__()
        # Match YOLO naming: conv1, pools, conv5
        self.cv1 = Conv(c1, c2, 1, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for _ in range(3)
        ])
        self.cv5 = Conv(4 * c2, c3, 1, 1)  # Concat 4 features -> output

    def forward(self, x):
        features = [self.cv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.cv5(torch.cat(features, 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Converts distribution predictions to coordinate offsets.
    """

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


class DDetect(nn.Module):
    """
    Decoupled Detection Head for v9.
    Anchor-free detection with DFL for box regression.

    Uses grouped convolutions (groups=4) in the box branch to match YOLO.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), reg_max=16, stride=(), use_group=True):
        """
        Args:
            nc: Number of classes
            ch: Input channels for each scale
            reg_max: Maximum value for DFL regression
            stride: Stride for each scale
            use_group: Use grouped convolutions in box branch (default True for YOLO compat)
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor(stride) if stride else torch.zeros(self.nl)

        # Groups for box branch (YOLO uses groups=4)
        groups = 4 if use_group else 1
        anchor_channels = 4 * reg_max  # 64

        # Box branch channel calculation (match YOLO's round_up logic)
        # anchor_neck = max(round_up(first_neck // 4, groups), anchor_channels, reg_max)
        # For ch[0]=256: 256//4=64, round to groups=4 -> 64, max(64, 64, 16) = 64
        c2 = max((ch[0] // 4 + groups - 1) // groups * groups, anchor_channels, reg_max)

        # Class branch (no grouping)
        c3 = max(ch[0], min(nc, 100))

        # Box branch with grouped convolutions
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),  # Regular conv: in -> 64
                Conv(c2, c2, 3, g=groups),  # Grouped conv: 64->64, groups=4
                nn.Conv2d(c2, anchor_channels, 1, groups=groups)  # Grouped conv: 64->64
            ) for x in ch
        )
        # Class branch (no grouping)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self._init_bias()

    def _init_bias(self):
        """Initialize biases for focal loss."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)  # cls

    def forward(self, x):
        """Forward pass returning box and class predictions."""
        shape = x[0].shape  # BCHW

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        # Inference mode
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self._make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # Flatten and concatenate all scales
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # DFL decoding
        dbox = self._decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y, x

    def _make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from feature maps."""
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
        """Decode bboxes from DFL output."""
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
# Model Architecture Definitions
# =============================================================================

# YOLOv9 configurations - exact channel dimensions from official YOLO configs
# Each variant has unique, non-linear channel structures
V9_CONFIGS = {
    't': {  # Tiny
        # Backbone: Conv(16) -> Conv(32) -> ELAN(32) -> [AConv -> RepNCSPELAN] x3
        'conv0_out': 16,
        'conv1_out': 32,
        'first_block': 'elan',  # ELAN for t/s, RepNCSPELAN for m/c
        'first_block_out': 32,
        'down_type': 'aconv',  # AConv for t/s/m, ADown for c
        'stages': [  # (down_out, elan_out, elan_part) for stages 2, 3, 4
            (64, 64, 64),    # B3: AConv->64, RepNCSPELAN->64
            (96, 96, 96),    # B4: AConv->96, RepNCSPELAN->96
            (128, 128, 128), # B5: AConv->128, RepNCSPELAN->128
        ],
        'spp_out': 128,
        'repeat_num': 3,  # RepNCSPELAN repeat for t/s
        # Neck
        'neck_elan_up1': (96, 96),   # N4: out=96, part=96
        'neck_elan_up2': (64, 64),   # P3: out=64, part=64
        'neck_down1_out': 48,        # AConv after P3
        'neck_elan_down1': (96, 96), # P4: out=96, part=96
        'neck_down2_out': 64,        # AConv after P4
        'neck_elan_down2': (128, 128), # P5: out=128, part=128
        # Detection head channels
        'detect_channels': (64, 96, 128),  # P3, P4, P5
    },
    's': {  # Small
        # Backbone: Conv(32) -> Conv(64) -> ELAN(64) -> [AConv -> RepNCSPELAN] x3
        'conv0_out': 32,
        'conv1_out': 64,
        'first_block': 'elan',
        'first_block_out': 64,
        'down_type': 'aconv',
        'stages': [
            (128, 128, 128),  # B3
            (192, 192, 192),  # B4
            (256, 256, 256),  # B5
        ],
        'spp_out': 256,
        'repeat_num': 3,
        # Neck
        'neck_elan_up1': (192, 192),
        'neck_elan_up2': (128, 128),
        'neck_down1_out': 96,
        'neck_elan_down1': (192, 192),
        'neck_down2_out': 128,
        'neck_elan_down2': (256, 256),
        # Detection
        'detect_channels': (128, 192, 256),
    },
    'm': {  # Medium
        # Backbone: Conv(32) -> Conv(64) -> RepNCSPELAN(128) -> [AConv -> RepNCSPELAN] x3
        'conv0_out': 32,
        'conv1_out': 64,
        'first_block': 'repncspelan',
        'first_block_out': 128,
        'first_block_part': 128,
        'down_type': 'aconv',
        'stages': [
            (240, 240, 240),  # B3
            (360, 360, 360),  # B4
            (480, 480, 480),  # B5
        ],
        'spp_out': 480,
        'repeat_num': 1,  # Default repeat for m/c
        # Neck
        'neck_elan_up1': (360, 360),
        'neck_elan_up2': (240, 240),
        'neck_down1_out': 184,
        'neck_elan_down1': (360, 360),
        'neck_down2_out': 240,
        'neck_elan_down2': (480, 480),
        # Detection
        'detect_channels': (240, 360, 480),
    },
    'c': {  # Compact (largest)
        # Backbone: Conv(64) -> Conv(128) -> RepNCSPELAN(256) -> [ADown -> RepNCSPELAN] x3
        'conv0_out': 64,
        'conv1_out': 128,
        'first_block': 'repncspelan',
        'first_block_out': 256,
        'first_block_part': 128,  # part_channels for first RepNCSPELAN
        'down_type': 'adown',
        'stages': [
            (256, 512, 256),  # B3: ADown->256, RepNCSPELAN->512, part=256
            (512, 512, 512),  # B4: ADown->512, RepNCSPELAN->512, part=512
            (512, 512, 512),  # B5: ADown->512, RepNCSPELAN->512, part=512
        ],
        'spp_out': 512,
        'repeat_num': 1,
        # Neck
        'neck_elan_up1': (512, 512),
        'neck_elan_up2': (256, 256),
        'neck_down1_out': 256,
        'neck_elan_down1': (512, 512),
        'neck_down2_out': 512,
        'neck_elan_down2': (512, 512),
        # Detection
        'detect_channels': (256, 512, 512),
    },
}


class Backbone9(nn.Module):
    """YOLOv9 Backbone.

    Supports all variants with their specific architectures:
    - v9-t/s: Conv -> Conv -> ELAN -> [AConv -> RepNCSPELAN] x3 -> SPPELAN
    - v9-m/c: Conv -> Conv -> RepNCSPELAN -> [AConv/ADown -> RepNCSPELAN] x3 -> SPPELAN
    """

    def __init__(self, config='c'):
        super().__init__()

        cfg = V9_CONFIGS[config]
        self.config = config

        # Stem
        self.conv0 = Conv(3, cfg['conv0_out'], 3, 2)
        self.conv1 = Conv(cfg['conv0_out'], cfg['conv1_out'], 3, 2)

        # First block (ELAN for t/s, RepNCSPELAN for m/c)
        if cfg['first_block'] == 'elan':
            # ELAN(c1, c2, c3, c4, n) where:
            #   c1 = input channels
            #   c2 = cv1 output (part_channels)
            #   c3 = cv2/cv3 output (part_channels // 2)
            #   c4 = output channels
            # For v9-t/s: ELAN {out_channels: X, part_channels: X}
            c1 = cfg['conv1_out']
            c4 = cfg['first_block_out']
            part = c4  # part_channels = out_channels for t/s ELAN
            self.elan1 = ELAN(c1, part, part // 2, c4, n=1)
        else:
            # RepNCSPELAN for m/c
            # RepNCSPELAN(c1, c2, c3, c4, n) where:
            #   c1 = input channels
            #   c2 = cv1 output = part_channels (gets split in half)
            #   c3 = cv2/cv3 internal = part_channels // 2
            #   c4 = output channels
            c1 = cfg['conv1_out']
            c4 = cfg['first_block_out']
            part = cfg.get('first_block_part', c4)
            self.elan1 = RepNCSPELAN(c1, part, part // 2, c4, cfg['repeat_num'])

        # Determine downsampling block type
        DownBlock = ADown if cfg['down_type'] == 'adown' else AConv
        n = cfg['repeat_num']

        # Stage 2 (B3) - first stage after initial block
        # stage = (down_out, elan_out, part_channels)
        stage = cfg['stages'][0]
        prev_ch = cfg['first_block_out']
        self.down2 = DownBlock(prev_ch, stage[0])
        # RepNCSPELAN: c1=down_out, c2=part, c3=part//2, c4=out
        self.elan2 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # Stage 3 (B4)
        stage = cfg['stages'][1]
        prev_ch = cfg['stages'][0][1]  # Previous elan output
        self.down3 = DownBlock(prev_ch, stage[0])
        self.elan3 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # Stage 4 (B5)
        stage = cfg['stages'][2]
        prev_ch = cfg['stages'][1][1]
        self.down4 = DownBlock(prev_ch, stage[0])
        self.elan4 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # SPP
        spp_in = cfg['stages'][2][1]
        spp_out = cfg['spp_out']
        self.spp = SPPELAN(spp_in, spp_out // 2, spp_out)

    def forward(self, x):
        # Stem
        x = self.conv0(x)
        x = self.conv1(x)

        # First block
        x = self.elan1(x)

        # Stage 2 - B3/P3
        x = self.down2(x)
        p3 = self.elan2(x)

        # Stage 3 - B4/P4
        x = self.down3(p3)
        p4 = self.elan3(x)

        # Stage 4 - B5/P5
        x = self.down4(p4)
        x = self.elan4(x)
        p5 = self.spp(x)

        return p3, p4, p5


class Neck9(nn.Module):
    """YOLOv9 PANet Neck + Head.

    Architecture (varies by config):
    Top-down path:
    - UpSample + Concat(B4) -> RepNCSPELAN (N4)
    - UpSample + Concat(B3) -> RepNCSPELAN (P3)
    Bottom-up path:
    - AConv/ADown + Concat(N4) -> RepNCSPELAN (P4)
    - AConv/ADown + Concat(SPP) -> RepNCSPELAN (P5)
    """

    def __init__(self, config='c'):
        super().__init__()

        cfg = V9_CONFIGS[config]
        self.config = config
        n = cfg['repeat_num']

        # Get backbone output channels for concatenation
        b3_ch = cfg['stages'][0][1]  # B3 output channels
        b4_ch = cfg['stages'][1][1]  # B4 output channels
        spp_ch = cfg['spp_out']       # SPP/P5 output channels

        # Top-down path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # Concat(SPP_up, B4) -> N4
        up1_in = spp_ch + b4_ch
        up1_out, up1_part = cfg['neck_elan_up1']
        # RepNCSPELAN: c1=concat_in, c2=part, c3=part//2, c4=out
        self.elan_up1 = RepNCSPELAN(up1_in, up1_part, up1_part // 2, up1_out, n)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # Concat(N4_up, B3) -> P3
        up2_in = up1_out + b3_ch
        up2_out, up2_part = cfg['neck_elan_up2']
        self.elan_up2 = RepNCSPELAN(up2_in, up2_part, up2_part // 2, up2_out, n)

        # Bottom-up path
        DownBlock = ADown if cfg['down_type'] == 'adown' else AConv

        # P3 -> down -> Concat(N4) -> P4
        p3_out = up2_out
        self.down1 = DownBlock(p3_out, cfg['neck_down1_out'])
        down1_concat_in = cfg['neck_down1_out'] + up1_out
        down1_out, down1_part = cfg['neck_elan_down1']
        self.elan_down1 = RepNCSPELAN(down1_concat_in, down1_part, down1_part // 2, down1_out, n)

        # P4 -> down -> Concat(SPP) -> P5
        p4_out = down1_out
        self.down2 = DownBlock(p4_out, cfg['neck_down2_out'])
        down2_concat_in = cfg['neck_down2_out'] + spp_ch
        down2_out, down2_part = cfg['neck_elan_down2']
        self.elan_down2 = RepNCSPELAN(down2_concat_in, down2_part, down2_part // 2, down2_out, n)

    def forward(self, p3, p4, p5):
        # Top-down path
        x = self.up1(p5)
        x = torch.cat([x, p4], 1)
        n4 = self.elan_up1(x)

        x = self.up2(n4)
        x = torch.cat([x, p3], 1)
        out_p3 = self.elan_up2(x)

        # Bottom-up path
        x = self.down1(out_p3)
        x = torch.cat([x, n4], 1)
        out_p4 = self.elan_down1(x)

        x = self.down2(out_p4)
        x = torch.cat([x, p5], 1)
        out_p5 = self.elan_down2(x)

        return out_p3, out_p4, out_p5


class LibreYOLO9Model(nn.Module):
    """
    Complete LibreYOLO9 model.

    Supports v9-t, v9-s, v9-m, and v9-c variants with their specific architectures.
    """

    def __init__(self, config='c', reg_max=16, nb_classes=80, img_size=640):
        """
        Initialize YOLOv9 model.

        Args:
            config: Model size ('t', 's', 'm', 'c')
            reg_max: Regression max value for DFL
            nb_classes: Number of classes
            img_size: Input image size
        """
        super().__init__()

        if config not in V9_CONFIGS:
            raise ValueError(f"Invalid config: {config}. Must be one of: {list(V9_CONFIGS.keys())}")

        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max
        self.img_size = img_size

        cfg = V9_CONFIGS[config]

        self.backbone = Backbone9(config)
        self.neck = Neck9(config)

        # Detection head - use exact channels from config
        detect_channels = cfg['detect_channels']
        self.detect = DDetect(
            nc=nb_classes,
            ch=detect_channels,
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
            # Return raw outputs for loss calculation
            return output

        # Inference mode
        y, x_list = output

        # Return in format compatible with postprocessing
        return {
            'predictions': y,  # (batch, 4+nc, total_anchors)
            'raw_outputs': x_list,
            'x8': {'features': n3},
            'x16': {'features': n4},
            'x32': {'features': n5}
        }

    def fuse(self):
        """Fuse Conv+BN and RepConvN for faster inference."""
        for m in self.modules():
            if isinstance(m, RepConvN):
                m.fuse_convs()
            elif isinstance(m, Conv) and hasattr(m, 'bn'):
                # Fuse Conv+BN
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
