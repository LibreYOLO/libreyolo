"""
Neural network architecture for LibreYOLO yolo9.

Supports yolo9-t (tiny), yolo9-s (small), yolo9-m (medium), and yolo9-c (compact/largest).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_pad(kernel_size, padding=None, dilation=1):
    """Return symmetric padding for stride-preserving convolutions."""
    if padding is not None:
        return padding
    if isinstance(kernel_size, int):
        return ((kernel_size - 1) * dilation) // 2
    if isinstance(dilation, int):
        dilation = [dilation] * len(kernel_size)
    return [((size - 1) * dil) // 2 for size, dil in zip(kernel_size, dilation)]


def create_activation(activation=True):
    """Build an activation module from the YOLOv9 config convention."""
    if isinstance(activation, nn.Module):
        return activation
    if activation is True:
        return nn.SiLU()
    if activation in (False, None):
        return nn.Identity()
    if isinstance(activation, str):
        if activation.lower() in {"false", "none", "identity"}:
            return nn.Identity()
        activation_cls = getattr(nn, activation, None)
        if activation_cls is None:
            raise ValueError(f"Unsupported activation: {activation}")
        try:
            return activation_cls(inplace=True)
        except TypeError:
            return activation_cls()
    raise TypeError(f"Unsupported activation specifier: {activation!r}")


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm + activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        activation=True,
        **legacy_kwargs,
    ):
        """
        Initialize Conv layer.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding override
            groups: Convolution groups
            dilation: Dilation
            activation: Activation specifier
        """
        super().__init__()
        if "k" in legacy_kwargs:
            kernel_size = legacy_kwargs.pop("k")
        if "s" in legacy_kwargs:
            stride = legacy_kwargs.pop("s")
        if "p" in legacy_kwargs:
            padding = legacy_kwargs.pop("p")
        if "g" in legacy_kwargs:
            groups = legacy_kwargs.pop("g")
        if "d" in legacy_kwargs:
            dilation = legacy_kwargs.pop("d")
        if "act" in legacy_kwargs:
            activation = legacy_kwargs.pop("act")
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected Conv arguments: {unknown}")

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            auto_pad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = create_activation(activation)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_without_bn(self, x):
        """Forward pass for fused Conv (inference only)."""
        return self.act(self.conv(x))


class RepConvN(nn.Module):
    """
    RepConv block for neural network re-parameterization.

    During training: 3x3 conv + 1x1 conv (+ identity if c1==c2)
    During inference: Single fused 3x3 conv
    """

    def __init__(
        self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False
    ):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = create_activation(act)

        self.bn = nn.BatchNorm2d(c2) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        """Forward pass with parallel paths."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def forward_deployed(self, x):
        """Forward pass for fused RepConv."""
        return self.act(self.conv(x))

    def fuse_convs(self):
        """Fuse parallel convolutions into single conv for inference."""
        if hasattr(self, "conv"):
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

        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.forward = self.forward_deployed

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
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = torch.zeros(
                    (self.c1, input_dim, 3, 3),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
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
        self.m = nn.Sequential(
            *(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class ELAN(nn.Module):
    """
    Efficient Layer Aggregation Network block.
    Used in yolo9-t and yolo9-s variants.

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
    Used in yolo9-m and yolo9-c variants.
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
    Used in yolo9-c variant.
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

    Architecture follows the YOLOv9 SPPELAN layout:
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
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for _ in range(3)]
        )
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
        self.c1 = c1
        self.register_buffer(
            "project",
            torch.arange(c1, dtype=torch.float32).view(1, 1, c1, 1),
            persistent=False,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "conv.weight", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        batch, _, anchors = x.shape
        logits = x.reshape(batch, 4, self.c1, anchors)
        weights = logits.softmax(dim=2)
        project = self.project.to(device=x.device, dtype=x.dtype)
        return (weights * project).sum(dim=2)


class MaskProto(nn.Module):
    """Prototype mask branch used by YOLO9 segmentation models."""

    def __init__(self, c1, c_mid=256, c_out=32):
        super().__init__()
        self.cv1 = Conv(c1, c_mid, 3)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_mid, c_mid, 3)
        self.cv3 = Conv(c_mid, c_out, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.up(self.cv1(x))))


class DDetect(nn.Module):
    """
    Decoupled Detection Head for yolo9.
    Anchor-free detection with DFL for box regression.

    Uses grouped convolutions (groups=4) in the box branch.
    Supports training mode with loss computation when targets are provided.
    """

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    @staticmethod
    def _box_branch_width(input_channels, groups, output_channels, reg_max):
        """Choose the hidden width used by the box-regression towers."""
        grouped_width = ((input_channels // 4 + groups - 1) // groups) * groups
        return max(grouped_width, output_channels, reg_max)

    @staticmethod
    def _build_box_towers(input_channels, hidden_channels, output_channels, groups):
        """Build box-regression towers for each feature-map scale."""
        return nn.ModuleList(
            nn.Sequential(
                Conv(channels, hidden_channels, 3),
                Conv(hidden_channels, hidden_channels, 3, g=groups),
                nn.Conv2d(hidden_channels, output_channels, 1, groups=groups),
            )
            for channels in input_channels
        )

    @staticmethod
    def _build_class_towers(input_channels, hidden_channels, num_classes):
        """Build class-score towers for each feature-map scale."""
        return nn.ModuleList(
            nn.Sequential(
                Conv(channels, hidden_channels, 3),
                Conv(hidden_channels, hidden_channels, 3),
                nn.Conv2d(hidden_channels, num_classes, 1),
            )
            for channels in input_channels
        )

    def __init__(self, nc=80, ch=(), reg_max=16, stride=(), use_group=True):
        """
        Args:
            nc: Number of classes
            ch: Input channels for each scale
            reg_max: Maximum value for DFL regression
            stride: Stride for each scale
            use_group: Use grouped convolutions in the box branch.
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max
        self.no = nc + reg_max * 4  # number of outputs per anchor
        # Register stride as a buffer so .to(device) moves it. Plain
        # attribute assignment leaves it on CPU even after model.to("cuda")
        # which silently breaks device-mismatch checks under DDP. dtype
        # matches the original (int64 when ``stride`` is an int tuple, else
        # float zeros) — downstream code in loss.py interprets it as
        # both int and float depending on path.
        stride_tensor = (
            torch.tensor(stride) if stride else torch.zeros(self.nl)
        )
        self.register_buffer("stride", stride_tensor, persistent=False)

        self._loss_fn = None

        self._box_groups = 4 if use_group else 1
        self._box_output_channels = 4 * reg_max
        self._box_hidden_channels = self._box_branch_width(
            ch[0], self._box_groups, self._box_output_channels, reg_max
        )
        self._class_hidden_channels = max(ch[0], min(nc, 100))

        self.cv2 = self._build_box_towers(
            ch,
            self._box_hidden_channels,
            self._box_output_channels,
            self._box_groups,
        )
        self.cv3 = self._build_class_towers(ch, self._class_hidden_channels, nc)
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self._init_bias()

    def _init_bias(self):
        """Initialize biases for focal loss."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / float(s)) ** 2
            )  # cls

    def _get_loss_fn(self, device):
        """Lazily initialize loss function for training."""
        if self._loss_fn is None:
            from .loss import YOLO9Loss

            self._loss_fn = YOLO9Loss(
                num_classes=self.nc,
                reg_max=self.reg_max,
                strides=self.stride.tolist(),
                image_size=None,  # Will be set dynamically
                device=device,
            )
        return self._loss_fn

    def forward(self, x, targets=None, img_size=None):
        """
        Forward pass returning box and class predictions.

        Args:
            x: List of feature maps [P3, P4, P5]
            targets: Optional ground truth [B, max_targets, 5] with [class, x1, y1, x2, y2] normalized
            img_size: Optional image size (W, H) for anchor generation

        Returns:
            Training with targets: Dict with loss values
            Training without targets: Raw predictions (list of tensors)
            Inference: Decoded predictions
        """
        shape = x[0].shape  # BCHW

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            if targets is not None:
                # Compute loss
                loss_fn = self._get_loss_fn(x[0].device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(x, targets)
            return x

        # Inference mode
        # In export mode, always regenerate anchors to ensure trace consistency
        # (JIT trace runs the model twice and checks for consistency)
        if self.export or self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in self._make_anchors(x, self.stride, 0.5)
            )
            if not self.export:
                self.shape = shape

        # Flatten and concatenate all scales
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # DFL decoding
        dbox = (
            self._decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        )

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y, x

    def _make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from feature maps."""
        centers_by_level = []
        stride_by_level = []
        for feature, stride in zip(feats, strides):
            dtype, device = feature.dtype, feature.device
            height, width = feature.shape[-2:]
            y_coords = torch.arange(height, device=device, dtype=dtype).add(
                grid_cell_offset
            )
            x_coords = torch.arange(width, device=device, dtype=dtype).add(
                grid_cell_offset
            )
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            centers = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
            stride_value = torch.as_tensor(stride, device=device, dtype=dtype)
            centers_by_level.append(centers)
            stride_by_level.append(stride_value.expand(centers.shape[0], 1))

        return torch.cat(centers_by_level, dim=0), torch.cat(stride_by_level, dim=0)

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


class DDetectSeg(DDetect):
    """YOLO9 detection head with YOLACT-style prototype mask prediction."""

    def __init__(
        self,
        nc=80,
        ch=(),
        reg_max=16,
        stride=(),
        use_group=True,
        num_masks=32,
        proto_channels=256,
    ):
        super().__init__(nc=nc, ch=ch, reg_max=reg_max, stride=stride, use_group=use_group)
        self.nm = num_masks
        self.proto = MaskProto(ch[0], proto_channels, self.nm)
        self._seg_loss_fn = None

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1))
            for x in ch
        )

    def _get_seg_loss_fn(self, device):
        if self._seg_loss_fn is None:
            from .loss import YOLO9SegmentationLoss

            self._seg_loss_fn = YOLO9SegmentationLoss(
                num_classes=self.nc,
                reg_max=self.reg_max,
                strides=self.stride.tolist(),
                image_size=None,
                device=device,
                num_masks=self.nm,
            )
        return self._seg_loss_fn

    def forward(self, x, targets=None, img_size=None, masks=None):
        features = list(x)
        proto = self.proto(features[0])
        batch_size = proto.shape[0]
        mask_coeffs = torch.cat(
            [
                self.cv4[i](features[i]).view(batch_size, self.nm, -1)
                for i in range(self.nl)
            ],
            dim=2,
        )

        det_outputs = super().forward(features, targets=None, img_size=img_size)

        if self.training:
            if targets is not None:
                loss_fn = self._get_seg_loss_fn(proto.device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(det_outputs, targets, mask_coeffs, proto, masks)
            return det_outputs, mask_coeffs, proto

        predictions, raw_outputs = det_outputs
        if self.export:
            return predictions, proto, mask_coeffs
        return predictions, raw_outputs, proto, mask_coeffs


# =============================================================================
# Model Architecture Definitions
# =============================================================================

# YOLOv9 configurations - exact channel dimensions from official YOLO configs
# Each variant has unique, non-linear channel structures
YOLO9_CONFIGS = {
    "t": {  # Tiny
        # Backbone: Conv(16) -> Conv(32) -> ELAN(32) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 16,
        "conv1_out": 32,
        "first_block": "elan",  # ELAN for t/s, RepNCSPELAN for m/c
        "first_block_out": 32,
        "down_type": "aconv",  # AConv for t/s/m, ADown for c
        "stages": [  # (down_out, elan_out, elan_part) for stages 2, 3, 4
            (64, 64, 64),  # B3: AConv->64, RepNCSPELAN->64
            (96, 96, 96),  # B4: AConv->96, RepNCSPELAN->96
            (128, 128, 128),  # B5: AConv->128, RepNCSPELAN->128
        ],
        "spp_out": 128,
        "repeat_num": 3,  # RepNCSPELAN repeat for t/s
        # Neck
        "neck_elan_up1": (96, 96),  # N4: out=96, part=96
        "neck_elan_up2": (64, 64),  # P3: out=64, part=64
        "neck_down1_out": 48,  # AConv after P3
        "neck_elan_down1": (96, 96),  # P4: out=96, part=96
        "neck_down2_out": 64,  # AConv after P4
        "neck_elan_down2": (128, 128),  # P5: out=128, part=128
        # Detection head channels
        "head_channels": (64, 96, 128),  # P3, P4, P5
    },
    "s": {  # Small
        # Backbone: Conv(32) -> Conv(64) -> ELAN(64) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "elan",
        "first_block_out": 64,
        "down_type": "aconv",
        "stages": [
            (128, 128, 128),  # B3
            (192, 192, 192),  # B4
            (256, 256, 256),  # B5
        ],
        "spp_out": 256,
        "repeat_num": 3,
        # Neck
        "neck_elan_up1": (192, 192),
        "neck_elan_up2": (128, 128),
        "neck_down1_out": 96,
        "neck_elan_down1": (192, 192),
        "neck_down2_out": 128,
        "neck_elan_down2": (256, 256),
        # Detection
        "head_channels": (128, 192, 256),
    },
    "m": {  # Medium
        # Backbone: Conv(32) -> Conv(64) -> RepNCSPELAN(128) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "repncspelan",
        "first_block_out": 128,
        "first_block_part": 128,
        "down_type": "aconv",
        "stages": [
            (240, 240, 240),  # B3
            (360, 360, 360),  # B4
            (480, 480, 480),  # B5
        ],
        "spp_out": 480,
        "repeat_num": 1,  # Default repeat for m/c
        # Neck
        "neck_elan_up1": (360, 360),
        "neck_elan_up2": (240, 240),
        "neck_down1_out": 184,
        "neck_elan_down1": (360, 360),
        "neck_down2_out": 240,
        "neck_elan_down2": (480, 480),
        # Detection
        "head_channels": (240, 360, 480),
    },
    "c": {  # Compact (largest)
        # Backbone: Conv(64) -> Conv(128) -> RepNCSPELAN(256) -> [ADown -> RepNCSPELAN] x3
        "conv0_out": 64,
        "conv1_out": 128,
        "first_block": "repncspelan",
        "first_block_out": 256,
        "first_block_part": 128,  # part_channels for first RepNCSPELAN
        "down_type": "adown",
        "stages": [
            (256, 512, 256),  # B3: ADown->256, RepNCSPELAN->512, part=256
            (512, 512, 512),  # B4: ADown->512, RepNCSPELAN->512, part=512
            (512, 512, 512),  # B5: ADown->512, RepNCSPELAN->512, part=512
        ],
        "spp_out": 512,
        "repeat_num": 1,
        # Neck
        "neck_elan_up1": (512, 512),
        "neck_elan_up2": (256, 256),
        "neck_down1_out": 256,
        "neck_elan_down1": (512, 512),
        "neck_down2_out": 512,
        "neck_elan_down2": (512, 512),
        # Detection
        "head_channels": (256, 512, 512),
    },
}


class Backbone9(nn.Module):
    """YOLOv9 Backbone.

    Supports all variants with their specific architectures:
    - yolo9-t/s: Conv -> Conv -> ELAN -> [AConv -> RepNCSPELAN] x3 -> SPPELAN
    - yolo9-m/c: Conv -> Conv -> RepNCSPELAN -> [AConv/ADown -> RepNCSPELAN] x3 -> SPPELAN
    """

    def __init__(self, config="c"):
        super().__init__()

        cfg = YOLO9_CONFIGS[config]
        self.config = config

        # Stem
        self.conv0 = Conv(3, cfg["conv0_out"], 3, 2)
        self.conv1 = Conv(cfg["conv0_out"], cfg["conv1_out"], 3, 2)

        # First block (ELAN for t/s, RepNCSPELAN for m/c)
        if cfg["first_block"] == "elan":
            # ELAN(c1, c2, c3, c4, n) where:
            #   c1 = input channels
            #   c2 = cv1 output (part_channels)
            #   c3 = cv2/cv3 output (part_channels // 2)
            #   c4 = output channels
            # For yolo9-t/s: ELAN {out_channels: X, part_channels: X}
            c1 = cfg["conv1_out"]
            c4 = cfg["first_block_out"]
            part = c4  # part_channels = out_channels for t/s ELAN
            self.elan1 = ELAN(c1, part, part // 2, c4, n=1)
        else:
            # RepNCSPELAN for m/c
            # RepNCSPELAN(c1, c2, c3, c4, n) where:
            #   c1 = input channels
            #   c2 = cv1 output = part_channels (gets split in half)
            #   c3 = cv2/cv3 internal = part_channels // 2
            #   c4 = output channels
            c1 = cfg["conv1_out"]
            c4 = cfg["first_block_out"]
            part = cfg.get("first_block_part", c4)
            self.elan1 = RepNCSPELAN(c1, part, part // 2, c4, cfg["repeat_num"])

        # Determine downsampling block type
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv
        n = cfg["repeat_num"]

        # Stage 2 (B3) - first stage after initial block
        # stage = (down_out, elan_out, part_channels)
        stage = cfg["stages"][0]
        prev_ch = cfg["first_block_out"]
        self.down2 = DownBlock(prev_ch, stage[0])
        # RepNCSPELAN: c1=down_out, c2=part, c3=part//2, c4=out
        self.elan2 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # Stage 3 (B4)
        stage = cfg["stages"][1]
        prev_ch = cfg["stages"][0][1]  # Previous elan output
        self.down3 = DownBlock(prev_ch, stage[0])
        self.elan3 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # Stage 4 (B5)
        stage = cfg["stages"][2]
        prev_ch = cfg["stages"][1][1]
        self.down4 = DownBlock(prev_ch, stage[0])
        self.elan4 = RepNCSPELAN(stage[0], stage[2], stage[2] // 2, stage[1], n)

        # SPP
        spp_in = cfg["stages"][2][1]
        spp_out = cfg["spp_out"]
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

    def __init__(self, config="c"):
        super().__init__()

        cfg = YOLO9_CONFIGS[config]
        self.config = config
        n = cfg["repeat_num"]

        # Get backbone output channels for concatenation
        b3_ch = cfg["stages"][0][1]  # B3 output channels
        b4_ch = cfg["stages"][1][1]  # B4 output channels
        spp_ch = cfg["spp_out"]  # SPP/P5 output channels

        # Top-down path
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        # Concat(SPP_up, B4) -> N4
        up1_in = spp_ch + b4_ch
        up1_out, up1_part = cfg["neck_elan_up1"]
        # RepNCSPELAN: c1=concat_in, c2=part, c3=part//2, c4=out
        self.elan_up1 = RepNCSPELAN(up1_in, up1_part, up1_part // 2, up1_out, n)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Concat(N4_up, B3) -> P3
        up2_in = up1_out + b3_ch
        up2_out, up2_part = cfg["neck_elan_up2"]
        self.elan_up2 = RepNCSPELAN(up2_in, up2_part, up2_part // 2, up2_out, n)

        # Bottom-up path
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv

        # P3 -> down -> Concat(N4) -> P4
        p3_out = up2_out
        self.down1 = DownBlock(p3_out, cfg["neck_down1_out"])
        down1_concat_in = cfg["neck_down1_out"] + up1_out
        down1_out, down1_part = cfg["neck_elan_down1"]
        self.elan_down1 = RepNCSPELAN(
            down1_concat_in, down1_part, down1_part // 2, down1_out, n
        )

        # P4 -> down -> Concat(SPP) -> P5
        p4_out = down1_out
        self.down2 = DownBlock(p4_out, cfg["neck_down2_out"])
        down2_concat_in = cfg["neck_down2_out"] + spp_ch
        down2_out, down2_part = cfg["neck_elan_down2"]
        self.elan_down2 = RepNCSPELAN(
            down2_concat_in, down2_part, down2_part // 2, down2_out, n
        )

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

    Supports yolo9-t, yolo9-s, yolo9-m, and yolo9-c variants with their specific architectures.
    """

    def __init__(
        self,
        config="c",
        reg_max=16,
        nb_classes=80,
        img_size=640,
        segmentation=False,
        num_masks=32,
        proto_channels=256,
    ):
        """
        Initialize YOLOv9 model.

        Args:
            config: Model size ('t', 's', 'm', 'c')
            reg_max: Regression max value for DFL
            nb_classes: Number of classes
            img_size: Input image size
        """
        super().__init__()

        if config not in YOLO9_CONFIGS:
            raise ValueError(
                f"Invalid config: {config}. Must be one of: {list(YOLO9_CONFIGS.keys())}"
            )

        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max
        self.img_size = img_size
        self.segmentation = segmentation

        cfg = YOLO9_CONFIGS[config]

        self.backbone = Backbone9(config)
        self.neck = Neck9(config)

        # Detection head - use exact channels from config
        head_channels = cfg["head_channels"]
        head_cls = DDetectSeg if segmentation else DDetect
        head_kwargs = {
            "nc": nb_classes,
            "ch": head_channels,
            "reg_max": reg_max,
            "stride": (8, 16, 32),
        }
        if segmentation:
            head_kwargs.update({"num_masks": num_masks, "proto_channels": proto_channels})
        self.head = head_cls(**head_kwargs)

    def forward(self, x, targets=None, masks=None):
        """
        Forward pass through backbone, neck, and detection head.

        Args:
            x: Input tensor [B, 3, H, W]
            targets: Optional ground truth [B, max_targets, 5] with [class, x1, y1, x2, y2] normalized
                    Only used during training to compute loss.

        Returns:
            Training with targets: Dict with loss values (total_loss, box_loss, dfl_loss, cls_loss)
            Training without targets: Raw predictions (list of tensors)
            Inference: Dict with decoded predictions and features
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(p3, p4, p5)

        # Detection head
        if self.training and targets is not None:
            # Pass image size for anchor generation
            img_size = (x.shape[3], x.shape[2])  # (W, H)
            output = self.head(
                [n3, n4, n5],
                targets=targets,
                img_size=img_size,
                **({"masks": masks} if self.segmentation else {}),
            )
            return output

        # Normal forward (training without targets or inference)
        output = self.head([n3, n4, n5])

        if self.training:
            # Return raw outputs for loss calculation
            return output

        # Inference mode
        if self.segmentation:
            if self.head.export:
                return output
            y, x_list, proto, mask_coeffs = output
        else:
            y, x_list = output

        # Export mode: return only the prediction tensor for ONNX/TorchScript
        if self.head.export:
            return y

        result = {
            "predictions": y,  # (batch, 4+nc, total_anchors)
            "raw_outputs": x_list,
            "x8": {"features": n3},
            "x16": {"features": n4},
            "x32": {"features": n5},
        }
        if self.segmentation:
            result["proto"] = proto
            result["mask_coeffs"] = mask_coeffs
        return result

    def fuse(self):
        """Fuse Conv+BN and RepConvN for faster inference."""
        for m in self.modules():
            if isinstance(m, RepConvN):
                m.fuse_convs()
            elif isinstance(m, Conv) and hasattr(m, "bn"):
                # Fuse Conv+BN
                m.conv = self._fuse_conv_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_without_bn
        return self

    def _fuse_conv_bn(self, conv, bn):
        """Fuse Conv2d and BatchNorm2d."""
        fusedconv = (
            nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        b_conv = (
            torch.zeros(conv.weight.size(0), device=conv.weight.device)
            if conv.bias is None
            else conv.bias
        )
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv
