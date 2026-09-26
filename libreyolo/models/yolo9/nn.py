"""
Neural network architecture for LibreYOLO yolo9.

Supports yolo9-t (tiny), yolo9-s (small), yolo9-m (medium), and yolo9-c (compact/largest).

The architecture blocks (Conv, ELAN/RepNCSPELAN, AConv/ADown, SPPELAN) and the
detection head are ported and adapted from MultimediaTechLab/YOLO
(https://github.com/MultimediaTechLab/YOLO, MIT License, Copyright (c) 2024
Kin-Yiu, Wong and Hao-Tang, Tsui), the official MIT release of YOLOv9 — see
``yolo/model/module.py`` and ``yolo/utils/bounding_box_utils.py`` there. The
RepConv fuse/re-parameterization logic follows DingXiaoH/RepVGG (MIT License).

LibreYOLO keeps its own module layout and state-dict key names
(``backbone.*`` / ``neck.*`` / ``head.cv2|cv3|dfl``): they are the published
LibreYOLO9 checkpoint format, and :mod:`libreyolo.models.yolo9.convert` maps
upstream checkpoints onto it.
"""

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
    """Integral over a discrete per-side distance distribution.

    Follows the softmax-projection integral of the in-tree yolonas heads
    (``proj_conv`` in :mod:`libreyolo.models.yolonas.nn`, ported from
    Deci-AI/super-gradients, Apache-2.0), which realizes the Distribution
    Focal Loss expectation from the Generalized Focal Loss paper.
    """

    def __init__(self, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer(
            "project",
            torch.arange(num_bins, dtype=torch.float32).view(1, 1, num_bins, 1),
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
        batch, _, num_anchors = x.shape
        dist = x.view(batch, 4, self.num_bins, num_anchors)
        return F.softmax(dist, dim=2).mul(self.project.to(dtype=x.dtype)).sum(2)


class DDetect(nn.Module):
    """
    Anchor-free, decoupled detection head for yolo9.

    Ported and adapted from MultimediaTechLab/YOLO (MIT): each feature-map
    scale gets a box-regression tower and a class-score tower — the
    ``anchor_conv`` / ``class_conv`` pair of ``yolo.model.module.Detection``,
    replicated per scale as in ``MultiheadDetection`` — and inference decodes
    LTRB distances against the anchor grid the way
    ``yolo.utils.bounding_box_utils.Vec2Box`` does. The DFL integral over
    ``reg_max`` bins (:class:`DFL`) turns per-side distributions into
    distances first.

    Checkpoint-format contract: the towers live in the ``cv2`` / ``cv3``
    ``ModuleList``s and the integral module at ``dfl``. These attribute names
    and the tower widths define the published LibreYOLO9 state-dict layout —
    renaming them breaks every released and user-trained checkpoint, so treat
    them as frozen API.

    Supports training mode with loss computation when targets are provided.
    """

    @staticmethod
    def _round_up(value, divisor):
        """Smallest multiple of ``divisor`` that is >= ``value``."""
        return ((value + divisor - 1) // divisor) * divisor

    @classmethod
    def _box_branch_width(cls, input_channels, groups, output_channels, reg_max):
        """Box-tower hidden width, as in MMT ``Detection``: a quarter of the
        first scale's channels rounded up to the group count, floored at the
        distribution width."""
        return max(
            cls._round_up(input_channels // 4, groups),
            output_channels,
            reg_max,
        )

    @staticmethod
    def _build_box_towers(input_channels, hidden_channels, output_channels, groups):
        """Per-scale box-regression towers (MMT ``Detection.anchor_conv``)."""
        return nn.ModuleList(
            nn.Sequential(
                Conv(channels, hidden_channels, 3),
                Conv(hidden_channels, hidden_channels, 3, groups=groups),
                nn.Conv2d(hidden_channels, output_channels, 1, groups=groups),
            )
            for channels in input_channels
        )

    @staticmethod
    def _build_class_towers(input_channels, hidden_channels, num_classes):
        """Per-scale class-score towers (MMT ``Detection.class_conv``)."""
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

        # Inference anchor-grid cache. ``dynamic`` forces a rebuild every
        # forward; ``export`` bypasses the cache entirely so traced graphs
        # stay shape-consistent. ``anchors``/``strides`` are read externally
        # after a warm-up forward (see export/coreml.py).
        self.dynamic = False
        self.export = False
        self.shape = None
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
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
        # Preserve the architecture's fixed stride values as Python scalars for
        # graph export. Iterating over the tensor buffer and calling float()
        # introduces aten.item/sym_float nodes that edge runtimes cannot lower.
        self._stride_values = tuple(float(value) for value in stride_tensor.tolist())

        self._loss_fn = None

        self._box_groups = 4 if use_group else 1
        self._box_output_channels = 4 * reg_max
        self._box_hidden_channels = self._box_branch_width(
            ch[0], self._box_groups, self._box_output_channels, reg_max
        )
        # The class-tower hidden width is part of the published LibreYOLO9
        # checkpoint geometry: the first scale's channel count, floored at
        # ``min(nc, 100)`` (width 80 for the released COCO models). Keep it —
        # it must reproduce the tensor shapes of released and user-trained
        # checkpoints. Checkpoints whose width differs (e.g. COCO-width
        # towers reused under a new ``nc``) are handled by the rebuild
        # helpers in ``model.py``, which read the width straight from the
        # checkpoint tensors.
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
        """Detection-prior bias init, as in MMT ``Detection.__init__``: box
        towers start at 1.0, class towers at -10 (a ~4.5e-5 foreground prior
        after sigmoid). Only affects training from scratch — any loaded
        checkpoint overwrites these."""
        for box_tower, class_tower in zip(self.cv2, self.cv3):
            box_tower[-1].bias.data.fill_(1.0)
            class_tower[-1].bias.data.fill_(-10.0)

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
        preds = [
            torch.cat((self.cv2[i](feat), self.cv3[i](feat)), 1)
            for i, feat in enumerate(x)
        ]

        if self.training:
            if targets is not None:
                # Compute loss
                loss_fn = self._get_loss_fn(preds[0].device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(preds, targets)
            return preds

        return self._decode_inference(preds), preds

    def _anchor_grid(self, feats):
        """Grid-cell centers and per-cell stride for the live feature maps.

        Adapted from ``generate_anchors_for_grid_cell`` in
        :mod:`libreyolo.models.yolonas.nn` (ported there from
        Deci-AI/super-gradients, Apache-2.0), kept in grid units so the
        matching stride scales boxes back to pixels at decode time.
        """
        anchor_points = []
        stride_scale = []
        dtype, device = feats[0].dtype, feats[0].device
        for feat, stride in zip(feats, self._stride_values):
            _, _, h, w = feat.shape
            shift_x = torch.arange(end=w, device=device, dtype=dtype) + 0.5
            shift_y = torch.arange(end=h, device=device, dtype=dtype) + 0.5
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_points.append(
                torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
            )
            stride_scale.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )
        return torch.cat(anchor_points), torch.cat(stride_scale)

    def _grid(self, feats):
        if self.export:
            anchor_points, stride_scale = self._anchor_grid(feats)
            return anchor_points.transpose(0, 1), stride_scale.transpose(0, 1)
        shape = feats[0].shape
        cached = not self.dynamic and self.shape == shape
        if not cached:
            anchor_points, stride_scale = self._anchor_grid(feats)
            self.anchors = anchor_points.transpose(0, 1)
            self.strides = stride_scale.transpose(0, 1)
            self.shape = shape
        return self.anchors, self.strides

    def _decode_inference(self, preds):
        anchor_points, stride_scale = self._grid(preds)
        box_levels = []
        score_levels = []
        for pred in preds:
            box_level, score_level = pred.flatten(2).split(
                (4 * self.reg_max, self.nc), 1
            )
            box_levels.append(box_level)
            score_levels.append(score_level)
        distances = self.dfl(torch.cat(box_levels, 2))
        boxes = (
            self._decode_bboxes(distances, anchor_points.unsqueeze(0)) * stride_scale
        )
        scores = torch.cat(score_levels, 2).sigmoid()
        return torch.cat((boxes, scores), 1)

    def _decode_bboxes(self, distances, anchor_points):
        """Decode LTRB distances into xyxy boxes, as in MMT ``Vec2Box``: the
        box corners are the anchor center minus the left-top pair and plus
        the right-bottom pair.

        Args:
            distances: (batch, 4, anchors) - l, t, r, b distances from anchor
            anchor_points: (1, 2, anchors) - anchor center coordinates
        """
        left_top, right_bottom = distances.chunk(2, 1)
        return torch.cat(
            (anchor_points - left_top, anchor_points + right_bottom), 1
        )


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

        # Stage 4 - B5 (pre-SPP) then SPP → P5. Stash B5 for the optional
        # PGI aux neck without changing this method's 3-tuple return.
        x = self.down4(p4)
        self.last_b5 = self.elan4(x)
        p5 = self.spp(self.last_b5)

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


class AuxNeck(nn.Module):
    """PGI auxiliary FPN used only during training.

    Mirrors MultimediaTechLab/YOLO ``auxiliary`` in ``v9-*.yaml``: SPPELAN
    on pre-SPP B5, then top-down concat with B4/B3. Inference never calls
    this module, so old single-head checkpoints keep their exact graph.
    """

    def __init__(self, config="c"):
        super().__init__()
        cfg = YOLO9_CONFIGS[config]
        n = cfg["repeat_num"]
        b3_ch = cfg["stages"][0][1]
        b4_ch = cfg["stages"][1][1]
        b5_ch = cfg["stages"][2][1]
        spp_out = cfg["spp_out"]

        self.spp = SPPELAN(b5_ch, spp_out // 2, spp_out)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        a4_out, a4_part = cfg["neck_elan_up1"]
        self.elan_a4 = RepNCSPELAN(
            spp_out + b4_ch, a4_part, a4_part // 2, a4_out, n
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        a3_out, a3_part = cfg["neck_elan_up2"]
        self.elan_a3 = RepNCSPELAN(
            a4_out + b3_ch, a3_part, a3_part // 2, a3_out, n
        )

    def forward(self, p3, p4, b5):
        a5 = self.spp(b5)
        x = self.up1(a5)
        x = torch.cat([x, p4], 1)
        a4 = self.elan_a4(x)
        x = self.up2(a4)
        x = torch.cat([x, p3], 1)
        a3 = self.elan_a3(x)
        return a3, a4, a5


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

        cfg = YOLO9_CONFIGS[config]

        self.backbone = Backbone9(config)
        self.neck = Neck9(config)

        # Detection head - use exact channels from config
        head_channels = cfg["head_channels"]
        self.head = DDetect(
            nc=nb_classes,
            ch=head_channels,
            reg_max=reg_max,
            stride=(8, 16, 32),
        )
        # Built only when training with PGI. Inference checkpoints stay
        # single-head so a 1.6 upgrade does not change the exported graph.
        self.aux = None
        self.aux_head = None
        self.aux_weight = 0.0

    def enable_aux(self, weight: float = 0.25):
        """Attach the PGI auxiliary neck/head if they are not already present."""
        self.aux_weight = float(weight)
        if self.aux is not None:
            return self
        cfg = YOLO9_CONFIGS[self.config]
        self.aux = AuxNeck(self.config)
        self.aux_head = DDetect(
            nc=self.nc,
            ch=cfg["head_channels"],
            reg_max=self.reg_max,
            stride=(8, 16, 32),
        )
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.aux.to(device)
        self.aux_head.to(device)
        return self

    def combine_aux_losses(self, main: dict, aux: dict) -> dict:
        """Add the PGI auxiliary losses to the main ones at ``aux_weight``."""
        combined = dict(main)
        for key in ("total_loss", "box_loss", "dfl_loss", "cls_loss", "box", "dfl", "cls"):
            if key in main and key in aux:
                combined[key] = main[key] + self.aux_weight * aux[key]
        return combined

    def forward(self, x, targets=None):
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
        # Backbone. ``last_b5`` is the pre-SPP B5 feature the PGI aux
        # branch needs; the public 3-tuple return stays (p3, p4, post-SPP p5).
        p3, p4, p5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(p3, p4, p5)

        # Detection head
        if self.training and targets is not None:
            # Pass image size for anchor generation
            img_size = (x.shape[3], x.shape[2])  # (W, H)
            main = self.head([n3, n4, n5], targets=targets, img_size=img_size)
            if self.aux is None or self.aux_weight <= 0:
                return main
            b5 = getattr(self.backbone, "last_b5", None)
            if b5 is None:
                return main
            a3, a4, a5 = self.aux(p3, p4, b5)
            aux = self.aux_head([a3, a4, a5], targets=targets, img_size=img_size)
            return self.combine_aux_losses(main, aux)

        # Normal forward (training without targets or inference)
        output = self.head([n3, n4, n5])

        if self.training:
            # Return raw outputs for loss calculation
            return output

        # Inference mode
        y, x_list = output

        # Export mode: return only the prediction tensor for ONNX/TorchScript
        if self.head.export:
            return y

        return {
            "predictions": y,  # (batch, 4+nc, total_anchors)
            "raw_outputs": x_list,
            "x8": {"features": n3},
            "x16": {"features": n4},
            "x32": {"features": n5},
        }

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
