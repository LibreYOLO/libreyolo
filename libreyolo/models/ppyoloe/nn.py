"""PP-YOLOE network: CSPResNet backbone, CSP-PAN neck, Efficient Task-aligned head.

Adapted from Deci-AI/super-gradients (Apache-2.0), commit
``63de22c404d5740f34f7706c302b37fce3c8fe5d``:

- ``src/super_gradients/training/models/detection_models/csp_resnet.py``
- ``src/super_gradients/training/models/detection_models/pp_yolo_e/pan.py``
- ``src/super_gradients/training/models/detection_models/pp_yolo_e/pp_yolo_head.py``
- ``src/super_gradients/training/models/detection_models/pp_yolo_e/pp_yolo_e.py``
- ``src/super_gradients/modules/{conv_bn_act_block,repvgg_block,se_blocks}.py``
- ``src/super_gradients/training/utils/bbox_utils.py``

Module and attribute names mirror upstream so the released checkpoints load
after nothing more than stripping a leading ``module.`` prefix.

Head output contract:

- ``train()``: ``(cls_logits, reg_distri, anchors, anchor_points,
  num_anchors_list, stride_tensor)`` in stride order 32, 16, 8. This is the
  tuple ``libreyolo.models.yolonas.loss.PPYoloELoss`` consumes.
- ``eval()``: ``((boxes_xyxy, scores), raw)`` where ``boxes_xyxy`` is in input
  canvas pixels and ``scores`` is sigmoid class probability. There is no
  objectness term.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "PPYOLOE_CONFIGS",
    "LibrePPYOLOEModel",
    "PPYOLOEHead",
    "CSPResNetBackbone",
    "PPYoloECSPPAN",
]


# Depth / width multipliers per released size (upstream
# ``recipes/arch_params/ppyoloe_{s,m,l,x}_arch_params.yaml``).
PPYOLOE_CONFIGS: Dict[str, Dict[str, float]] = {
    "s": {"depth_mult": 0.33, "width_mult": 0.50},
    "m": {"depth_mult": 0.67, "width_mult": 0.75},
    "l": {"depth_mult": 1.00, "width_mult": 1.00},
    "x": {"depth_mult": 1.33, "width_mult": 1.25},
}

# Base (width_mult=1.0) shapes from ``ppyoloe_arch_params.yaml``.
_BACKBONE_LAYERS = (3, 6, 6, 3)
_BACKBONE_CHANNELS = (64, 128, 256, 512, 1024)
_BACKBONE_RETURN_IDX = (1, 2, 3)
_NECK_IN_CHANNELS = (256, 512, 1024)
_NECK_OUT_CHANNELS = (768, 384, 192)
_NECK_STAGE_NUM = 1
_NECK_BLOCK_NUM = 3
_FPN_STRIDES = (32, 16, 8)
_GRID_CELL_SCALE = 5.0
_GRID_CELL_OFFSET = 0.5
_REG_MAX = 16


def _scale_channels(channels, width_mult: float) -> List[int]:
    return [max(round(c * width_mult), 1) for c in channels]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> activation, kept under a ``seq`` submodule.

    The ``seq`` nesting is not cosmetic: upstream checkpoint keys are
    ``...seq.conv.weight`` / ``...seq.bn.weight``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        activation_type: Type[nn.Module],
        stride=1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
        )
        self.seq.add_module("bn", nn.BatchNorm2d(out_channels))
        if activation_type is not None:
            self.seq.add_module("act", activation_type())

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class RepVGGBlock(nn.Module):
    """RepVGG block as PP-YOLOE uses it: 3x3 + 1x1 branches, no SE, no identity.

    PP-YOLOE builds every RepVGG block with ``use_residual_connection=False``
    and ``use_alpha=False``, so the identity BatchNorm branch and the learnable
    ``alpha`` of PP-YOLOE-Plus are never materialised here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: Type[nn.Module],
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.nonlinearity = activation_type()
        self.branch_3x3 = self._conv_bn(in_channels, out_channels, 3, stride, 1)
        self.branch_1x1 = self._conv_bn(in_channels, out_channels, 1, stride, 0)
        self.build_residual_branches = True

    @staticmethod
    def _conv_bn(in_channels, out_channels, kernel_size, stride, padding):
        seq = nn.Sequential()
        seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )
        seq.add_module("bn", nn.BatchNorm2d(out_channels))
        return seq

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.build_residual_branches:
            return self.nonlinearity(self.rbr_reparam(inputs))
        return self.nonlinearity(self.branch_3x3(inputs) + self.branch_1x1(inputs))

    def _fuse_bn_tensor(self, branch: nn.Sequential):
        kernel = branch.conv.weight
        bn = branch.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def fuse_block_residual_branches(self) -> None:
        """Collapse the two branches into one 3x3 conv (deployment form)."""
        if not self.build_residual_branches:
            return
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernel = kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1])
        bias = bias3x3 + bias1x1

        conv = self.branch_3x3.conv
        self.rbr_reparam = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for param in self.parameters():
            param.detach_()
        del self.branch_3x3
        del self.branch_1x1
        self.build_residual_branches = False


class EffectiveSEBlock(nn.Module):
    """Effective Squeeze-Excitation (CenterMask), as used by CSPResStage."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.project(x_se)
        return x * self.act(x_se)


class CSPResNetBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_type: Type[nn.Module],
        use_residual_connection: bool = True,
    ) -> None:
        super().__init__()
        if use_residual_connection and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels "
                f"({out_channels}) when use_residual_connection=True"
            )
        self.conv1 = ConvBNAct(
            in_channels, out_channels, 3, stride=1, padding=1,
            activation_type=activation_type, bias=False,
        )
        self.conv2 = RepVGGBlock(
            out_channels, out_channels, activation_type=activation_type
        )
        self.use_residual_connection = use_residual_connection

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.use_residual_connection else y


class CSPResStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        activation_type: Type[nn.Module],
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        half_mid_channels = mid_channels // 2
        mid_channels = 2 * half_mid_channels

        if stride != 1:
            self.conv_down = ConvBNAct(
                in_channels, mid_channels, 3, stride=stride, padding=1,
                activation_type=activation_type, bias=False,
            )
        else:
            self.conv_down = None
        self.conv1 = ConvBNAct(
            mid_channels, half_mid_channels, 1, stride=1, padding=0,
            activation_type=activation_type, bias=False,
        )
        self.conv2 = ConvBNAct(
            mid_channels, half_mid_channels, 1, stride=1, padding=0,
            activation_type=activation_type, bias=False,
        )
        self.blocks = nn.Sequential(
            *[
                CSPResNetBasicBlock(
                    half_mid_channels, half_mid_channels,
                    activation_type=activation_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.attn = EffectiveSEBlock(mid_channels) if use_attention else nn.Identity()
        self.conv3 = ConvBNAct(
            mid_channels, out_channels, 1, stride=1, padding=0,
            activation_type=activation_type, bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        y = self.attn(y)
        return self.conv3(y)


class CSPResNetBackbone(nn.Module):
    """CSPResNet backbone returning strides 8, 16 and 32."""

    def __init__(
        self,
        layers=_BACKBONE_LAYERS,
        channels=_BACKBONE_CHANNELS,
        activation: Type[nn.Module] = nn.SiLU,
        return_idx=_BACKBONE_RETURN_IDX,
        use_large_stem: bool = True,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        channels = _scale_channels(channels, width_mult)
        layers = [max(round(n * depth_mult), 1) for n in layers]

        if use_large_stem:
            stem = [
                ("conv1", ConvBNAct(in_channels, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation, bias=False)),
                ("conv2", ConvBNAct(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, activation_type=activation, bias=False)),
                ("conv3", ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation, bias=False)),
            ]
        else:
            stem = [
                ("conv1", ConvBNAct(in_channels, channels[0] // 2, 3, stride=2, padding=1, activation_type=activation, bias=False)),
                ("conv2", ConvBNAct(channels[0] // 2, channels[0], 3, stride=1, padding=1, activation_type=activation, bias=False)),
            ]
        self.stem = nn.Sequential(collections.OrderedDict(stem))

        n = len(channels) - 1
        self.stages = nn.ModuleList(
            [
                CSPResStage(
                    channels[i], channels[i + 1], layers[i], stride=2,
                    activation_type=activation,
                )
                for i in range(n)
            ]
        )
        self._out_channels = channels[1:]
        self.return_idx = tuple(return_idx)

    @property
    def out_channels(self) -> Tuple[int, ...]:
        return tuple(self._out_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs: List[Tensor] = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


class PPYoloESPP(nn.Module):
    """Spatial pyramid pooling with kernels 5, 9 and 13, stride 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: Tuple[int, ...],
        activation_type: Type[nn.Module],
    ) -> None:
        super().__init__()
        mid_channels = in_channels * (1 + len(pool_size))
        self.pool = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
                for size in pool_size
            ]
        )
        self.conv = ConvBNAct(
            mid_channels, out_channels, kernel_size, padding=kernel_size // 2,
            activation_type=activation_type, stride=1, bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        return self.conv(torch.cat(outs, dim=1))


class CSPStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int,
        activation_type: Type[nn.Module],
        spp: bool,
    ) -> None:
        super().__init__()
        ch_mid = int(out_channels // 2)
        self.conv1 = ConvBNAct(in_channels, ch_mid, 1, padding=0, activation_type=activation_type, stride=1, bias=False)
        self.conv2 = ConvBNAct(in_channels, ch_mid, 1, padding=0, activation_type=activation_type, stride=1, bias=False)

        convs = []
        next_ch_in = ch_mid
        for i in range(n):
            convs.append(
                (
                    str(i),
                    CSPResNetBasicBlock(
                        next_ch_in, ch_mid,
                        activation_type=activation_type,
                        use_residual_connection=False,
                    ),
                )
            )
            if i == (n - 1) // 2 and spp:
                convs.append(("spp", PPYoloESPP(ch_mid, ch_mid, 1, (5, 9, 13), activation_type=activation_type)))
            next_ch_in = ch_mid

        self.convs = nn.Sequential(collections.OrderedDict(convs))
        self.conv3 = ConvBNAct(ch_mid * 2, out_channels, 1, padding=0, activation_type=activation_type, stride=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.conv1(x)
        y2 = self.convs(self.conv2(x))
        return self.conv3(torch.cat([y1, y2], dim=1))


class PPYoloECSPPAN(nn.Module):
    """CSP-PAN neck: top-down FPN then bottom-up PAN, SPP on the deepest stage."""

    def __init__(
        self,
        in_channels=_NECK_IN_CHANNELS,
        out_channels=_NECK_OUT_CHANNELS,
        activation: Type[nn.Module] = nn.SiLU,
        stage_num: int = _NECK_STAGE_NUM,
        block_num: int = _NECK_BLOCK_NUM,
        spp: bool = True,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
    ) -> None:
        super().__init__()
        in_channels = _scale_channels(in_channels, width_mult)
        out_channels = _scale_channels(out_channels, width_mult)
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length")

        block_num = max(round(block_num * depth_mult), 1)
        self.num_blocks = len(in_channels)
        self._out_channels = out_channels
        in_channels = in_channels[::-1]

        fpn_stages, fpn_routes = [], []
        ch_pre = None
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2
            stage = [
                (
                    str(j),
                    CSPStage(
                        ch_in if j == 0 else ch_out, ch_out, block_num,
                        activation_type=activation, spp=(spp and i == 0),
                    ),
                )
                for j in range(stage_num)
            ]
            fpn_stages.append(nn.Sequential(collections.OrderedDict(stage)))
            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNAct(ch_out, ch_out // 2, 1, stride=1, padding=0, activation_type=activation, bias=False)
                )
            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages, pan_routes = [], []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNAct(
                    out_channels[i + 1], out_channels[i + 1], 3, stride=2, padding=1,
                    activation_type=activation, bias=False,
                )
            )
            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = [
                (
                    str(j),
                    CSPStage(
                        ch_in if j == 0 else ch_out, ch_out, block_num,
                        activation_type=activation, spp=False,
                    ),
                )
                for j in range(stage_num)
            ]
            pan_stages.append(nn.Sequential(collections.OrderedDict(stage)))

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    @property
    def out_channels(self) -> Tuple[int, ...]:
        return tuple(self._out_channels)

    def forward(self, blocks: List[Tensor]) -> List[Tensor]:
        blocks = blocks[::-1]
        fpn_feats: List[Tensor] = []
        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], dim=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)
            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2, mode="nearest")

        pan_feats = [fpn_feats[-1]]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


def _bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


@torch.no_grad()
def generate_anchors_for_grid_cell(
    feats: Tuple[Tensor, ...],
    fpn_strides: Tuple[int, ...],
    grid_cell_size: float = _GRID_CELL_SCALE,
    grid_cell_offset: float = _GRID_CELL_OFFSET,
    dtype: torch.dtype = torch.float,
):
    """ATSS-style grid-cell anchors, one square box per cell.

    Returns ``(anchors_xyxy, anchor_points_xy, num_anchors_list, stride_tensor)``
    with the levels concatenated in ``fpn_strides`` order.
    """
    assert len(feats) == len(fpn_strides)
    device = feats[0].device
    anchors, anchor_points, num_anchors_list, stride_tensor = [], [], [], []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

        anchor = torch.stack(
            [
                shift_x - cell_half_size,
                shift_y - cell_half_size,
                shift_x + cell_half_size,
                shift_y + cell_half_size,
            ],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype))

    return (
        torch.cat(anchors).to(device),
        torch.cat(anchor_points).to(device),
        num_anchors_list,
        torch.cat(stride_tensor).to(device),
    )


def batch_distance2bbox(points: Tensor, distance: Tensor) -> Tensor:
    """Turn ltrb distances around ``points`` into xyxy boxes."""
    lt, rb = torch.split(distance, 2, dim=-1)
    return torch.cat([points - lt, rb + points], dim=-1)


class ESEAttn(nn.Module):
    """Efficient Squeeze-and-Excitation attention used by both head stems."""

    def __init__(self, feat_channels: int, activation_type: Type[nn.Module]) -> None:
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.conv = ConvBNAct(
            feat_channels, feat_channels, 1, padding=0, stride=1,
            activation_type=activation_type, bias=False,
        )
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat: Tensor, avg_feat: Tensor) -> Tensor:
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    """Efficient Task-aligned head: per-class logits plus four DFL distributions.

    Levels are consumed in stride order 32, 16, 8 (the neck's deepest output
    first), which fixes the anchor ordering the loss and the decoder rely on.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        activation: Type[nn.Module] = nn.SiLU,
        fpn_strides: Tuple[int, int, int] = _FPN_STRIDES,
        grid_cell_scale: float = _GRID_CELL_SCALE,
        grid_cell_offset: float = _GRID_CELL_OFFSET,
        reg_max: int = _REG_MAX,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        in_channels = _scale_channels(in_channels, width_mult)
        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.fpn_strides = tuple(fpn_strides)
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        # Set by the export wrapper; when True ``forward`` returns only the
        # decoded (boxes, scores) pair so the traced graph stays minimal.
        self.export = False

        self.stem_cls = nn.ModuleList(
            [ESEAttn(c, activation_type=activation) for c in self.in_channels]
        )
        self.stem_reg = nn.ModuleList(
            [ESEAttn(c, activation_type=activation) for c in self.in_channels]
        )
        self.pred_cls = nn.ModuleList(
            [nn.Conv2d(c, self.num_classes, 3, padding=1) for c in self.in_channels]
        )
        self.pred_reg = nn.ModuleList(
            [nn.Conv2d(c, 4 * (self.reg_max + 1), 3, padding=1) for c in self.in_channels]
        )

        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape(
            [1, self.reg_max + 1, 1, 1]
        )
        self.register_buffer("proj_conv", proj, persistent=False)
        self._init_weights()

    @torch.jit.ignore
    def _init_weights(self) -> None:
        bias_cls = _bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            nn.init.constant_(cls_.weight, 0.0)
            nn.init.constant_(cls_.bias, bias_cls)
            nn.init.constant_(reg_.weight, 0.0)
            nn.init.constant_(reg_.bias, 1.0)

    @torch.jit.ignore
    def replace_num_classes(self, num_classes: int) -> None:
        """Rebuild only the class-prediction convolutions for a new class count."""
        bias_cls = _bias_init_with_prob(0.01)
        device = self.pred_cls[0].weight.device
        dtype = self.pred_cls[0].weight.dtype
        self.num_classes = num_classes
        pred_cls = nn.ModuleList()
        for in_c in self.in_channels:
            layer = nn.Conv2d(in_c, num_classes, 3, padding=1, device=device, dtype=dtype)
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, bias_cls)
            pred_cls.append(layer)
        self.pred_cls = pred_cls

    def _generate_anchors(self, feats):
        """Per-cell centre points (offset 0.5) and their strides, for decoding."""
        dtype, device = feats[0].dtype, feats[0].device
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(self.fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, dtype=torch.float32, device=device) + self.grid_cell_offset
            shift_y = torch.arange(end=h, dtype=torch.float32, device=device) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_points.append(torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype).reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def forward_train(self, feats):
        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )
        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # No sigmoid here: the loss wants logits for numerical stability.
            cls_score_list.append(torch.permute(cls_logit.flatten(2), [0, 2, 1]))
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))
        return (
            torch.cat(cls_score_list, dim=1),
            torch.cat(reg_distri_list, dim=1),
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        )

    def forward_eval(self, feats):
        cls_score_list, reg_distri_list, reg_dist_reduced_list = [], [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            hw = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(
                reg_distri.reshape([-1, 4, self.reg_max + 1, hw]), [0, 2, 3, 1]
            )
            # Softmax-expectation over the DFL bins. Written as a multiply and
            # sum rather than a 1x1 conv because OpenVINO cannot handle the
            # conv form (upstream note).
            reg_dist_reduced = F.softmax(reg_dist_reduced, dim=1) * self.proj_conv
            reg_dist_reduced = reg_dist_reduced.sum(dim=1, keepdim=False)

            cls_score_list.append(cls_logit.reshape([b, self.num_classes, hw]))
            reg_dist_reduced_list.append(reg_dist_reduced)

        cls_score_list = torch.permute(torch.cat(cls_score_list, dim=-1), [0, 2, 1])
        reg_distri_list = torch.cat(reg_distri_list, dim=1)
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)

        anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list) * stride_tensor
        decoded_predictions = pred_bboxes, pred_scores

        if self.export or torch.jit.is_tracing():
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, _ = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )
        raw_predictions = (
            cls_score_list,
            reg_distri_list,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        )
        return decoded_predictions, raw_predictions

    def forward(self, feats):
        return self.forward_train(feats) if self.training else self.forward_eval(feats)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class LibrePPYOLOEModel(nn.Module):
    """PP-YOLOE detector (sizes s, m, l, x)."""

    def __init__(
        self,
        size: str = "s",
        nb_classes: int = 80,
        in_channels: int = 3,
        reg_max: int = _REG_MAX,
    ) -> None:
        super().__init__()
        if size not in PPYOLOE_CONFIGS:
            raise ValueError(
                f"Unknown PP-YOLOE size '{size}'. Expected one of "
                f"{sorted(PPYOLOE_CONFIGS)}."
            )
        cfg = PPYOLOE_CONFIGS[size]
        depth_mult, width_mult = cfg["depth_mult"], cfg["width_mult"]

        self.size = size
        self.nc = nb_classes
        self.reg_max = reg_max

        self.backbone = CSPResNetBackbone(
            width_mult=width_mult, depth_mult=depth_mult, in_channels=in_channels
        )
        self.neck = PPYoloECSPPAN(width_mult=width_mult, depth_mult=depth_mult)
        self.head = PPYOLOEHead(
            num_classes=nb_classes,
            in_channels=_NECK_OUT_CHANNELS,
            width_mult=width_mult,
            reg_max=reg_max,
        )

    def replace_num_classes(self, num_classes: int) -> None:
        self.head.replace_num_classes(num_classes)
        self.nc = num_classes

    def fuse_reparam(self):
        """Collapse every RepVGG block into its deployment 3x3 form."""
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()
        return self

    def forward(self, x: Tensor):
        return self.head(self.neck(self.backbone(x)))
