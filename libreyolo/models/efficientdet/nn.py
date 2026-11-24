"""Native EfficientDet D0-D4 architecture.

This module follows ``rwightman/efficientdet-pytorch`` 0.4.1 (Apache-2.0),
including its TensorFlow-compatible EfficientNet backbone, learned fast-attention
BiFPN, and shared separable class/box heads. Attribute names intentionally match
the upstream graph so official state dictionaries load with ``strict=True``.

The EfficientNet blocks use TensorFlow SAME padding and BN epsilon 1e-3. BiFPN
and head BatchNorm additionally use momentum 0.01, matching ``effdet``'s
``tf_efficientdet_d*`` configuration. See this family's NOTICE for provenance.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SCALE_CONFIGS

BN_EPS = 1e-3
BN_MOMENTUM = 0.01
MIN_LEVEL = 3
MAX_LEVEL = 7
NUM_LEVELS = MAX_LEVEL - MIN_LEVEL + 1
NUM_SCALES = 3
ASPECT_RATIOS = ((1.0, 1.0), (1.4, 0.7), (0.7, 1.4))
ANCHOR_SCALE = 4.0


def make_divisible(
    value: float,
    divisor: int = 8,
    min_value: int | None = None,
    round_limit: float = 0.9,
) -> int:
    """Round a channel count using timm/EfficientNet semantics."""
    min_value = min_value or divisor
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if rounded < round_limit * value:
        rounded += divisor
    return rounded


def round_channels(channels: int, multiplier: float) -> int:
    return make_divisible(channels * multiplier)


def _same_padding(size: int, kernel: int, stride: int, dilation: int = 1) -> int:
    return max(
        (math.ceil(size / stride) - 1) * stride
        + (kernel - 1) * dilation
        + 1
        - size,
        0,
    )


def _pad_same(
    x: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int] = (1, 1),
    value: float = 0.0,
) -> torch.Tensor:
    height, width = x.shape[-2:]
    if torch.jit.is_tracing():
        # Export graphs have a fixed H/W. Bake the pads as constants: traced
        # shape arithmetic is folded to wrong pads by onnxsim when the batch
        # axis is dynamic.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            height, width = int(height), int(width)
    pad_h = _same_padding(height, kernel_size[0], stride[0], dilation[0])
    pad_w = _same_padding(width, kernel_size[1], stride[1], dilation[1])
    if pad_h or pad_w:
        x = F.pad(
            x,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            value=value,
        )
    return x


class Conv2dSame(nn.Conv2d):
    """Convolution with TensorFlow-style dynamic SAME padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_same(x, self.weight.shape[-2:], self.stride, self.dilation)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )


def create_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = False,
) -> nn.Conv2d:
    """Create the SAME-padded convolution used by all shipped TF variants."""
    static = stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0
    if static:
        padding = dilation * (kernel_size - 1) // 2
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
    return Conv2dSame(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        bias,
    )


def _backbone_bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=BN_EPS)


def _fpn_bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=BN_EPS, momentum=BN_MOMENTUM)


class SqueezeExcite(nn.Module):
    """EfficientNet squeeze/excitation with timm-compatible names."""

    def __init__(self, channels: int, reduced_channels: int) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, reduced_channels, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        return x * self.gate(scale)


class DepthwiseSeparableConv(nn.Module):
    """EfficientNet's non-expanded first stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        se_ratio: float,
    ) -> None:
        super().__init__()
        self.has_skip = stride == 1 and in_channels == out_channels
        self.conv_dw = create_conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.bn1 = _backbone_bn(in_channels)
        reduced = max(1, round(in_channels * se_ratio))
        self.se = SqueezeExcite(in_channels, reduced)
        self.conv_pw = create_conv2d(in_channels, out_channels, 1)
        self.bn2 = _backbone_bn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = F.silu(self.bn1(self.conv_dw(x)), inplace=True)
        x = self.se(x)
        x = self.bn2(self.conv_pw(x))
        if self.has_skip:
            x = x + shortcut
        return x


class InvertedResidual(nn.Module):
    """EfficientNet MBConv block with squeeze/excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_ratio: float,
        se_ratio: float,
    ) -> None:
        super().__init__()
        mid_channels = make_divisible(in_channels * expansion_ratio)
        self.has_skip = in_channels == out_channels and stride == 1
        self.conv_pw = create_conv2d(in_channels, mid_channels, 1)
        self.bn1 = _backbone_bn(mid_channels)
        self.conv_dw = create_conv2d(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
        )
        self.bn2 = _backbone_bn(mid_channels)
        reduced = max(1, round(mid_channels * (se_ratio / expansion_ratio)))
        self.se = SqueezeExcite(mid_channels, reduced)
        self.conv_pwl = create_conv2d(mid_channels, out_channels, 1)
        self.bn3 = _backbone_bn(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = F.silu(self.bn1(self.conv_pw(x)), inplace=True)
        x = F.silu(self.bn2(self.conv_dw(x)), inplace=True)
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


_BACKBONE_STAGES = (
    # type, repeats, kernel, stride, expansion, output channels, SE ratio
    ("ds", 1, 3, 1, 1.0, 16, 0.25),
    ("ir", 2, 3, 2, 6.0, 24, 0.25),
    ("ir", 2, 5, 2, 6.0, 40, 0.25),
    ("ir", 3, 3, 2, 6.0, 80, 0.25),
    ("ir", 3, 5, 1, 6.0, 112, 0.25),
    ("ir", 4, 5, 2, 6.0, 192, 0.25),
    ("ir", 1, 3, 1, 6.0, 320, 0.25),
)


class EfficientNetBackbone(nn.Module):
    """Feature-only ``tf_efficientnet_b0`` through ``b4`` backbone."""

    def __init__(self, size: str) -> None:
        super().__init__()
        cfg = SCALE_CONFIGS[size]
        channel_multiplier = cfg.channel_multiplier
        depth_multiplier = cfg.depth_multiplier
        stem_channels = round_channels(32, channel_multiplier)
        self.conv_stem = create_conv2d(3, stem_channels, 3, stride=2)
        self.bn1 = _backbone_bn(stem_channels)

        stages: List[nn.Module] = []
        in_channels = stem_channels
        for block_type, repeats, kernel, stride, expansion, base_out, se_ratio in _BACKBONE_STAGES:
            out_channels = round_channels(base_out, channel_multiplier)
            blocks: List[nn.Module] = []
            for block_index in range(int(math.ceil(repeats * depth_multiplier))):
                block_stride = stride if block_index == 0 else 1
                if block_type == "ds":
                    block = DepthwiseSeparableConv(
                        in_channels,
                        out_channels,
                        kernel,
                        block_stride,
                        se_ratio,
                    )
                else:
                    block = InvertedResidual(
                        in_channels,
                        out_channels,
                        kernel,
                        block_stride,
                        expansion,
                        se_ratio,
                    )
                blocks.append(block)
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*stages)
        self.feature_channels = (
            round_channels(40, channel_multiplier),
            round_channels(112, channel_multiplier),
            round_channels(320, channel_multiplier),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = F.silu(self.bn1(self.conv_stem(x)), inplace=True)
        outputs: List[torch.Tensor] = []
        for stage_index, stage in enumerate(self.blocks):
            x = stage(x)
            if stage_index in (2, 4, 6):
                outputs.append(x)
        return outputs


def get_feat_sizes(image_size: Tuple[int, int], max_level: int = MAX_LEVEL):
    sizes = [image_size]
    current = image_size
    for _ in range(max_level):
        current = ((current[0] - 1) // 2 + 1, (current[1] - 1) // 2 + 1)
        sizes.append(current)
    return sizes


class ConvBnAct2d(nn.Module):
    """Convolution, optional BatchNorm, and optional SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = False,
        norm_layer: bool = True,
        act_layer: bool = True,
    ) -> None:
        super().__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.bn = _fpn_bn(out_channels) if norm_layer else None
        self.act = nn.SiLU(inplace=True) if act_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """Depthwise plus pointwise convolution used throughout BiFPN and heads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
        norm_layer: bool = True,
        act_layer: bool = True,
    ) -> None:
        super().__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.conv_pw = create_conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = _fpn_bn(out_channels) if norm_layer else None
        self.act = nn.SiLU(inplace=True) if act_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MaxPool2dSame(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool2d(x, self.kernel_size, self.stride)


class Interpolate2d(nn.Module):
    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.size, mode="nearest")


class ResampleFeatureMap(nn.Sequential):
    """Project and resize one pyramid feature to a requested node shape."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: Tuple[int, int],
        output_size: Tuple[int, int],
        apply_bn: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            self.add_module(
                "conv",
                ConvBnAct2d(
                    in_channels,
                    out_channels,
                    1,
                    # The TF checkpoints retain a redundant bias immediately
                    # before BatchNorm (``redundant_bias=True`` upstream).
                    bias=True,
                    norm_layer=apply_bn,
                    act_layer=False,
                ),
            )
        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            stride_h = int((input_size[0] - 1) // output_size[0] + 1)
            stride_w = int((input_size[1] - 1) // output_size[1] + 1)
            if stride_h != stride_w:
                raise ValueError("EfficientDet only supports square pyramid strides.")
            self.add_module("downsample", MaxPool2dSame(stride_h + 1, stride_h))
        elif input_size[0] < output_size[0] or input_size[1] < output_size[1]:
            self.add_module("upsample", Interpolate2d(output_size))


def _bifpn_nodes() -> List[Dict[str, object]]:
    node_ids = {level: [level - MIN_LEVEL] for level in range(MIN_LEVEL, MAX_LEVEL + 1)}
    next_id = NUM_LEVELS
    nodes: List[Dict[str, object]] = []
    for level in range(MAX_LEVEL - 1, MIN_LEVEL - 1, -1):
        nodes.append(
            {
                "feat_level": level,
                "inputs_offsets": [node_ids[level][-1], node_ids[level + 1][-1]],
            }
        )
        node_ids[level].append(next_id)
        next_id += 1
    for level in range(MIN_LEVEL + 1, MAX_LEVEL + 1):
        nodes.append(
            {
                "feat_level": level,
                "inputs_offsets": node_ids[level] + [node_ids[level - 1][-1]],
            }
        )
        node_ids[level].append(next_id)
        next_id += 1
    return nodes


class FpnCombine(nn.Module):
    def __init__(
        self,
        feature_info: List[Dict[str, object]],
        fpn_channels: int,
        inputs_offsets: Sequence[int],
        output_size: Tuple[int, int],
        apply_resample_bn: bool,
    ) -> None:
        super().__init__()
        self.inputs_offsets = tuple(int(offset) for offset in inputs_offsets)
        self.resample = nn.ModuleDict()
        for offset in self.inputs_offsets:
            info = feature_info[offset]
            self.resample[str(offset)] = ResampleFeatureMap(
                int(info["num_chs"]),
                fpn_channels,
                info["size"],
                output_size,
                apply_bn=apply_resample_bn,
            )
        self.edge_weights = nn.Parameter(torch.ones(len(self.inputs_offsets)))

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        nodes: List[torch.Tensor] = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            nodes.append(resample(x[offset]))
        weights = F.relu(self.edge_weights.to(dtype=x[0].dtype))
        weights_sum = torch.sum(weights)
        stacked = torch.stack(
            [nodes[index] * weights[index] / (weights_sum + 0.0001) for index in range(len(nodes))],
            dim=-1,
        )
        return torch.sum(stacked, dim=-1)


class Fnode(nn.Module):
    def __init__(self, combine: nn.Module, after_combine: nn.Module) -> None:
        super().__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(
        self,
        feature_info: List[Dict[str, object]],
        feature_sizes: List[Tuple[int, int]],
        fpn_channels: int,
    ) -> None:
        super().__init__()
        nodes = _bifpn_nodes()
        expanded_info = feature_info + [
            {
                "num_chs": fpn_channels,
                "size": feature_sizes[int(node["feat_level"])],
            }
            for node in nodes
        ]
        self.fnode = nn.ModuleList()
        for node in nodes:
            combine = FpnCombine(
                expanded_info,
                fpn_channels,
                node["inputs_offsets"],
                feature_sizes[int(node["feat_level"])],
                apply_resample_bn=True,
            )
            after_combine = nn.Sequential(
                OrderedDict(
                    [
                        ("act", nn.SiLU(inplace=True)),
                        (
                            "conv",
                            SeparableConv2d(
                                fpn_channels,
                                fpn_channels,
                                3,
                                bias=True,
                                norm_layer=True,
                                act_layer=False,
                            ),
                        ),
                    ]
                )
            )
            self.fnode.append(Fnode(combine, after_combine))
        self.feature_info = expanded_info[-NUM_LEVELS:]

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for node in self.fnode:
            x.append(node(x))
        return x[-NUM_LEVELS:]


class SequentialList(nn.Sequential):
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class BiFpn(nn.Module):
    def __init__(
        self,
        image_size: int,
        feature_channels: Sequence[int],
        fpn_channels: int,
        repeats: int,
    ) -> None:
        super().__init__()
        feature_sizes = get_feat_sizes((image_size, image_size), MAX_LEVEL)
        feature_info: List[Dict[str, object]] = []
        for index, channels in enumerate(feature_channels):
            feature_info.append(
                {"num_chs": channels, "size": feature_sizes[index + MIN_LEVEL]}
            )

        previous_size = feature_sizes[MIN_LEVEL]
        self.resample = nn.ModuleDict()
        in_channels = int(feature_channels[-1])
        for level_index in range(NUM_LEVELS):
            feature_size = feature_sizes[level_index + MIN_LEVEL]
            if level_index >= len(feature_info):
                self.resample[str(level_index)] = ResampleFeatureMap(
                    in_channels,
                    fpn_channels,
                    previous_size,
                    feature_size,
                    apply_bn=True,
                )
                in_channels = fpn_channels
                feature_info.append({"num_chs": in_channels, "size": feature_size})
            previous_size = feature_size

        self.cell = SequentialList()
        for repeat in range(repeats):
            layer = BiFpnLayer(feature_info, feature_sizes, fpn_channels)
            self.cell.add_module(str(repeat), layer)
            feature_info = layer.feature_info

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x = list(x)
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        return self.cell(x)


class HeadNet(nn.Module):
    """Shared class or box prediction tower with per-level BatchNorm."""

    def __init__(
        self,
        fpn_channels: int,
        repeats: int,
        num_outputs: int,
        num_levels: int = NUM_LEVELS,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.bn_level_first = False
        self.conv_rep = nn.ModuleList(
            [
                SeparableConv2d(
                    fpn_channels,
                    fpn_channels,
                    3,
                    bias=True,
                    norm_layer=False,
                    act_layer=False,
                )
                for _ in range(repeats)
            ]
        )
        self.bn_rep = nn.ModuleList()
        for _ in range(repeats):
            self.bn_rep.append(
                nn.ModuleList(
                    [
                        nn.Sequential(OrderedDict([("bn", _fpn_bn(fpn_channels))]))
                        for _ in range(num_levels)
                    ]
                )
            )
        self.act = nn.SiLU(inplace=True)
        anchors_per_location = len(ASPECT_RATIOS) * NUM_SCALES
        self.predict = SeparableConv2d(
            fpn_channels,
            num_outputs * anchors_per_location,
            3,
            bias=True,
            norm_layer=False,
            act_layer=False,
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for level in range(self.num_levels):
            level_tensor = x[level]
            for conv, bn_levels in zip(self.conv_rep, self.bn_rep):
                level_tensor = conv(level_tensor)
                level_tensor = bn_levels[level](level_tensor)
                level_tensor = self.act(level_tensor)
            outputs.append(self.predict(level_tensor))
        return outputs


class LibreEfficientDetModel(nn.Module):
    """Strict state-dict-compatible EfficientDet D0-D4 graph."""

    def __init__(self, size: str, num_classes: int = 90) -> None:
        super().__init__()
        if size not in SCALE_CONFIGS:
            raise ValueError(f"Unknown EfficientDet size {size!r}.")
        cfg = SCALE_CONFIGS[size]
        self.size = size
        self.num_classes = num_classes
        self.image_size = cfg.image_size
        self.backbone = EfficientNetBackbone(size)
        self.fpn = BiFpn(
            cfg.image_size,
            self.backbone.feature_channels,
            cfg.fpn_channels,
            cfg.fpn_repeats,
        )
        self.class_net = HeadNet(
            cfg.fpn_channels,
            cfg.head_repeats,
            num_outputs=num_classes,
        )
        self.box_net = HeadNet(
            cfg.fpn_channels,
            cfg.head_repeats,
            num_outputs=4,
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        features = self.fpn(features)
        return self.class_net(features), self.box_net(features)


class EfficientDetExportWrapper(nn.Module):
    """Bake fixed anchors and top-candidate decode into one export output."""

    def __init__(
        self,
        model: nn.Module,
        input_size: int,
        *,
        max_candidates: int = 5000,
        sparse_coco: bool = True,
    ) -> None:
        super().__init__()
        from ...postprocess.efficientdet import generate_anchors
        from ...utils.coco import COCO91_TO_COCO80

        self.model = model
        self.input_size = int(input_size)
        self.num_classes = int(getattr(model, "num_classes", 90))
        self.max_candidates = int(max_candidates)
        self.sparse_coco = bool(sparse_coco)
        self.register_buffer("anchors", generate_anchors(self.input_size))
        class_map = [
            COCO91_TO_COCO80.get(index + 1, -1) for index in range(self.num_classes)
        ]
        self.register_buffer("class_map", torch.tensor(class_map, dtype=torch.long))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        class_outputs, box_outputs = self.model(images)
        batch = class_outputs[0].shape[0]
        logits = torch.cat(
            [
                level.permute(0, 2, 3, 1).reshape(batch, -1, self.num_classes)
                for level in class_outputs
            ],
            dim=1,
        )
        regression = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1, 4) for level in box_outputs],
            dim=1,
        )
        flat_logits = logits.reshape(batch, -1)
        top_logits, flat_indices = torch.topk(flat_logits, self.max_candidates, dim=1)
        anchor_indices = flat_indices // self.num_classes
        classes = flat_indices % self.num_classes
        regression = torch.gather(
            regression,
            1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
        )
        anchors = torch.gather(
            self.anchors.unsqueeze(0).expand(batch, -1, -1),
            1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        anchor_y = (anchors[..., 0] + anchors[..., 2]) * 0.5
        anchor_x = (anchors[..., 1] + anchors[..., 3]) * 0.5
        anchor_h = anchors[..., 2] - anchors[..., 0]
        anchor_w = anchors[..., 3] - anchors[..., 1]
        ty, tx, th, tw = regression.unbind(dim=-1)
        center_y = ty * anchor_h + anchor_y
        center_x = tx * anchor_w + anchor_x
        height = torch.exp(th) * anchor_h
        width = torch.exp(tw) * anchor_w
        boxes = torch.stack(
            (
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            ),
            dim=-1,
        )
        if self.sparse_coco:
            classes = self.class_map[classes]
        return torch.cat(
            (
                boxes,
                top_logits.sigmoid().unsqueeze(-1),
                classes.to(boxes.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )


__all__ = [
    "ANCHOR_SCALE",
    "ASPECT_RATIOS",
    "EfficientDetExportWrapper",
    "LibreEfficientDetModel",
    "MAX_LEVEL",
    "MIN_LEVEL",
    "NUM_LEVELS",
    "NUM_SCALES",
    "get_feat_sizes",
]
