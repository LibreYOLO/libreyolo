"""EfficientNetV2 (base ``b0``-``b3``) — native LibreYOLO implementation.

The architecture and the per-variant block definitions are vendored from timm's
EfficientNet implementation (``timm/models/efficientnet.py`` +
``_efficientnet_blocks.py`` + ``_efficientnet_builder.py``, Apache-2.0). The
EfficientNetV2-*base* family (``tf_efficientnetv2_b{0,1,2,3}``) shares one
architecture string scaled by a per-variant (channel, depth) multiplier pair and
uses three block types::

    ConvBnAct ('cn'), EdgeResidual ('er' = FusedMBConv), InvertedResidual
    ('ir' = MBConv) with Squeeze-Excite.

Two details set EfficientNetV2 apart from the MobileNetV4-conv family that this
port extends and are essential for bit-exact parity:

* **SiLU** activation everywhere (``F.silu``), not ReLU. We use the plain
  functional SiLU, never timm's memory-efficient autograd swish, so ONNX export
  stays clean while eval math is identical.
* **TensorFlow "SAME" padding** (the ``tf_`` checkpoints): stride-2 convolutions
  pad asymmetrically at runtime; stride-1 convolutions fall back to static
  symmetric padding. BatchNorm uses the TF epsilon ``1e-3``.

Module and attribute names mirror timm exactly so a timm pretrained
``state_dict`` loads into this module with ``strict=True`` and produces
bit-identical outputs (enforced by the parity test). See the family NOTICE for
attribution.

Upstream paper: "EfficientNetV2: Smaller Models and Faster Training"
(https://arxiv.org/abs/2104.00298).
"""

from __future__ import annotations

import math
import re
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Architecture definition — copied verbatim from timm ``_gen_efficientnetv2_base``.
# All four "base" variants share this stage list; they differ only by the
# (channel_multiplier, depth_multiplier) pair applied during the build.
# --------------------------------------------------------------------------- #
ARCH_DEF: List[List[str]] = [
    ["cn_r1_k3_s1_e1_c16_skip"],
    ["er_r2_k3_s2_e4_c32"],
    ["er_r2_k3_s2_e4_c48"],
    ["ir_r3_k3_s2_e4_c96_se0.25"],
    ["ir_r5_k3_s1_e6_c112_se0.25"],
    ["ir_r8_k3_s2_e6_c192_se0.25"],
]

# (channel_multiplier, depth_multiplier) per timm ``tf_efficientnetv2_b*``.
VARIANTS: Dict[str, Dict[str, float]] = {
    "b0": {"channel_multiplier": 1.0, "depth_multiplier": 1.0},
    "b1": {"channel_multiplier": 1.0, "depth_multiplier": 1.1},
    "b2": {"channel_multiplier": 1.1, "depth_multiplier": 1.2},
    "b3": {"channel_multiplier": 1.2, "depth_multiplier": 1.4},
}

STEM_SIZE = 32  # base stem channels (rounded by channel_multiplier)
NUM_FEATURES_BASE = 1280  # base conv_head output channels (rounded too)
BN_EPS = 1e-3  # timm BN_EPS_TF_DEFAULT — the ``tf_`` checkpoints were trained with this


# --------------------------------------------------------------------------- #
# Channel rounding — timm ``make_divisible`` / ``round_channels``.
# The V2-base ``round_chs_fn`` uses ``round_limit=0.`` (no 10% floor bump).
# --------------------------------------------------------------------------- #
def make_divisible(v, divisor: int = 8, min_value=None, round_limit: float = 0.9) -> int:
    """timm ``make_divisible`` — round channels to a multiple of ``divisor``."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier: float = 1.0, divisor: int = 8, round_limit: float = 0.9) -> int:
    """timm ``round_channels`` — scale by ``multiplier`` then snap to ``divisor``."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, round_limit=round_limit)


def _num_groups(group_size: Optional[int], channels: int) -> int:
    """timm ``num_groups`` — 0/None -> 1 (dense conv); else channels // group_size."""
    if not group_size:
        return 1
    assert channels % group_size == 0
    return channels // group_size


# --------------------------------------------------------------------------- #
# TensorFlow "SAME" padding (vendored from timm ``layers/padding.py`` +
# ``layers/conv2d_same.py``). Stride-1 convs resolve to static symmetric
# padding; stride-2 convs pad dynamically at runtime (ONNX-export friendly:
# for a fixed input the F.pad amounts trace to constants).
# --------------------------------------------------------------------------- #
def _get_padding(kernel_size: int, stride: int = 1, dilation: int = 1) -> int:
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def _get_same_padding(x: int, kernel_size: int, stride: int, dilation: int) -> int:
    return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)


def _is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1) -> bool:
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def _pad_same(x: torch.Tensor, kernel_size: Tuple[int, int], stride: Tuple[int, int],
              dilation: Tuple[int, int] = (1, 1), value: float = 0.0) -> torch.Tensor:
    ih, iw = x.size()[-2:]
    if torch.jit.is_tracing():
        # Export graphs have a fixed H/W. Bake the pads as constants: traced
        # shape arithmetic is folded to wrong pads by onnxsim when the batch
        # axis is dynamic.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            ih, iw = int(ih), int(iw)
    pad_h = _get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w = _get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    return F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)


class Conv2dSame(nn.Conv2d):
    """nn.Conv2d wrapper applying TF-like SAME padding dynamically (``padding=0``)."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_chs, out_chs, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_same(x, self.weight.shape[-2:], self.stride, self.dilation)
        return F.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)


def create_conv2d(in_chs, out_chs, kernel_size, stride=1, dilation=1, groups=1, bias=False) -> nn.Conv2d:
    """timm ``create_conv2d`` with ``padding='same'`` (the only mode these models use)."""
    if _is_static_pad(kernel_size, stride, dilation):
        padding = _get_padding(kernel_size, stride, dilation)
        return nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, dilation, groups, bias=bias)
    return Conv2dSame(in_chs, out_chs, kernel_size, stride, dilation, groups, bias=bias)


def _bn(num_features: int) -> nn.BatchNorm2d:
    """BatchNorm with the TF epsilon used by the ``tf_efficientnetv2`` checkpoints."""
    return nn.BatchNorm2d(num_features, eps=BN_EPS)


# --------------------------------------------------------------------------- #
# Block modules. Submodule names mirror timm so state dicts map 1:1.
# Activation is SiLU throughout (functional, export-clean).
# --------------------------------------------------------------------------- #
class SqueezeExcite(nn.Module):
    """timm ``SqueezeExcite`` (EfficientNet flavour). Submodules: ``conv_reduce``, ``conv_expand``.

    Reduces from the *expanded* (mid) channels. ``rd_channels`` is computed from
    the block input via the builder's ``se_ratio /= exp_ratio`` adjustment
    (``se_from_exp=False``), rounded with Python ``round``. Gate is sigmoid.
    """

    def __init__(self, in_chs: int, rd_channels: int):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = nn.SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """timm ``ConvBnAct`` ('cn'). Submodules: ``conv``, ``bn1`` (BN + SiLU)."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, skip=False):
        super().__init__()
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride)
        self.bn1 = _bn(out_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs

    def forward(self, x):
        shortcut = x
        x = F.silu(self.bn1(self.conv(x)), inplace=True)
        if self.has_skip:
            x = x + shortcut
        return x


class EdgeResidual(nn.Module):
    """timm ``EdgeResidual`` / FusedMBConv ('er').

    Submodules: ``conv_exp``, ``bn1``, ``se`` (Identity here — no SE in the V2
    ``er`` stages), ``conv_pwl``, ``bn2``.
    """

    def __init__(self, in_chs, out_chs, exp_kernel_size, stride, exp_ratio,
                 se_ratio=0.0, pw_kernel_size=1, noskip=False):
        super().__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size, stride=stride)
        self.bn1 = _bn(mid_chs)
        self.se = _build_se(mid_chs, exp_ratio, se_ratio)
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size)
        self.bn2 = _bn(out_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

    def forward(self, x):
        shortcut = x
        x = F.silu(self.bn1(self.conv_exp(x)), inplace=True)
        x = self.se(x)
        x = self.bn2(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


class InvertedResidual(nn.Module):
    """timm ``InvertedResidual`` / MBConv ('ir') with Squeeze-Excite.

    Submodules: ``conv_pw``, ``bn1``, ``conv_dw``, ``bn2``, ``se``, ``conv_pwl``,
    ``bn3``.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size, stride, exp_ratio,
                 se_ratio=0.0, exp_kernel_size=1, pw_kernel_size=1, noskip=False):
        super().__init__()
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size)
        self.bn1 = _bn(mid_chs)
        # Depth-wise convolution (groups == channels)
        self.conv_dw = create_conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, groups=mid_chs)
        self.bn2 = _bn(mid_chs)
        # Squeeze-and-excitation
        self.se = _build_se(mid_chs, exp_ratio, se_ratio)
        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size)
        self.bn3 = _bn(out_chs)

    def forward(self, x):
        shortcut = x
        x = F.silu(self.bn1(self.conv_pw(x)), inplace=True)
        x = F.silu(self.bn2(self.conv_dw(x)), inplace=True)
        x = self.se(x)
        x = self.bn3(self.conv_pwl(x))
        if self.has_skip:
            x = x + shortcut
        return x


def _build_se(mid_chs: int, exp_ratio: float, se_ratio: float) -> nn.Module:
    """Construct the SE module for a block, or ``Identity`` when ``se_ratio == 0``.

    Mirrors timm's ``se_from_exp=False`` path: the squeeze ratio is divided by
    the expansion ratio so the reduction is computed relative to the *block
    input*, then rounded against the expanded channel count with Python ``round``.
    """
    if not se_ratio:
        return nn.Identity()
    rd_channels = round(mid_chs * (se_ratio / exp_ratio))
    return SqueezeExcite(mid_chs, rd_channels)


# --------------------------------------------------------------------------- #
# Arch-string decoding (trimmed timm ``_decode_block_str`` for cn/er/ir) +
# depth scaling (``_scale_stage_depth``, ceil truncation).
# --------------------------------------------------------------------------- #
def _parse_block_str(block_str: str):
    parts = block_str.split("_")
    block_type = parts[0]
    options: Dict[str, str] = {}
    skip = None
    for op in parts[1:]:
        if op == "noskip":
            skip = False
            continue
        if op == "skip":
            skip = True
            continue
        m = re.split(r"(\d.*)", op)
        if len(m) >= 2:
            options[m[0]] = m[1]

    repeats = int(options["r"])
    ba = {
        "block_type": block_type,
        "out_chs": int(options["c"]),
        "stride": int(options["s"]),
    }
    if block_type == "cn":
        ba.update(kernel_size=int(options["k"]), skip=skip is True)
    elif block_type == "er":
        ba.update(
            exp_kernel_size=int(options["k"]),
            pw_kernel_size=1,
            exp_ratio=float(options["e"]),
            se_ratio=float(options.get("se", 0.0)),
            noskip=skip is False,
        )
    elif block_type == "ir":
        ba.update(
            dw_kernel_size=int(options["k"]),
            exp_kernel_size=1,
            pw_kernel_size=1,
            exp_ratio=float(options["e"]),
            se_ratio=float(options.get("se", 0.0)),
            noskip=skip is False,
        )
    else:  # pragma: no cover - the V2-base def only uses cn/er/ir
        raise ValueError(f"Unsupported EfficientNetV2-base block type: {block_type!r}")
    return ba, repeats


def _scale_stage_depth(stack_args: List[Dict], repeats: List[int], depth_multiplier: float = 1.0) -> List[Dict]:
    """timm ``_scale_stage_depth`` (``depth_trunc='ceil'``)."""
    num_repeat = sum(repeats)
    num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round(r / num_repeat * num_repeat_scaled))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    sa_scaled: List[Dict] = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend(deepcopy(ba) for _ in range(rep))
    return sa_scaled


def decode_arch_def(arch_def: List[List[str]], depth_multiplier: float = 1.0) -> List[List[Dict]]:
    """timm ``decode_arch_def`` — strings -> per-stage lists of block-arg dicts."""
    stages: List[List[Dict]] = []
    for block_strings in arch_def:
        stack_args: List[Dict] = []
        repeats: List[int] = []
        for block_str in block_strings:
            ba, rep = _parse_block_str(block_str)
            stack_args.append(ba)
            repeats.append(rep)
        stages.append(_scale_stage_depth(stack_args, repeats, depth_multiplier))
    return stages


def _make_block(ba: Dict, in_chs: int, out_chs: int, stride: int) -> nn.Module:
    bt = ba["block_type"]
    if bt == "cn":
        return ConvBnAct(in_chs, out_chs, ba["kernel_size"], stride, skip=ba["skip"])
    if bt == "er":
        return EdgeResidual(
            in_chs, out_chs, ba["exp_kernel_size"], stride, ba["exp_ratio"],
            se_ratio=ba["se_ratio"], pw_kernel_size=ba["pw_kernel_size"], noskip=ba["noskip"],
        )
    return InvertedResidual(
        in_chs, out_chs, ba["dw_kernel_size"], stride, ba["exp_ratio"],
        se_ratio=ba["se_ratio"], exp_kernel_size=ba["exp_kernel_size"],
        pw_kernel_size=ba["pw_kernel_size"], noskip=ba["noskip"],
    )


# --------------------------------------------------------------------------- #
# The model.
# --------------------------------------------------------------------------- #
class EfficientNetV2(nn.Module):
    """EfficientNetV2-base backbone + classification head (timm-compatible)."""

    def __init__(self, size: str = "b0", num_classes: int = 1000):
        super().__init__()
        if size not in VARIANTS:
            raise ValueError(f"Unknown EfficientNetV2 size {size!r}; choose from {list(VARIANTS)}.")
        self.size = size
        self.num_classes = num_classes
        cfg = VARIANTS[size]
        ch_mult = cfg["channel_multiplier"]
        depth_mult = cfg["depth_multiplier"]

        def round_chs(c):
            return round_channels(c, ch_mult, round_limit=0.0)

        # Stem
        stem_chs = round_chs(STEM_SIZE)
        self.conv_stem = create_conv2d(3, stem_chs, 3, stride=2)
        self.bn1 = _bn(stem_chs)

        # Middle stages
        stages: List[nn.Module] = []
        in_chs = stem_chs
        for stage in decode_arch_def(ARCH_DEF, depth_mult):
            blocks: List[nn.Module] = []
            for block_idx, ba in enumerate(stage):
                stride = ba["stride"] if block_idx == 0 else 1  # only first block strides
                out_chs = round_chs(ba["out_chs"])
                blocks.append(_make_block(ba, in_chs, out_chs, stride))
                in_chs = out_chs
            stages.append(nn.Sequential(*blocks))
        self.blocks = nn.Sequential(*stages)

        # Head + pooling
        num_features = round_chs(NUM_FEATURES_BASE)
        self.conv_head = create_conv2d(in_chs, num_features, 1)
        self.bn2 = _bn(num_features)
        self.num_features = self.head_hidden_size = num_features
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.bn1(self.conv_stem(x)), inplace=True)
        x = self.blocks(x)
        x = F.silu(self.bn2(self.conv_head(x)), inplace=True)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if pre_logits:
            return x
        return self.classifier(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Swap the final Linear for fine-tuning to a new class count.

        Preserves the existing head's device AND dtype so half/bfloat16 flows
        survive a class-count change.
        """
        self.num_classes = num_classes
        w = self.classifier.weight
        self.classifier = nn.Linear(self.head_hidden_size, num_classes).to(device=w.device, dtype=w.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
