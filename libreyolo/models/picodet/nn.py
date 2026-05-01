"""PicoDet network: ESNet backbone + CSP-PAN neck + PicoHead.

Faithful port of Bo396543018/Picodet_Pytorch, with mmcv stripped out.
Activations match upstream exactly (HardSwish in backbone/neck/head,
HardSigmoid in the SE gate, ReLU in the SE bottleneck) so PaddlePaddle
checkpoints converted by Bo's pipeline load with no numerical drift.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class HSigmoid(nn.Module):
    """Hard sigmoid: ``clip((x + bias) / divisor, 0, max_value / divisor)``.

    Defaults match mmcv's ``HSigmoid`` (``bias=3, divisor=6, max=6``), which
    is what PicoDet's SE gate uses.
    """

    def __init__(self, bias: float = 3.0, divisor: float = 6.0, max_value: float = 6.0) -> None:
        super().__init__()
        self.bias = bias
        self.divisor = divisor
        self.upper = max_value / divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + self.bias) / self.divisor, 0.0, self.upper)


def _make_act(act: str | None) -> nn.Module:
    """Build an activation module from a short name. ``None`` -> Identity."""
    if act is None:
        return nn.Identity()
    name = act.lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("hswish", "hardswish", "h_swish"):
        return nn.Hardswish(inplace=True)
    if name in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.1, inplace=True)
    if name in ("hsigmoid", "hardsigmoid"):
        return HSigmoid()
    raise ValueError(f"Unsupported activation: {act!r}")


# ---------------------------------------------------------------------------
# Conv blocks
# ---------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + activation. Replaces mmcv's ``ConvModule``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        act: str | None = "hswish",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = _make_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise, each followed by BN + activation.

    Matches mmcv's ``DepthwiseSeparableConvModule`` layout (BN+Act after
    *both* convolutions). The simplified one-act variant used in some
    other ports is **not** weight-compatible with PaddlePaddle/Bo
    checkpoints.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        act: str | None = "hswish",
    ) -> None:
        super().__init__()
        self.depthwise_conv = ConvBNAct(
            in_channels, in_channels, kernel_size,
            stride=stride, groups=in_channels, act=act,
        )
        self.pointwise_conv = ConvBNAct(
            in_channels, out_channels, kernel_size=1, act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


# ---------------------------------------------------------------------------
# ESNet backbone (Enhanced ShuffleNet)
# ---------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round ``v`` to the nearest multiple of ``divisor`` (>= 90% of v)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class SELayer(nn.Module):
    """Squeeze-and-Excitation block. Gate: ReLU bottleneck -> HSigmoid."""

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        mid = channels // ratio
        self.conv1 = nn.Conv2d(channels, mid, 1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, channels, 1)
        self.act2 = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.act1(self.conv1(scale))
        scale = self.act2(self.conv2(scale))
        return x * scale


class ESBlock(nn.Module):
    """Stride-1 ES block: split, branch through SE, concat, channel-shuffle."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_pw = ConvBNAct(in_channels // 2, mid_channels // 2, 1, act="hswish")
        self.conv_dw = ConvBNAct(
            mid_channels // 2, mid_channels // 2, 3,
            groups=mid_channels // 2, act=None,
        )
        self.se = SELayer(mid_channels)
        self.conv_linear = ConvBNAct(mid_channels, out_channels // 2, 1, act="hswish")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x2 = self.conv_pw(x2)
        x3 = self.conv_dw(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x3 = self.se(x3)
        x3 = self.conv_linear(x3)
        out = torch.cat([x1, x3], dim=1)
        return _channel_shuffle(out, 2)


class ESBlockDS(nn.Module):
    """Stride-2 ES downsample block: dual-branch -> concat -> dw+pw."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        # Branch 1: depthwise stride-2 + linear pointwise
        self.conv_dw_1 = ConvBNAct(
            in_channels, in_channels, 3, stride=2,
            groups=in_channels, act=None,
        )
        self.conv_linear_1 = ConvBNAct(in_channels, out_channels // 2, 1, act="hswish")

        # Branch 2: pw -> dw stride-2 -> SE -> linear pw
        self.conv_pw_2 = ConvBNAct(in_channels, mid_channels // 2, 1, act="hswish")
        self.conv_dw_2 = ConvBNAct(
            mid_channels // 2, mid_channels // 2, 3, stride=2,
            groups=mid_channels // 2, act=None,
        )
        self.se = SELayer(mid_channels // 2)
        self.conv_linear_2 = ConvBNAct(mid_channels // 2, out_channels // 2, 1, act="hswish")

        # Post-concat refinement (HSwish on both, per Bo's repo)
        self.conv_dw_mv1 = ConvBNAct(
            out_channels, out_channels, 3, groups=out_channels, act="hswish",
        )
        self.conv_pw_mv1 = ConvBNAct(out_channels, out_channels, 1, act="hswish")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_linear_1(self.conv_dw_1(x))
        x2 = self.conv_pw_2(x)
        x2 = self.conv_dw_2(x2)
        x2 = self.se(x2)
        x2 = self.conv_linear_2(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv_dw_mv1(out)
        out = self.conv_pw_mv1(out)
        return out


class ESNet(nn.Module):
    """Enhanced ShuffleNet backbone, sizes ``s`` / ``m`` / ``l``.

    Returns C3, C4, C5 feature maps (block indices 2, 9, 12).
    """

    ARCH = {
        "s": {
            "scale": 0.75,
            "ratios": [0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625,
                       0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        },
        "m": {
            "scale": 1.0,
            "ratios": [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625,
                       0.625, 0.5, 0.625, 1.0, 0.625, 0.75],
        },
        "l": {
            "scale": 1.25,
            "ratios": [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625,
                       0.625, 0.5, 0.625, 1.0, 0.625, 0.75],
        },
    }
    STAGE_REPEATS = (3, 7, 3)
    OUT_INDICES = (2, 9, 12)

    def __init__(self, model_size: str = "s") -> None:
        super().__init__()
        if model_size not in self.ARCH:
            raise ValueError(f"ESNet size must be one of {list(self.ARCH)}, got {model_size!r}")
        cfg = self.ARCH[model_size]
        scale: float = cfg["scale"]  # type: ignore[assignment]
        ratios: List[float] = cfg["ratios"]  # type: ignore[assignment]

        stage_channels = [
            24,
            _make_divisible(128 * scale, 16),
            _make_divisible(256 * scale, 16),
            _make_divisible(512 * scale, 16),
        ]
        self.out_channels: Tuple[int, int, int] = (
            stage_channels[1], stage_channels[2], stage_channels[3]
        )

        self.conv1 = ConvBNAct(3, stage_channels[0], 3, stride=2, act="hswish")
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # Block list. Bo's repo uses ``setattr`` with names like ``2_1``, but a
        # plain ModuleList is functionally identical and serializes cleanly.
        self.blocks = nn.ModuleList()
        arch_idx = 0
        for stage_id, repeats in enumerate(self.STAGE_REPEATS):
            stage_in = stage_channels[stage_id]
            stage_out = stage_channels[stage_id + 1]
            for i in range(repeats):
                in_ch = stage_in if i == 0 else stage_out
                mid_ch = _make_divisible(int(stage_out * ratios[arch_idx]), 8)
                if i == 0:
                    block: nn.Module = ESBlockDS(in_ch, mid_ch, stage_out)
                else:
                    block = ESBlock(in_ch, mid_ch, stage_out)
                self.blocks.append(block)
                arch_idx += 1

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.max_pool(self.conv1(x))
        outs: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.OUT_INDICES:
                outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# CSP-PAN neck
# ---------------------------------------------------------------------------


class DarknetBottleneck(nn.Module):
    """1x1 conv -> kxk conv (depthwise-separable by default) + optional residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        expansion: float = 1.0,
        add_identity: bool = True,
        use_depthwise: bool = True,
        act: str = "hswish",
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden, 1, act=act)
        if use_depthwise:
            self.conv2: nn.Module = DepthwiseSeparableConv(hidden, out_channels, kernel_size, act=act)
        else:
            self.conv2 = ConvBNAct(hidden, out_channels, kernel_size, act=act)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross Stage Partial layer: two 1x1 convs split the input, one path
    goes through ``num_blocks`` bottlenecks, then concat + final 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        expansion: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = True,
        act: str = "hswish",
    ) -> None:
        super().__init__()
        mid = int(out_channels * expansion)
        self.main_conv = ConvBNAct(in_channels, mid, 1, act=act)
        self.short_conv = ConvBNAct(in_channels, mid, 1, act=act)
        self.final_conv = ConvBNAct(2 * mid, out_channels, 1, act=act)
        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                mid, mid, kernel_size=kernel_size, expansion=1.0,
                add_identity=add_identity, use_depthwise=use_depthwise, act=act,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.blocks(self.main_conv(x))
        x_short = self.short_conv(x)
        return self.final_conv(torch.cat([x_main, x_short], dim=1))


class CSPPAN(nn.Module):
    """CSP-Path Aggregation Network neck.

    Takes [C3, C4, C5] from the backbone, normalises them all to
    ``out_channels`` via 1x1 convs, runs a top-down + bottom-up pyramid
    with CSP blocks, and (when ``num_features=4``) appends a P6 obtained
    by stride-2 downsampling C5 plus the highest-level fused feature.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 96,
        kernel_size: int = 5,
        num_features: int = 4,
        expansion: float = 1.0,
        num_csp_blocks: int = 1,
        use_depthwise: bool = True,
        act: str = "hswish",
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_features = num_features

        self.trans = nn.ModuleList([
            ConvBNAct(c, out_channels, 1, act=act) for c in self.in_channels
        ])

        if num_features == 4:
            self.first_top_conv = DepthwiseSeparableConv(
                out_channels, out_channels, kernel_size, stride=2, act=act,
            ) if use_depthwise else ConvBNAct(
                out_channels, out_channels, kernel_size, stride=2, act=act,
            )
            self.second_top_conv = DepthwiseSeparableConv(
                out_channels, out_channels, kernel_size, stride=2, act=act,
            ) if use_depthwise else ConvBNAct(
                out_channels, out_channels, kernel_size, stride=2, act=act,
            )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.top_down_blocks = nn.ModuleList([
            CSPLayer(
                out_channels * 2, out_channels, kernel_size=kernel_size,
                expansion=expansion, num_blocks=num_csp_blocks,
                add_identity=False, use_depthwise=use_depthwise, act=act,
            )
            for _ in range(len(self.in_channels) - 1)
        ])

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsamples.append(
                DepthwiseSeparableConv(out_channels, out_channels, kernel_size, stride=2, act=act)
                if use_depthwise else
                ConvBNAct(out_channels, out_channels, kernel_size, stride=2, act=act)
            )
            self.bottom_up_blocks.append(CSPLayer(
                out_channels * 2, out_channels, kernel_size=kernel_size,
                expansion=expansion, num_blocks=num_csp_blocks,
                add_identity=False, use_depthwise=use_depthwise, act=act,
            ))

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        feats = [t(x) for t, x in zip(self.trans, inputs)]

        # Top-down
        inner_outs = [feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            up = self.upsample(inner_outs[0])
            low = feats[idx - 1]
            # Guard against odd resolutions (e.g. 416 -> 13x13 -> 26x26 OK,
            # but 256 with size_divisor!=32 can drift).
            if up.shape[-2:] != low.shape[-2:]:
                up = F.interpolate(up, size=low.shape[-2:], mode="nearest")
            inner = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([up, low], dim=1)
            )
            inner_outs.insert(0, inner)

        # Bottom-up
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            down = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](
                torch.cat([down, inner_outs[idx + 1]], dim=1)
            ))

        if self.num_features == 4:
            top = self.first_top_conv(feats[-1]) + self.second_top_conv(outs[-1])
            outs.append(top)

        return tuple(outs)


# ---------------------------------------------------------------------------
# PicoHead (anchor-free, GFL-style with shared cls/reg branch)
# ---------------------------------------------------------------------------


class Integral(nn.Module):
    """Discrete -> continuous regression via softmax expectation (DFL)."""

    def __init__(self, reg_max: int = 7) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 4 * (reg_max + 1)) -> (..., 4)
        shape = x.shape
        x = x.reshape(-1, self.reg_max + 1)
        x = F.softmax(x, dim=-1)
        if self.project.dtype != x.dtype:  # type: ignore[attr-defined]
            self.project = self.project.to(dtype=x.dtype)  # type: ignore[assignment]
        x = F.linear(x, self.project.view(1, -1)).squeeze(-1)  # type: ignore[attr-defined]
        return x.reshape(*shape[:-1], 4)


class PicoHead(nn.Module):
    """Per-level shared cls/reg head.

    When ``share_cls_reg=True``, the final 1x1 produces ``num_classes + 4*(reg_max+1)``
    channels and is split. The ``self.export`` flag flips the head into a
    decode-friendly mode for ONNX (matches landmine #13).

    Output (training/eval, ``export=False``):
        cls_scores: list of (B, num_classes, H, W)
        bbox_preds: list of (B, 4*(reg_max+1), H, W)
    """

    def __init__(
        self,
        in_channels: int = 96,
        num_classes: int = 80,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        kernel_size: int = 5,
        reg_max: int = 7,
        strides: Sequence[int] = (8, 16, 32, 64),
        share_cls_reg: bool = True,
        use_depthwise: bool = True,
        act: str = "hswish",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.reg_max = reg_max
        self.strides = tuple(strides)
        self.share_cls_reg = share_cls_reg
        self.export: bool = False  # flipped by exporter

        def _stack(ch_in: int) -> nn.ModuleList:
            mods: List[nn.Module] = []
            chn = ch_in
            for _ in range(stacked_convs):
                if use_depthwise:
                    mods.append(DepthwiseSeparableConv(chn, feat_channels, kernel_size, act=act))
                else:
                    mods.append(ConvBNAct(chn, feat_channels, kernel_size, act=act))
                chn = feat_channels
            return nn.ModuleList(mods)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.gfl_cls = nn.ModuleList()
        self.gfl_reg: nn.ModuleList | List[None] = nn.ModuleList()

        cls_out_channels = num_classes + (4 * (reg_max + 1) if share_cls_reg else 0)
        reg_out_channels = 4 * (reg_max + 1)

        for _ in self.strides:
            self.cls_convs.append(_stack(in_channels))
            if share_cls_reg:
                self.reg_convs.append(nn.ModuleList())  # placeholder; not used
            else:
                self.reg_convs.append(_stack(in_channels))

            self.gfl_cls.append(nn.Conv2d(feat_channels, cls_out_channels, 1))
            if share_cls_reg:
                self.gfl_reg.append(None)  # type: ignore[arg-type]
            else:
                assert isinstance(self.gfl_reg, nn.ModuleList)
                self.gfl_reg.append(nn.Conv2d(feat_channels, reg_out_channels, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        # Bo uses Normal(std=0.01) for all Conv2d; bias_prob=0.01 for gfl_cls.
        import math

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        bias_init = -math.log((1 - 0.01) / 0.01)  # ~ -4.595
        for gfl_cls in self.gfl_cls:
            if isinstance(gfl_cls, nn.Conv2d) and gfl_cls.bias is not None:
                nn.init.constant_(gfl_cls.bias[: self.num_classes], bias_init)

    def forward(
        self, feats: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores: List[torch.Tensor] = []
        bbox_preds: List[torch.Tensor] = []
        for level, x in enumerate(feats):
            cls_feat = x
            for conv in self.cls_convs[level]:
                cls_feat = conv(cls_feat)
            if self.share_cls_reg:
                feat = self.gfl_cls[level](cls_feat)
                cls_score, bbox_pred = torch.split(
                    feat, [self.num_classes, 4 * (self.reg_max + 1)], dim=1,
                )
            else:
                reg_feat = x
                for conv in self.reg_convs[level]:
                    reg_feat = conv(reg_feat)
                cls_score = self.gfl_cls[level](cls_feat)
                bbox_pred = self.gfl_reg[level](reg_feat)  # type: ignore[index]

            if self.export:
                # Flatten + sigmoid so backends can NMS without re-shaping.
                B = cls_score.shape[0]
                cls_score = (
                    torch.sigmoid(cls_score)
                    .reshape(B, self.num_classes, -1)
                    .permute(0, 2, 1)
                )
                bbox_pred = bbox_pred.reshape(B, 4 * (self.reg_max + 1), -1).permute(0, 2, 1)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


# Per-size architecture spec. Matches Bo's configs:
#   s/320: ESNet-s, neck=96ch, head stacked_convs=2
#   m/416: ESNet-m, neck=128ch, head stacked_convs=4
#   l/640: ESNet-l, neck=160ch, head stacked_convs=4 (inherited from m)
SIZE_SPEC = {
    "s": {"backbone": "s", "neck_ch": 96,  "head_ch": 96,  "stacked_convs": 2},
    "m": {"backbone": "m", "neck_ch": 128, "head_ch": 128, "stacked_convs": 4},
    "l": {"backbone": "l", "neck_ch": 160, "head_ch": 160, "stacked_convs": 4},
}


class LibrePicoDetModel(nn.Module):
    """Top-level PicoDet network: backbone -> neck -> head."""

    def __init__(self, size: str = "s", nb_classes: int = 80) -> None:
        super().__init__()
        if size not in SIZE_SPEC:
            raise ValueError(f"PicoDet size must be one of {list(SIZE_SPEC)}, got {size!r}")
        spec = SIZE_SPEC[size]

        self.backbone = ESNet(model_size=spec["backbone"])  # type: ignore[arg-type]
        self.neck = CSPPAN(
            in_channels=self.backbone.out_channels,
            out_channels=spec["neck_ch"],  # type: ignore[arg-type]
            kernel_size=5,
            num_features=4,
            expansion=1.0,
            num_csp_blocks=1,
            use_depthwise=True,
            act="hswish",
        )
        self.head = PicoHead(
            in_channels=spec["neck_ch"],  # type: ignore[arg-type]
            num_classes=nb_classes,
            feat_channels=spec["head_ch"],  # type: ignore[arg-type]
            stacked_convs=spec["stacked_convs"],  # type: ignore[arg-type]
            kernel_size=5,
            reg_max=7,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
            act="hswish",
        )

        self._init_backbone_neck()

    def _init_backbone_neck(self) -> None:
        for m in [self.backbone, self.neck]:
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.BatchNorm2d):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)
