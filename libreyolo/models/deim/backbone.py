"""HGNetV2 backbone used by every DEIM size.

Ported from DEIM (https://github.com/Intellindust-AI-Lab/DEIM).
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

Differences from upstream:
- ``@register()`` decorator removed (LibreYOLO wires modules with plain Python).
- Pretrained auto-download path removed. Backbone weights come from the full
  DEIM checkpoint; if a user wants to train from scratch they must supply
  their own HGNetV2 stage-1 weights. ``pretrained=True`` is a no-op.
- Distributed-rank/barrier helpers removed — LibreYOLO training is single-GPU
  from the perspective of this module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import FrozenBatchNorm2d
from .ms_deform import get_activation


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        groups=1,
        padding="",
        use_act=True,
        use_lab=False,
        act="relu",
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs, out_chs, kernel_size, stride, groups=groups, bias=False
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = get_activation(act) if self.use_act else nn.Identity()
        self.lab = (
            LearnableAffineBlock() if (self.use_act and self.use_lab) else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        groups=1,
        use_lab=False,
        act="relu",
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            act=act,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
            act=act,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False, act="relu"):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act
        )
        self.stem2a = ConvBNAct(
            mid_chs, mid_chs // 2, kernel_size=2, stride=1, use_lab=use_lab, act=act
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2, mid_chs, kernel_size=2, stride=1, use_lab=use_lab, act=act
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2, mid_chs, kernel_size=3, stride=2, use_lab=use_lab, act=act
        )
        self.stem4 = ConvBNAct(
            mid_chs, out_chs, kernel_size=1, stride=1, use_lab=use_lab, act=act
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg="ese",
        drop_path=0.0,
        act="relu",
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        act=act,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                        act=act,
                    )
                )

        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
                act=act,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_num,
        layer_num,
        downsample=True,
        light_block=False,
        kernel_size=3,
        use_lab=False,
        agg="se",
        drop_path=0.0,
        act="relu",
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
                act=act,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=i != 0,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, (list, tuple))
                    else drop_path,
                    act=act,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class HGNetv2(nn.Module):
    """HGNetV2 backbone.

    Args:
        name: One of ``"Atto"``, ``"Femto"``, ``"Pico"``, or ``"B0"``-``"B6"``.
        use_lab: Use LearnableAffineBlock after activations (N/S/M).
        return_idx: Indices of stages to return as outputs.
            DEIM uses ``[2, 3]`` for N (2-level FPN) and ``[1, 2, 3]`` for S/M/L/X.
        freeze_stem_only: Only freeze stem; ignored if ``freeze_at < 0``.
        freeze_at: Freeze stages up to and including this index. ``-1`` disables.
        freeze_norm: Convert BatchNorm2d layers to FrozenBatchNorm2d (L/X).
        pretrained: No-op. Kept for signature compatibility.
    """

    arch_configs = {
        "Atto": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 256, 1, True, True, 3, 3],
            },
        },
        "Femto": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 1, True, True, 5, 3],
            },
        },
        "Pico": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
            },
        },
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
        },
        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
        },
        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
        },
        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
        },
    }

    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=(1, 2, 3),
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=False,
        act="relu",
    ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = list(return_idx)

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
            act=act,
        )

        self.stages = nn.ModuleList()
        for k in stage_config:
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    act=act,
                )
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
