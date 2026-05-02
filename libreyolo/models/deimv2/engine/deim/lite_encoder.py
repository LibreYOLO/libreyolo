"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from libreyolo.models.deim.ms_deform import get_activation
from .hybrid_encoder import ConvNormLayer_fuse
from .hybrid_encoder import RepNCSPELAN4

__all__ = ["LiteEncoder"]


# Copy from https://github.com/meituan/YOLOv6/blob/main/yolov6/layers/common.py#L695
class GAP_Fusion(nn.Module):
    """BiFusion Block in PAN"""

    def __init__(self, in_channels, out_channels, act=None):
        super().__init__()
        self.cv = ConvNormLayer_fuse(out_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        # global average pooling
        gap = F.adaptive_avg_pool2d(x, 1)
        x = x + gap
        return self.cv(x)


# Two-scale encoder
class LiteEncoder(nn.Module):
    __share__ = [
        "eval_spatial_size",
    ]

    def __init__(
        self,
        in_channels=[512],
        feat_strides=[16],
        hidden_dim=256,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        csp_type="csp2",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection: unify the channel dimension of the input features
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                in_channel, hidden_dim, kernel_size=1, bias=False
                            ),
                        ),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )

            self.input_proj.append(proj)

        # get the small-scale feature
        down_sample = nn.Sequential(  # avg pooling
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            get_activation(act),
        )
        self.down_sample1 = copy.deepcopy(down_sample)
        self.down_sample2 = copy.deepcopy(down_sample)

        # Bi-Fusion
        self.bi_fusion = GAP_Fusion(hidden_dim, hidden_dim, act=act)

        # fuse block
        c1, c2, c3, c4, num_blocks = (
            hidden_dim,
            hidden_dim,
            hidden_dim * 2,
            round(expansion * hidden_dim // 2),
            round(3 * depth_mult),
        )
        fuse_block = RepNCSPELAN4(
            c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type
        )
        self.fpn_block = copy.deepcopy(fuse_block)
        self.pan_block = copy.deepcopy(fuse_block)

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats.append(
            self.down_sample1(proj_feats[-1])
        )  # get the small-scale feature

        # fuse the global feature and the small-scale feature
        proj_feats[-1] = self.bi_fusion(proj_feats[-1])

        outs = []
        # fpn
        fuse_feat = proj_feats[0] + F.interpolate(
            proj_feats[1], scale_factor=2.0, mode="nearest"
        )
        outs.append(self.fpn_block(fuse_feat))

        fuse_feat = proj_feats[1] + self.down_sample2(outs[-1])
        outs.append(self.pan_block(fuse_feat))

        return outs
