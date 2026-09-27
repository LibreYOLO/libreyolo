"""TinyFormer PBM neck: 4-scale hybrid encoder with 3-way bi-fusion.

Ported from the official TinyFormer release (mmpmmpmmpjosh/TinyFormer,
Apache-2.0). The stride-4 level enters the FPN through a downsample shortcut
(the Parallel Bi-fusion Module); the encoder still returns 3 levels at strides
(8, 16, 32), selected by ``out_indices``. Shared conv/transformer blocks come
from the vendored DEIMv2 engine.
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deimv2.engine.deim.hybrid_encoder import (
    ConvNormLayer_fuse,
    CSPLayer2,
    TransformerEncoder,
    TransformerEncoderLayer,
    VGGBlock,
)


class RepNCSPMSP5(nn.Module):
    """Multi-scale CSP block used by the PBM FPN/PAN paths."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer2(self.c, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv3 = nn.Sequential(
            CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv4 = ConvNormLayer_fuse(self.c + 2 * c4, c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        out_cv2 = self.cv2(y[0])
        out_cv3 = self.cv3(out_cv2)
        fused = torch.cat([out_cv2, out_cv3, y[1]], 1)
        return self.cv4(fused)


class HybridEncoder_4Scale(nn.Module):
    """4-input hybrid encoder: AIFI on the stride-32 level + PBM FPN/PAN."""

    def __init__(
        self,
        in_channels=(256, 256, 256, 256),
        feat_strides=(4, 8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(3,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        fuse_op="sum",
        out_indices=(1, 2, 3),
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.fuse_op = fuse_op
        self.out_indices = list(out_indices)

        self.out_channels = [hidden_dim for _ in self.out_indices]
        self.out_strides = [self.feat_strides[i] for i in self.out_indices]

        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            if in_channel != hidden_dim:
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
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(self.use_encoder_idx))
            ]
        )

        num_blocks = int(3 * depth_mult)

        lateral_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU() if act == "silu" else nn.Identity(),
        )
        downsample_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        fpn_block = RepNCSPMSP5(
            c1=hidden_dim * 2,
            c2=hidden_dim,
            c3=hidden_dim * 2,
            c4=round(expansion * hidden_dim // 2),
            n=num_blocks,
            act=act,
        )
        pan_block = RepNCSPMSP5(
            c1=hidden_dim if self.fuse_op == "sum" else hidden_dim * 2,
            c2=hidden_dim,
            c3=hidden_dim * 2,
            c4=round(expansion * hidden_dim // 2),
            n=num_blocks,
            act=act,
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        self.fpn_downsample_convs = nn.ModuleList()
        for _ in range(len(self.in_channels) - 2):
            self.lateral_convs.append(copy.deepcopy(lateral_conv))
            self.fpn_blocks.append(copy.deepcopy(fpn_block))
            self.fpn_downsample_convs.append(copy.deepcopy(downsample_conv))

        self.pan_blocks = nn.ModuleList()
        self.pan_downsample_convs = nn.ModuleList()
        for _ in range(len(self.in_channels) - 2):
            self.pan_blocks.append(copy.deepcopy(pan_block))
            self.pan_downsample_convs.append(copy.deepcopy(downsample_conv))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                pos_embed = self.build_2d_sincos_position_embedding(
                    w, h, self.hidden_dim, self.pe_temperature
                ).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, h, w)
                    .contiguous()
                )

        # FPN with 3-way fusion: upsampled-high + current, concat downsampled-low.
        inner_outs = [None] * len(self.in_channels)
        inner_outs[-1] = proj_feats[-1]
        current_high_feat = proj_feats[-1]
        for idx in range(len(self.in_channels) - 2, 0, -1):
            feat_current = proj_feats[idx]
            feat_low = proj_feats[idx - 1]
            block_idx = (len(self.in_channels) - 2) - idx

            feat_high_lateral = self.lateral_convs[block_idx](current_high_feat)
            feat_high_up = F.interpolate(
                feat_high_lateral, scale_factor=2.0, mode="nearest"
            )
            feat_low_down = self.fpn_downsample_convs[block_idx](feat_low)

            base_feat = feat_high_up + feat_current
            fused_feat = torch.cat([base_feat, feat_low_down], dim=1)
            out = self.fpn_blocks[block_idx](fused_feat)
            inner_outs[idx] = out
            current_high_feat = out

        inner_outs[0] = proj_feats[0]

        # PAN over the 3 finest fused levels (strides 8/16/32).
        outs = [inner_outs[1]]
        for idx in range(len(self.in_channels) - 2):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 2]
            downsample_feat = self.pan_downsample_convs[idx](feat_low)
            if self.fuse_op == "sum":
                fused_feat = downsample_feat + feat_height
            else:
                fused_feat = torch.cat([downsample_feat, feat_height], dim=1)
            outs.append(self.pan_blocks[idx](fused_feat))

        return [outs[i - 1] for i in self.out_indices]
