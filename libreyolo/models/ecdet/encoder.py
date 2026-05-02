"""ECDET HybridEncoder (RT-DETR-style CSP-PAN with sum fusion + CSPLayer2).

Differs from D-FINE's HybridEncoder in three ways:
  * no `input_proj` (channel-match happens in the ViTAdapter projector)
  * `csp_type='csp2'` -> `CSPLayer2` (RepC3-equivalent), single conv splitting
    the channels into two halves and adding instead of concatenating
  * `fuse_op='sum'` for FPN/PAN merges, halving the fusion-block input dim
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dfine.encoder import (
    ConvNormLayer_fuse,
    SCDown,
    TransformerEncoder,
    TransformerEncoderLayer,
    VGGBlock,
    get_activation,
)


class CSPLayer2(nn.Module):
    """RepC3-equivalent: one input conv that splits into two halves; the
    second half goes through `num_blocks` bottlenecks; final add + (optional)
    project to out_channels."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(
            in_channels, hidden * 2, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden, hidden, act=get_activation(act))
                for _ in range(num_blocks)
            ]
        )
        self.conv3 = (
            ConvNormLayer_fuse(hidden, out_channels, 1, 1, bias=bias, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))


class RepNCSPELAN4(nn.Module):
    """ELAN block with switchable CSP layer (csp / csp2)."""

    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu", csp_type="csp2"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)

        if csp_type == "csp2":
            CSP = CSPLayer2
        else:
            from ..dfine.encoder import CSPLayer as CSP  # default csp

        self.cv2 = nn.Sequential(
            CSP(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv3 = nn.Sequential(
            CSP(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, 1))


class HybridEncoder(nn.Module):
    """ECDET's encoder. Same skeleton as D-FINE's HybridEncoder, but:
    * no input_proj (backbone projector matches channels)
    * `csp_type` and `fuse_op` knobs threaded through.
    """

    def __init__(
        self,
        in_channels=(192, 192, 192),
        feat_strides=(8, 16, 32),
        hidden_dim=192,
        nhead=8,
        dim_feedforward=512,
        dropout=0.0,
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        csp_type="csp2",
        fuse_op="sum",
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.feat_strides = list(feat_strides)
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = list(use_encoder_idx)
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim] * len(self.in_channels)
        self.out_strides = self.feat_strides
        self.fuse_op = fuse_op

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=act,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(self.use_encoder_idx))
            ]
        )

        input_dim = hidden_dim if fuse_op == "sum" else hidden_dim * 2

        c1, c2, c3 = input_dim, hidden_dim, hidden_dim * 2
        c4 = round(expansion * hidden_dim // 2)
        n = round(3 * depth_mult)

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))
        Fuse_Block = RepNCSPELAN4(
            c1=c1, c2=c2, c3=c3, c4=c4, n=n, act=act, csp_type=csp_type
        )

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(self, proj_feats):
        assert len(proj_feats) == len(self.in_channels)

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(
                        src_flatten.device
                    )
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, h, w)
                    .contiguous()
                )

        # top-down FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            up = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fused = (
                (up + feat_low)
                if self.fuse_op == "sum"
                else torch.concat([up, feat_low], dim=1)
            )
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](fused)
            inner_outs.insert(0, inner_out)

        # bottom-up PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            ds = self.downsample_convs[idx](feat_low)
            fused = (
                (ds + feat_high)
                if self.fuse_op == "sum"
                else torch.concat([ds, feat_high], dim=1)
            )
            outs.append(self.pan_blocks[idx](fused))

        return outs
