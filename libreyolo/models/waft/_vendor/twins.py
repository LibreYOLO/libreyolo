# Twins-SVT-Large feature encoder used by WAFT a2 (``feature_encoder="twins"``).
#
# The Twins backbone below is a minimal inference-only copy of timm's
# ``timm/models/twins.py`` (timm 1.0.31, Apache License 2.0, Copyright 2019
# Ross Wightman), itself from https://github.com/Meituan-AutoML/Twins:
#   Copyright (c) 2021 Meituan. Licensed under The Apache 2.0 License.
#   Written by Xinjie Li, Xiangxiang Chu.
# Kept: the modules and parameter names WAFT checkpoints hold
# (``backbone.model.*``) and the features_only intermediates forward.
# Dropped: classifier head, final norm, drop path, weight init, timm registry.
# Attention always runs on ``F.scaled_dot_product_attention``.
#
# ``TwinsFeatureEncoder`` (the DPT-style fusion on top) is from WAFT
# model/backbone/twins.py, Copyright (c) 2025 Princeton Vision & Learning Lab,
# BSD-3-Clause (see LICENSE in this folder).
import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import _make_fusion_block


class LocallyGroupedAttn(nn.Module):
    def __init__(self, dim, num_heads=8, ws=1):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.ws = ws

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads
        ).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = x.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return self.proj(x.reshape(B, N, C))


class GlobalSubSampleAttn(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, ws=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, ws)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, size):
        x = x + self.attn(self.norm1(x), size)
        return x + self.mlp(self.norm2(x))


class PosConv(nn.Module):
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim)
        )
        self.stride = stride

    def forward(self, x, size):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x = x + cnn_feat_token
        return x.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.norm(self.proj(x).flatten(2).transpose(1, 2))
        return x, (H // self.patch_size[0], W // self.patch_size[1])


class Twins(nn.Module):
    """Twins-SVT backbone returning the four stage maps (NCHW)."""

    def __init__(self, patch_size=4, embed_dims=(128, 256, 512, 1024),
                 num_heads=(4, 8, 16, 32), mlp_ratios=(4, 4, 4, 4),
                 depths=(2, 2, 18, 2), sr_ratios=(8, 4, 2, 1), wss=(7, 7, 7, 7)):
        super().__init__()
        prev = 3
        self.patch_embeds = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(patch_size, prev, embed_dims[i]))
            prev = embed_dims[i]
            patch_size = 2
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                Block(embed_dims[k], num_heads[k], mlp_ratios[k], sr_ratios[k],
                      ws=1 if wss is None or i % 2 == 1 else wss[k])
                for i in range(depths[k])
            )
            for k in range(len(depths))
        )
        self.pos_block = nn.ModuleList(PosConv(d, d) for d in embed_dims)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for embed, blocks, pos_blk in zip(self.patch_embeds, self.blocks, self.pos_block):
            x, size = embed(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class _FeatureGetter(nn.Module):
    """Holds the backbone under ``model`` like timm's features_only getter."""

    def __init__(self):
        super().__init__()
        self.model = Twins()

    def forward(self, x):
        return self.model(x)


class TwinsFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FeatureGetter()
        self.out_channels = [128, 256, 512, 1024]
        self.features = 128
        self.output_dim = self.features // 2
        self.scratch = nn.ModuleList(
            nn.Conv2d(c, self.features, 3, 1, 1, bias=False) for c in self.out_channels
        )
        self.refine = nn.ModuleList(
            _make_fusion_block(self.features, False) for _ in self.out_channels
        )
        self.final = nn.ConvTranspose2d(self.features, self.features // 2, 4, 2, 1, bias=True)

    def forward(self, x):
        out = self.backbone(x)
        out_rn = [self.scratch[i](out[i]) for i in range(len(out))]
        for i in range(1, len(out_rn) + 1):
            if i == 1:
                out_rn[-i] = self.refine[-i](out_rn[-i], size=out_rn[-i].shape[2:])
            else:
                up = F.interpolate(out_rn[-i + 1], scale_factor=2, mode="bilinear", align_corners=True)
                out_rn[-i] = self.refine[-i](out_rn[-i], up, size=out_rn[-i].shape[2:])
        return self.final(out_rn[0])
