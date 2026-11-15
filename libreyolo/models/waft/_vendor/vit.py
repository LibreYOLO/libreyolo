# Copyright (c) 2025, Princeton Vision & Learning Lab. BSD-3-Clause, see LICENSE.
# Adapted from WAFT model/backbone/vit.py: the iterative refinement ViT.
# Upstream builds the transformer blocks with timm's ``vit_*_patch16_224``;
# ``Block`` below is a plain-torch equivalent of timm's ViT block (pre-norm,
# fused qkv, GELU MLP, no layer scale) with the same parameter names, running
# attention on ``F.scaled_dot_product_attention``.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...depth_anything._vendor.dpt import DPTHead as DAv2DPTHead
from .patch_embed import PatchEmbed

MODEL_CONFIGS = {
    "vitl": {"embed_dim": 1024, "depth": 24, "heads": 16, "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitb": {"embed_dim": 768, "depth": 12, "heads": 12, "features": 128, "out_channels": [96, 192, 384, 768]},
    "vits": {"embed_dim": 384, "depth": 12, "heads": 6, "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitt": {"embed_dim": 192, "depth": 12, "heads": 3, "features": 32, "out_channels": [24, 48, 96, 192]},
}

INTERMEDIATE_LAYER_IDX = {
    "vitt": [2, 5, 8, 11],
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def dpt_intermediate(head, out_features, patch_h, patch_w):
    """Depth Anything V2 ``DPTHead.forward(..., return_intermediate=True)``.

    WAFT patches the Depth Anything V2 head to stop before ``output_conv2`` and
    return the fused paths; this reproduces that on the repo's vendored head.
    """
    out = []
    for i, x in enumerate(out_features):
        if head.use_clstoken:
            x, cls_token = x[0], x[1]
            readout = cls_token.unsqueeze(1).expand_as(x)
            x = head.readout_projects[i](torch.cat((x, readout), -1))
        else:
            x = x[0]
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
        x = head.projects[i](x)
        x = head.resize_layers[i](x)
        out.append(x)
    layer_1, layer_2, layer_3, layer_4 = out
    s = head.scratch
    layer_1_rn = s.layer1_rn(layer_1)
    layer_2_rn = s.layer2_rn(layer_2)
    layer_3_rn = s.layer3_rn(layer_3)
    layer_4_rn = s.layer4_rn(layer_4)
    path_4 = s.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
    path_3 = s.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
    path_2 = s.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
    path_1 = s.refinenet1(path_2, layer_1_rn)
    out = s.output_conv1(path_1)
    out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
    return out, path_1, path_2, path_3, path_4


class VisionTransformer(nn.Module):
    def __init__(self, model_name, input_dim, patch_size=16):
        super().__init__()
        cfg = MODEL_CONFIGS[model_name]
        self.idx = INTERMEDIATE_LAYER_IDX[model_name]
        self.embed_dim = cfg["embed_dim"]
        self.blks = nn.ModuleList(
            [Block(self.embed_dim, cfg["heads"]) for _ in range(cfg["depth"])]
        )
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = cfg["features"]
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size,
            in_chans=input_dim, embed_dim=self.embed_dim,
        )
        self.dpt_head = DAv2DPTHead(self.embed_dim, cfg["features"], out_channels=cfg["out_channels"])

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = F.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx), mode="bicubic", antialias=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)
        outputs = []
        for i, blk in enumerate(self.blks):
            x = blk(x)
            if i in self.idx:
                outputs.append([x])
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        out, *_ = dpt_intermediate(self.dpt_head, outputs, patch_h, patch_w)
        return F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
