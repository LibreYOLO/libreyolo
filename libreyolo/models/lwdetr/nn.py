"""LW-DETR architecture: plain-ViT encoder, multi-scale projector, DETR decoder.

Ported from LW-DETR (https://github.com/Atten4Vis/LW-DETR).
Copyright (c) 2024 Baidu. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).
Copyright (c) Facebook, Inc. and its affiliates.
Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR).
Copyright (c) 2021 Microsoft. All Rights Reserved.
Modified from DETR (https://github.com/facebookresearch/detr).
Copyright (c) Facebook, Inc. and its affiliates.
Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR).
Copyright (c) 2020 SenseTime. All Rights Reserved.

Module and attribute names mirror upstream so converting an upstream checkpoint
is a metadata wrap with no key surgery.

Scope: inference. The Group-DETR one-to-many training path (``group_detr``
query groups in self-attention, Hungarian matching, the IoU-aware
classification loss) is not implemented here; the checkpoint's 13 query groups
and per-group two-stage heads are still materialised so upstream weights load
strictly and a trainer can be added later without reconverting.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ...kernels.attention.ms_deform_attn import (
    maybe_ms_deform_attn,
    ms_deform_attn_available,
    spatial_shapes_tensor,
)
from ...kernels.attention.sdpa import manual_attention_required

__all__ = ["LWDETR_CONFIGS", "LWDETRExportWrapper", "LibreLWDETRModel"]


# =============================================================================
# Per-size configuration (mirrors scripts/lwdetr_<size>_coco_eval.sh upstream,
# cross-checked against the ``args`` namespace stored in each released
# checkpoint).
# =============================================================================

_VIT_VARIANTS = {
    # name: (embed_dim, num_heads)
    "vit_tiny": (192, 12),
    "vit_small": (384, 12),
    "vit_base": (768, 12),
}

# Upstream ``level2scalefactor`` in models/backbone/backbone.py.
_LEVEL_TO_SCALE_FACTOR = {"P3": 2.0, "P4": 1.0, "P5": 0.5, "P6": 0.25}

LWDETR_CONFIGS: dict[str, dict] = {
    "t": {
        "encoder": "vit_tiny",
        "vit_encoder_num_layers": 6,
        "window_block_indexes": (0, 2, 4),
        "out_feature_indexes": (1, 3, 5),
        "projector_scale": ("P4",),
        "hidden_dim": 256,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "num_queries": 100,
        "num_select": 100,
    },
    "s": {
        "encoder": "vit_tiny",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": (0, 1, 3, 6, 7, 9),
        "out_feature_indexes": (2, 4, 5, 9),
        "projector_scale": ("P4",),
        "hidden_dim": 256,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "num_queries": 300,
        "num_select": 300,
    },
    "m": {
        "encoder": "vit_small",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": (0, 1, 3, 6, 7, 9),
        "out_feature_indexes": (2, 4, 5, 9),
        "projector_scale": ("P4",),
        "hidden_dim": 256,
        "sa_nheads": 8,
        "ca_nheads": 16,
        "dec_n_points": 2,
        "num_queries": 300,
        "num_select": 300,
    },
    "l": {
        "encoder": "vit_small",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": (0, 1, 3, 6, 7, 9),
        "out_feature_indexes": (2, 4, 5, 9),
        "projector_scale": ("P3", "P5"),
        "hidden_dim": 384,
        "sa_nheads": 12,
        "ca_nheads": 24,
        "dec_n_points": 4,
        "num_queries": 300,
        "num_select": 300,
    },
    "x": {
        "encoder": "vit_base",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": (0, 1, 3, 6, 7, 9),
        "out_feature_indexes": (2, 4, 5, 9),
        "projector_scale": ("P3", "P5"),
        "hidden_dim": 384,
        "sa_nheads": 12,
        "ca_nheads": 24,
        "dec_n_points": 4,
        "num_queries": 300,
        "num_select": 300,
    },
}

# Shared by every released size.
_SHARED_CONFIG = {
    "dec_layers": 3,
    "group_detr": 13,
    "dim_feedforward": 2048,
    "dropout": 0.0,
    "two_stage": True,
    "bbox_reparam": True,
    "lite_refpoint_refine": True,
    "decoder_norm": "LN",
    "patch_size": 16,
    "pretrain_img_size": 224,
    "mlp_ratio": 4.0,
}

# The ViT partitions the patch grid into a fixed 4x4 arrangement of windows, so
# both patch-grid dimensions must be divisible by 4 — i.e. the image side must
# be a multiple of ``patch_size * 4``.  Upstream enforces this with
# ``--square_resize_div_64``.
SIZE_DIVISOR = 64


def build_config(size: str) -> dict:
    """Return the full architecture config for a size code."""
    if size not in LWDETR_CONFIGS:
        raise ValueError(
            f"Unknown LW-DETR size '{size}'. Valid: {', '.join(LWDETR_CONFIGS)}"
        )
    config = dict(_SHARED_CONFIG)
    config.update(LWDETR_CONFIGS[size])
    return config


# =============================================================================
# ViT encoder (ViTDet-style plain ViT with interleaved window/global attention)
# =============================================================================


def get_abs_pos(abs_pos: Tensor, has_cls_token: bool, hw: tuple[int, int]) -> Tensor:
    """Resize absolute position embeddings to the current token grid.

    Drops the class token when present and bicubically resizes the square
    pretraining grid to ``hw``.
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    if size * size != xy_num:
        raise ValueError(
            f"Position embedding length {xy_num} is not a perfect square."
        )

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)


class PatchEmbed(nn.Module):
    """Image to patch embedding (strided conv), emitting ``B H W C``."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


class Mlp(nn.Module):
    """Two-layer MLP.

    Matches the ``timm.layers.Mlp`` parameter layout (``fc1`` / ``fc2``) that
    upstream instantiates, without taking a timm dependency.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """ViT multi-head self-attention.

    ``use_cae`` mirrors the CAE v2 pretraining layout: a bias-free packed
    ``qkv`` projection plus separate ``q_bias`` / ``v_bias`` parameters (the key
    projection is deliberately unbiased).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_cae: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.use_cae = use_cae
        if use_cae:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
            # The key projection is deliberately unbiased. Upstream materialises
            # the zero block with ``torch.zeros_like(v_bias)``; holding it as a
            # non-persistent buffer keeps the value identical while making it an
            # ONNX initializer on the module's device. Built as a traced
            # constant instead, ONNX constant folding would try to concatenate a
            # CPU tensor with CUDA parameters and fail on GPU export. Not
            # persistent, so checkpoints are unaffected.
            self.register_buffer("k_bias", torch.zeros(dim), persistent=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # Opt-in fused SDPA. Default off: weights/parity_lwdetr.py pins
        # max_abs_diff == 0.0 against the official model and fused kernels
        # accumulate in a different order. Flip with
        # libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

    def forward(self, x: Tensor) -> Tensor:
        batch, num_tokens, channels = x.shape
        if self.use_cae:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch, num_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if self.fused_attn and not manual_attention_required():
            x = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=self.scale,
            )
        else:
            attn = (query * self.scale) @ key.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ value
        x = x.transpose(1, 2).reshape(batch, num_tokens, channels)
        return self.proj(x)


class Block(nn.Module):
    """ViT block; ``window=True`` keeps attention inside one of the 4x4 windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        window: bool = False,
        use_cae: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_cae=use_cae
        )
        # Upstream wires a DropPath here; with drop_path_rate=0 for every
        # released size it is an identity, and this port is inference-only.
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer
        )

        self.window = window
        self.use_cae = use_cae
        if use_cae:
            init_values = 0.1
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        batch_windows, tokens_per_window, channels = x.shape
        shortcut = x
        x = self.norm1(x)

        if not self.window:
            # Merge the 16 windows back into one sequence for global attention.
            x = x.reshape(batch_windows // 16, 16 * tokens_per_window, channels)

        x = self.gamma_1 * self.attn(x) if self.use_cae else self.attn(x)

        if not self.window:
            x = x.reshape(batch_windows, tokens_per_window, channels)

        x = shortcut + self.drop_path(x)
        if self.use_cae:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    """Plain ViT encoder emitting several intermediate feature maps.

    The patch grid is split into a fixed 4x4 window arrangement; blocks listed
    in ``window_block_indexes`` attend inside a window, the rest attend
    globally. Selected blocks in ``out_feature_indexes`` are unfolded back to
    ``B C H W`` and handed to the projector.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        window_block_indexes: Sequence[int] = (),
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        out_feature_indexes: Optional[Sequence[int]] = None,
        use_cae: bool = False,
    ) -> None:
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) ** 2
            num_positions = num_patches + 1 if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                window=i in window_block_indexes,
                use_cae=use_cae,
            )
            for i in range(depth)
        )

        self.window_block_indexes = window_block_indexes
        out_feature_indexes = [
            idx if idx >= 0 else idx + depth for idx in (out_feature_indexes or [-1])
        ]
        out_feature_indexes = [i for i in range(depth) if i in out_feature_indexes]

        self._out_features = [i in out_feature_indexes for i in range(depth)]
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)
        if not self._out_features[-1]:
            raise ValueError("The last ViT block must be an output feature.")

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        batch, height, width, channels = x.shape
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError(
                f"Patch grid {height}x{width} must be divisible by 4 — feed images "
                f"whose sides are multiples of {SIZE_DIVISOR}."
            )
        win_h, win_w = height // 4, width // 4

        x = (
            x.reshape(batch, 4, win_h, 4, win_w, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch * 16, win_h * win_w, channels)
        )
        out: list[Tensor] = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if self._out_features[idx]:
                out.append(
                    x.reshape(batch, 4, 4, win_h, win_w, channels)
                    .permute(0, 5, 1, 3, 2, 4)
                    .reshape(batch, channels, height, width)
                )
        return out


# =============================================================================
# Multi-scale projector
# =============================================================================


class LayerNorm(nn.Module):
    """Channel-first LayerNorm over ``(B, C, H, W)`` inputs."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def get_norm(norm: Optional[str], out_channels: int) -> Optional[nn.Module]:
    if not norm:
        return None
    if norm != "LN":
        raise ValueError(f"Unsupported norm type: {norm}")
    return LayerNorm(out_channels)


def get_activation(name: Optional[str], inplace: bool = False) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in ("LeakyReLU", "leakyrelu", "lrelu"):
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name is None:
        return nn.Identity()
    raise ValueError(f"Unsupported act type: {name}")


class ConvX(nn.Module):
    """Conv-BN-act block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck used inside :class:`C2f`."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
        act: str = "silu",
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with two convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act)
            for _ in range(n)
        )

    def forward(self, x: Tensor) -> Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """Turn the ViT's single-stride features into a small feature pyramid."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float],
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.scale_factors = scale_factors

        stages_sampling: list[nn.ModuleList] = []
        stages: list[nn.Module] = []
        self.use_extra_pool = False
        for scale in scale_factors:
            per_scale: list[nn.Module] = []
            out_dim = 0
            for in_dim in in_channels:
                out_dim = in_dim
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm("LN", in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            in_dim // 2, in_dim // 4, kernel_size=2, stride=2
                        ),
                    ]
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    if in_dim > 512:
                        # Upstream trims the channel count first when the ViT
                        # feature is wide (xlarge), to keep params/FLOPs down.
                        layers = [
                            ConvX(in_dim, in_dim // 2, kernel=1),
                            nn.ConvTranspose2d(
                                in_dim // 2, in_dim // 4, kernel_size=2, stride=2
                            ),
                        ]
                        out_dim = in_dim // 4
                    else:
                        layers = [
                            nn.ConvTranspose2d(
                                in_dim, in_dim // 2, kernel_size=2, stride=2
                            )
                        ]
                        out_dim = in_dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [ConvX(in_dim, in_dim, 3, 2)]
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError(f"Unsupported scale_factor: {scale}")
                per_scale.append(nn.Sequential(*layers))
            stages_sampling.append(nn.ModuleList(per_scale))

            fused_dim = out_dim * len(in_channels)
            stages.append(
                nn.Sequential(
                    C2f(fused_dim, out_channels, num_blocks),
                    get_norm("LN", out_channels),
                )
            )

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Sequence[Tensor]) -> list[Tensor]:
        results: list[Tensor] = []
        for i, stage in enumerate(self.stages):
            feat_fuse = [
                stage_sampling(x[j])
                for j, stage_sampling in enumerate(self.stages_sampling[i])
            ]
            fused = torch.cat(feat_fuse, dim=1) if len(feat_fuse) > 1 else feat_fuse[0]
            results.append(stage(fused))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class Backbone(nn.Module):
    """ViT encoder plus multi-scale projector."""

    def __init__(
        self,
        name: str,
        vit_encoder_num_layers: int,
        window_block_indexes: Sequence[int],
        out_channels: int,
        out_feature_indexes: Sequence[int],
        projector_scale: Sequence[str],
        patch_size: int = 16,
        pretrain_img_size: int = 224,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.name = name
        if name not in _VIT_VARIANTS:
            raise NotImplementedError(f"Backbone {name} is not supported.")
        embed_dim, num_heads = _VIT_VARIANTS[name]

        self.encoder = ViT(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_encoder_num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
            window_block_indexes=window_block_indexes,
            use_abs_pos=True,
            pretrain_img_size=pretrain_img_size,
            out_feature_indexes=out_feature_indexes,
            use_cae=True,
        )

        self.projector_scale = list(projector_scale)
        if not self.projector_scale:
            raise ValueError("projector_scale must list at least one level.")
        if sorted(self.projector_scale) != self.projector_scale:
            raise ValueError("projector scales must be given in ascending order.")

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=[_LEVEL_TO_SCALE_FACTOR[lvl] for lvl in self.projector_scale],
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        return self.projector(self.encoder(x))


# =============================================================================
# Position encoding
# =============================================================================


class PositionEmbeddingSine(nn.Module):
    """Sine position embedding over a feature map, returned as ``B C H W``."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, mask: Tensor) -> Tensor:
        """``mask`` is ``(B, H, W)`` with True marking padded pixels."""
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class Joiner(nn.Sequential):
    """``nn.Sequential(backbone, position_embedding)`` — upstream key layout.

    Keeping the Sequential means checkpoint keys stay ``backbone.0.*``; the
    position embedding at index 1 carries no parameters.

    The sine embedding is built but never consumed upstream: the transformer
    flattens it into ``lvl_pos_embed_flatten``, threads it down to
    ``TransformerDecoderLayer.forward_post`` as ``pos``, and that method never
    reads it — decoder positional information arrives through ``query_pos``
    (from ``ref_point_head``) and the deformable reference points instead. This
    port therefore skips computing it. The module is retained so the
    architecture stays legible and a future encoder-side user has it available.
    """

    def __init__(self, backbone: Backbone, position_embedding: PositionEmbeddingSine):
        super().__init__(backbone, position_embedding)

    def forward(self, x: Tensor) -> list[Tensor]:
        return self[0](x)


# =============================================================================
# Deformable attention
# =============================================================================


def ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Sequence[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Pure-PyTorch multi-scale deformable attention core.

    Upstream ships a CUDA extension plus this reference path and takes the
    reference path whenever exporting or running fp16. LibreYOLO always takes
    it: no build step, runs on every device, and exports to ONNX opset >= 16
    (``grid_sample`` -> ``GridSample``).

    When the optional accelerated ``ms_deform_attn`` kernel slot resolves
    (see ``libreyolo/kernels/attention/ms_deform_attn.py``) it takes over;
    the grid_sample path below stays the default and the export path.
    """
    if ms_deform_attn_available(value):
        weights = attention_weights
        if weights.dim() == 4:
            # The caller flattens (levels, points); the slot takes them split.
            weights = weights.unflatten(-1, sampling_locations.shape[-3:-1])
        accelerated = maybe_ms_deform_attn(
            # (bs, heads, c, Len_in) -> the slot's (bs, Len_in, heads, c).
            value.permute(0, 3, 1, 2),
            spatial_shapes_tensor(value_spatial_shapes, value.device),
            sampling_locations,
            weights,
        )
        if accelerated is not None:
            return accelerated
    batch, n_heads, head_dim, _ = value.shape
    _, len_query, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (height, width) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(batch * n_heads, head_dim, height, width)
        sampling_grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch * n_heads, 1, len_query, num_levels * num_points
    )
    stacked = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (stacked * attention_weights).sum(-1).view(
        batch, n_heads * head_dim, len_query
    )
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Deformable DETR)."""

    def __init__(
        self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        batch, len_query, _ = query.shape
        _, len_input, _ = input_flatten.shape

        value = self.value_proj(input_flatten)

        sampling_offsets = self.sampling_offsets(query).view(
            batch, len_query, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch, len_query, self.n_heads, self.n_levels * self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.as_tensor(
                [[w, h] for h, w in input_spatial_shapes],
                dtype=sampling_offsets.dtype,
                device=sampling_offsets.device,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but got "
                f"{reference_points.shape[-1]} instead."
            )
        attention_weights = F.softmax(attention_weights, -1)

        value = (
            value.transpose(1, 2)
            .contiguous()
            .view(batch, self.n_heads, self.d_model // self.n_heads, len_input)
        )
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


# =============================================================================
# Decoder self-attention
# =============================================================================


class MultiheadAttention(nn.Module):
    """Batch-first multi-head attention with ``nn.MultiheadAttention`` weights.

    Written out explicitly rather than delegating to ``nn.MultiheadAttention``:
    upstream vendors its own copy of the pre-fast-path implementation, and
    torch's fused kernels select a different accumulation order, which would
    put a small numeric gap between this port and the reference. The parameter
    layout (``in_proj_weight`` / ``in_proj_bias`` / ``out_proj``) is unchanged,
    so checkpoints interchange.

    Inference only: attention and key-padding masks are not plumbed because
    LW-DETR's decoder never uses them outside Group-DETR training.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Opt-in fused SDPA: exactly the accumulation-order gap the class
        # docstring above describes, which weights/parity_lwdetr.py's
        # max_abs_diff == 0.0 bar would catch. Flip with
        # libreyolo.kernels.attention.set_fused_attention(model).
        self.fused_attn = False

        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        batch, tgt_len, embed_dim = query.shape

        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        def _split_heads(tensor: Tensor) -> Tensor:
            return (
                tensor.contiguous()
                .view(batch, -1, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(batch * self.num_heads, -1, self.head_dim)
            )

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)

        if self.fused_attn and not manual_attention_required():
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            q = q / math.sqrt(self.head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            attn_output = torch.bmm(attn, v)

        attn_output = (
            attn_output.permute(1, 0, 2)
            .contiguous()
            .view(tgt_len, batch, embed_dim)
            .transpose(0, 1)
            .contiguous()
        )
        return F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)


# =============================================================================
# Transformer
# =============================================================================


class MLP(nn.Module):
    """Multi-layer perceptron (FFN) with ReLU between layers."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor: Tensor, dim: int = 128) -> Tensor:
    """Sine-embed reference boxes (cx, cy[, w, h]) for the decoder query pos."""
    scale = 2 * math.pi
    # Built from pos_tensor rather than torch.arange so the frequency table is a
    # derived tensor, not a baked constant. A traced constant keeps whatever
    # device it was created on, which breaks TorchScript artifacts run on a
    # different device. cumsum-of-ones reproduces arange exactly.
    dim_t = pos_tensor.new_ones(int(dim)).cumsum(0) - 1.0
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")


def gen_encoder_output_proposals(
    memory: Tensor,
    spatial_shapes: Sequence[tuple[int, int]],
    unsigmoid: bool = True,
) -> tuple[Tensor, Tensor]:
    """Build the dense anchor grid the two-stage query selection scores.

    ``memory`` is ``(B, sum(H*W), C)``; the returned proposals are one anchor
    per feature-map cell, in cxcywh. No padding mask is threaded: LibreYOLO
    feeds a single un-padded square canvas, which makes the mask all-False and
    every masked_fill a no-op.
    """
    batch, _, _ = memory.shape
    proposals = []
    for level, (height, width) in enumerate(spatial_shapes):
        # The anchor grid depends only on the feature-map shape, so a literal
        # meshgrid/linspace would be traced as a constant pinned to whatever
        # device built it — and a TorchScript artifact then fails when run
        # elsewhere. Deriving it from ``memory`` keeps the device following the
        # input. cumsum-of-ones equals linspace(0, n-1, n) exactly, and
        # normalising by Python floats matches dividing by a [W, H] tensor.
        ones = memory.new_ones((height, width))
        grid_y = (ones.cumsum(0) - 1.0 + 0.5) / float(height)
        grid_x = (ones.cumsum(1) - 1.0 + 0.5) / float(width)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

        wh = torch.ones_like(grid) * 0.05 * (2.0**level)
        proposals.append(torch.cat((grid, wh), -1).reshape(batch, -1, 4))

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
        -1, keepdim=True
    )

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )
    else:
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class TransformerDecoderLayer(nn.Module):
    """Self-attention over queries, deformable cross-attention, FFN."""

    def __init__(
        self,
        d_model: int,
        sa_nhead: int,
        ca_nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points
        )
        self.nhead = ca_nhead

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        query_pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> Tensor:
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)


class TransformerDecoder(nn.Module):
    """Shallow DETR decoder with optional per-layer reference-box refinement."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        d_model: int = 256,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.lite_refpoint_refine = lite_refpoint_refine
        self.bbox_reparam = bbox_reparam

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)

        # Set by LibreLWDETRModel: the shared box head when iterative refinement
        # is on, otherwise None (every released size uses lite refinement).
        self.bbox_embed: Optional[nn.Module] = None

    def refpoints_refine(
        self, refpoints_unsigmoid: Tensor, new_refpoints_delta: Tensor
    ) -> Tensor:
        if self.bbox_reparam:
            new_cxcy = (
                new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:]
                + refpoints_unsigmoid[..., :2]
            )
            new_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            return torch.concat([new_cxcy, new_wh], dim=-1)
        return refpoints_unsigmoid + new_refpoints_delta

    def _get_reference(self, refpoints: Tensor) -> tuple[Tensor, Tensor]:
        obj_center = refpoints[..., :4]
        reference_points = obj_center[:, :, None]
        query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model / 2)
        return reference_points, self.ref_point_head(query_sine_embed)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        refpoints_unsigmoid: Tensor,
        spatial_shapes: Sequence[tuple[int, int]],
    ) -> tuple[Tensor, Tensor]:
        output = tgt
        intermediate: list[Tensor] = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        if self.lite_refpoint_refine:
            reference_points, query_pos = self._get_reference(
                refpoints_unsigmoid
                if self.bbox_reparam
                else refpoints_unsigmoid.sigmoid()
            )

        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                reference_points, query_pos = self._get_reference(
                    refpoints_unsigmoid
                    if self.bbox_reparam
                    else refpoints_unsigmoid.sigmoid()
                )

            output = layer(
                output,
                memory,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
            )

            if not self.lite_refpoint_refine:
                new_refpoints_unsigmoid = self.refpoints_refine(
                    refpoints_unsigmoid, self.bbox_embed(output)
                )
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()

            intermediate.append(self.norm(output) if self.norm is not None else output)

        if self.bbox_embed is not None:
            return torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)
        return torch.stack(intermediate), refpoints_unsigmoid.unsqueeze(0)


class Transformer(nn.Module):
    """Two-stage query selection plus the shallow deformable decoder."""

    def __init__(
        self,
        d_model: int = 512,
        sa_nhead: int = 8,
        ca_nhead: int = 8,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
        group_detr: int = 1,
        two_stage: bool = False,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        lite_refpoint_refine: bool = False,
        decoder_norm_type: str = "LN",
        bbox_reparam: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(
            d_model,
            sa_nhead,
            ca_nhead,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
        )
        if decoder_norm_type not in ("LN", "Identity"):
            raise ValueError(f"Unsupported decoder norm: {decoder_norm_type}")
        decoder_norm = (
            nn.LayerNorm(d_model) if decoder_norm_type == "LN" else nn.Identity()
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            d_model=d_model,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(group_detr)]
            )
            self.enc_output_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(group_detr)]
            )

        self._reset_parameters()

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(
        self,
        srcs: Sequence[Tensor],
        refpoint_embed: Tensor,
        query_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = srcs[0].shape[0]
        src_flatten = []
        spatial_shapes: list[tuple[int, int]] = []
        for src in srcs:
            spatial_shapes.append((int(src.shape[2]), int(src.shape[3])))
            src_flatten.append(src.flatten(2).transpose(1, 2))
        memory = torch.cat(src_flatten, 1)

        # Inference decodes a single query group; the other 12 exist only for
        # Group-DETR's one-to-many training supervision.
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, spatial_shapes, unsigmoid=not self.bbox_reparam
            )
            output_memory = self.enc_output_norm[0](self.enc_output[0](output_memory))

            enc_outputs_class = self.enc_out_class_embed[0](output_memory)
            if self.bbox_reparam:
                coord_delta = self.enc_out_bbox_embed[0](output_memory)
                coord_cxcy = (
                    coord_delta[..., :2] * output_proposals[..., 2:]
                    + output_proposals[..., :2]
                )
                coord_wh = coord_delta[..., 2:].exp() * output_proposals[..., 2:]
                enc_outputs_coord = torch.concat([coord_cxcy, coord_wh], dim=-1)
            else:
                enc_outputs_coord = (
                    self.enc_out_bbox_embed[0](output_memory) + output_proposals
                )

            topk_proposals = torch.topk(
                enc_outputs_class.max(-1)[0], self.num_queries, dim=1
            )[1]
            refpoint_embed_ts = torch.gather(
                enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )

        tgt = query_feat.unsqueeze(0).repeat(batch, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(batch, 1, 1)
        if self.two_stage:
            if self.bbox_reparam:
                cxcy = (
                    refpoint_embed[..., :2] * refpoint_embed_ts[..., 2:]
                    + refpoint_embed_ts[..., :2]
                )
                wh = refpoint_embed[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                refpoint_embed = torch.concat([cxcy, wh], dim=-1)
            else:
                refpoint_embed = refpoint_embed + refpoint_embed_ts

        return self.decoder(
            tgt,
            memory,
            refpoints_unsigmoid=refpoint_embed,
            spatial_shapes=spatial_shapes,
        )


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# =============================================================================
# Detector
# =============================================================================


class LibreLWDETRModel(nn.Module):
    """LW-DETR detector.

    Forward takes a batched image tensor and returns
    ``{"pred_logits": (B, Q, nc), "pred_boxes": (B, Q, 4)}`` with boxes in
    cxcywh normalized to [0, 1] — the LibreYOLO DETR output contract.
    """

    def __init__(self, size: str = "s", nc: int = 91) -> None:
        super().__init__()
        config = build_config(size)
        self.size = size
        self.num_classes = nc
        self.num_queries = config["num_queries"]
        self.num_select = config["num_select"]
        self.group_detr = config["group_detr"]
        self.two_stage = config["two_stage"]
        self.bbox_reparam = config["bbox_reparam"]
        self.lite_refpoint_refine = config["lite_refpoint_refine"]
        self.hidden_dim = config["hidden_dim"]

        backbone = Backbone(
            config["encoder"],
            config["vit_encoder_num_layers"],
            window_block_indexes=config["window_block_indexes"],
            out_channels=config["hidden_dim"],
            out_feature_indexes=config["out_feature_indexes"],
            projector_scale=config["projector_scale"],
            patch_size=config["patch_size"],
            pretrain_img_size=config["pretrain_img_size"],
            mlp_ratio=config["mlp_ratio"],
        )
        position_embedding = PositionEmbeddingSine(
            config["hidden_dim"] // 2, normalize=True
        )
        self.backbone = Joiner(backbone, position_embedding)

        self.transformer = Transformer(
            d_model=config["hidden_dim"],
            sa_nhead=config["sa_nheads"],
            ca_nhead=config["ca_nheads"],
            num_queries=config["num_queries"],
            num_decoder_layers=config["dec_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            group_detr=config["group_detr"],
            two_stage=config["two_stage"],
            num_feature_levels=len(config["projector_scale"]),
            dec_n_points=config["dec_n_points"],
            lite_refpoint_refine=config["lite_refpoint_refine"],
            decoder_norm_type=config["decoder_norm"],
            bbox_reparam=config["bbox_reparam"],
        )

        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, nc)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.refpoint_embed = nn.Embedding(self.num_queries * self.group_detr, 4)
        self.query_feat = nn.Embedding(
            self.num_queries * self.group_detr, hidden_dim
        )
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        self.transformer.decoder.bbox_embed = (
            None if self.lite_refpoint_refine else self.bbox_embed
        )

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(nc) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(self.group_detr)]
            )
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(self.group_detr)]
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feats = self.backbone(x)

        # Inference uses one query group; the remaining group_detr-1 groups are
        # a training-time device (Group-DETR one-to-many supervision).
        refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
        query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid = self.transformer(
            feats, refpoint_embed_weight, query_feat_weight
        )

        if self.bbox_reparam:
            coord_delta = self.bbox_embed(hs)
            coord_cxcy = (
                coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            )
            coord_wh = coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.concat([coord_cxcy, coord_wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        outputs_class = self.class_embed(hs)
        return {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}


class LWDETRExportWrapper(nn.Module):
    """Flatten the output dict to a tuple for tracing-based export.

    ``torch.onnx.export`` cannot give named outputs to a dict return, so the
    graph emits ``(pred_logits, pred_boxes)`` in declaration order — the same
    contract the other DETR families export.
    """

    def __init__(self, model: LibreLWDETRModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.model(x)
        return out["pred_logits"], out["pred_boxes"]
