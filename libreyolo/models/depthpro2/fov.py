#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .encoder import _require_timm, autocast_for

#: Wider frames (long over short side) are centre-cropped to this; few training photos are wider.
MAX_ASPECT = 1.7778


class FOVHead(nn.Module):
    """A query token attends over four blocks' pooled patch, class and register tokens."""

    def __init__(
        self,
        embed_dim: int = 384,
        num_taps: int = 4,
        num_prefix: int = 5,
        dim: int = 248,
        depth: int = 2,
        heads: int = 4,
        pool_stride: int = 2,
    ):
        """Initialize the head; the defaults are the released estimator's."""
        super().__init__()
        _require_timm()
        from timm.models.vision_transformer import Block

        self.embed_dim = embed_dim
        self.num_prefix = num_prefix
        self.dim = dim
        self.pool_stride = pool_stride
        self.tap_norms = nn.ModuleList(
            nn.LayerNorm(embed_dim) for _ in range(num_taps)
        )
        self.proj = nn.Linear(embed_dim * num_taps, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.prefix_embed = nn.Parameter(torch.zeros(1, num_prefix, dim))
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)

    def forward(
        self, tokens: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """Map tapped tokens ``(T, num_prefix + gh*gw, num_taps*C)`` to ``(T, 2)`` logits."""
        num_frames = tokens.shape[0]
        gh, gw = grid_size
        taps = tokens.float().split(self.embed_dim, dim=-1)
        x = self.proj(
            torch.cat([norm(t) for norm, t in zip(self.tap_norms, taps)], dim=-1)
        )
        prefix, x = x[:, : self.num_prefix], x[:, self.num_prefix :]
        x = x.transpose(1, 2).reshape(num_frames, self.dim, gh, gw)
        x = F.avg_pool2d(x, self.pool_stride, ceil_mode=True)
        x = x.flatten(2).transpose(1, 2) + sincos_2d(*x.shape[-2:], self.dim, x.device)
        query = self.query.expand(num_frames, -1, -1)
        x = torch.cat([query, prefix + self.prefix_embed, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, 0]))


def sincos_2d(gh: int, gw: int, dim: int, device: torch.device) -> torch.Tensor:
    """``(gh*gw, dim)`` sin-cos code; both axes scale by the longer side, so it carries aspect."""
    g = max(gh, gw)
    ys = (torch.arange(gh, device=device) + 0.5 - gh / 2) / g
    xs = (torch.arange(gw, device=device) + 0.5 - gw / 2) / g
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    freq = torch.pi * 2.0 ** torch.linspace(0.0, 7.0, dim // 4, device=device)
    return torch.cat(
        [
            f(c.reshape(-1, 1) * freq)
            for c in (yy, xx)
            for f in (torch.sin, torch.cos)
        ],
        dim=-1,
    )


class FOVEstimator(nn.Module):
    """DINOv3 ViT-S/16 and an attention head, regressing each frame's two FoV angles."""

    max_short_side = 512
    min_fov_deg = 1.0
    max_fov_deg = 179.0
    tapped_blocks = frozenset((2, 5, 8, 11))
    # Cross-frame blocks in training; a frame alone keeps its input class and register tokens.
    cross_frame_blocks = frozenset((5, 7, 9, 11))

    def __init__(self, precision: torch.dtype = torch.float32):
        super().__init__()
        timm = _require_timm()
        vit = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )
        vit.rope.normalize_coords = "max"
        self.trunk = nn.Module()
        self.trunk.patch_embed = vit.patch_embed
        self.trunk.cls_token = vit.cls_token
        self.trunk.reg_token = vit.reg_token
        self.trunk.rope = vit.rope
        self.trunk.blocks = vit.blocks
        self.num_prefix_tokens: int = vit.num_prefix_tokens
        self.head = FOVHead(embed_dim=vit.embed_dim, num_prefix=self.num_prefix_tokens)
        self.precision = precision

    def input_size(self, height: int, width: int) -> tuple[int, int]:
        """Resize target of a ``height`` x ``width`` frame, before any crop; never upsampled."""
        short = max(32, min(self.max_short_side, min(height, width) // 32 * 32))
        scale = short / min(height, width)
        return (
            max(32, round(height * scale / 32) * 32),
            max(32, round(width * scale / 32) * 32),
        )

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize frames ``(T, 3, H, W)`` to ``input_size``; centre-crop those wider than 16:9."""
        new_h, new_w = self.input_size(*x.shape[-2:])
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", antialias=True)
        lo, hi = min(new_h, new_w), max(new_h, new_w)
        if hi <= MAX_ASPECT * lo:
            return x
        keep = max(32, int(MAX_ASPECT * lo) // 32 * 32)
        off = (hi - keep) // 2
        return x[..., off : off + keep] if new_w > new_h else x[..., off : off + keep, :]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Tapped tokens ``(T, num_prefix + gh*gw, 4*C)`` and grid of frames ``(T, 3, h, w)``."""
        trunk: Any = self.trunk
        grid_size = (x.shape[-2] // 16, x.shape[-1] // 16)
        rope = trunk.rope.get_embed(shape=grid_size)
        tokens = trunk.patch_embed(x).flatten(1, 2)
        num_frames = tokens.shape[0]
        prefix = [
            t.expand(num_frames, -1, -1) for t in (trunk.cls_token, trunk.reg_token)
        ]
        tokens = torch.cat(prefix + [tokens], dim=1)
        npt = self.num_prefix_tokens
        taps = []
        for idx, block in enumerate(trunk.blocks):
            out = block(tokens, rope=rope)
            if idx in self.cross_frame_blocks:
                out = torch.cat([tokens[:, :npt], out[:, npt:]], dim=1)
            tokens = out
            if idx in self.tapped_blocks:
                taps.append(tokens)
        return torch.cat(taps, dim=-1), grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict ``(T, 2)`` horizontal and vertical FoV in degrees for frames ``(T, 3, H, W)``.

        Angles of frames cropped to 16:9 are converted back to the full frame.
        """
        height, width = x.shape[-2:]
        y = self.resize(x)
        with autocast_for(x.device, self.precision)():
            tokens, grid_size = self.encode(y)
        logits = self.head(tokens, grid_size)
        fov_deg = self.min_fov_deg + torch.sigmoid(logits) * (
            self.max_fov_deg - self.min_fov_deg
        )
        full_h, full_w = self.input_size(height, width)
        crop_h, crop_w = y.shape[-2:]
        if (crop_h, crop_w) == (full_h, full_w):
            return fov_deg
        crop_in_source = (crop_h * height / full_h, crop_w * width / full_w)
        return crop_angles_to_full(fov_deg, crop_in_source, (height, width))

    def focal_length_px(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate one focal length in pixels for frames ``(T, 3, H, W)`` from the shorter side."""
        return median_focal_length_px(self.forward(x), *x.shape[-2:])


def median_focal_length_px(
    fov_deg: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """The focal length in pixels at the lower median of ``(T, 2)`` FoVs of the shorter side."""
    hfov_deg, vfov_deg = fov_deg.median(dim=0).values
    if width < height:
        return 0.5 * width / torch.tan(torch.deg2rad(hfov_deg) / 2.0)
    return 0.5 * height / torch.tan(torch.deg2rad(vfov_deg) / 2.0)


def crop_angles_to_full(
    angles: torch.Tensor, crop_hw: tuple[float, float], full_hw: tuple[float, float]
) -> torch.Tensor:
    """``(..., 2)`` (hfov, vfov) degrees of a centred crop -> the full frame's at the same focal."""
    (ch, cw), (h, w) = crop_hw, full_hw
    # Float32 is within 3e-5 degrees of float64 over the clamp range, and MPS has no float64.
    a = torch.deg2rad(angles.to(torch.promote_types(angles.dtype, torch.float32)))
    fx = (cw / 2) / torch.tan(a[..., 0] / 2)
    fy = (ch / 2) / torch.tan(a[..., 1] / 2)
    out = torch.stack(
        [2 * torch.atan((w / 2) / fx), 2 * torch.atan((h / 2) / fy)], dim=-1
    )
    return torch.rad2deg(out).to(angles.dtype)
