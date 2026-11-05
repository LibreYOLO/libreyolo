"""Video Depth Anything network for LibreYOLO.

The architecture is upstream ``VideoDepthAnything`` (DINOv2 encoder plus the
temporal DPT head) with ImageNet normalization moved inside ``forward`` so the
network reads ``[0, 1]`` frames like every other LibreYOLO dense family.
Parameter names match upstream (``pretrained.*``, ``head.*``), so an upstream
checkpoint loads with ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..depth_anything._vendor.dinov2 import DINOv2
from ._vendor.dpt_temporal import DPTHeadTemporal

# Upstream ``run.py`` model_configs and ``VideoDepthAnything.intermediate_layer_idx``.
VIDEO_DEPTH_ANYTHING_CONFIGS: dict[str, dict] = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384], "layers": [2, 5, 8, 11]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768], "layers": [2, 5, 8, 11]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024], "layers": [4, 11, 17, 23]},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Upstream infer settings ("do not change"): the temporal head was trained on
# 32-frame clips, windows overlap by 10 frames, and those 10 slots carry
# keyframes of the previous window.
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
INTERP_LEN = 8


class LibreVideoDepthAnythingNet(nn.Module):
    """``(B, T, 3, H, W)`` frames in ``[0, 1]`` to ``(B, T, H, W)`` depth."""

    def __init__(self, encoder: str = "vits", num_frames: int = INFER_LEN):
        super().__init__()
        if encoder not in VIDEO_DEPTH_ANYTHING_CONFIGS:
            raise ValueError(
                f"Unknown Video Depth Anything encoder {encoder!r}; expected one "
                f"of {sorted(VIDEO_DEPTH_ANYTHING_CONFIGS)}."
            )
        config = VIDEO_DEPTH_ANYTHING_CONFIGS[encoder]
        self.encoder = encoder
        self.intermediate_layer_idx = list(config["layers"])
        self.pretrained = DINOv2(model_name=encoder)
        self.head = DPTHeadTemporal(
            self.pretrained.embed_dim,
            config["features"],
            False,
            out_channels=config["out_channels"],
            use_clstoken=False,
            num_frames=num_frames,
            pe="ape",
        )
        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Video Depth Anything reads a (B, T, 3, H, W) clip; got shape "
                f"{tuple(x.shape)}."
            )
        x = (x - self.pixel_mean) / self.pixel_std
        # Upstream VideoDepthAnything.forward from here on.
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(
            x.flatten(0, 1), self.intermediate_layer_idx, return_class_token=True
        )
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))
