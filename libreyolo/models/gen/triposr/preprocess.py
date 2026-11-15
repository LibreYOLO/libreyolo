# Adapted from TripoSR (https://github.com/VAST-AI-Research/TripoSR, commit
# 107cefdc244c39106fa830359024f6a2f1c78871: run.py, tsr/utils.py), MIT,
# Copyright (c) 2024 Tripo AI & Stability AI.
"""TripoSR's input: the object cut out, framed square, on mid-grey, at 512 px."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..image3d import has_real_alpha, remove_background


def _to_rgba(image: Image.Image, birefnet) -> Image.Image:
    """Keep a real alpha channel as the cutout; otherwise BiRefNet predicts one."""
    if image.mode in ("LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
    if has_real_alpha(image):
        return image
    return remove_background(birefnet, image.convert("RGB"))


def frame_object(rgba: np.ndarray, margin: float) -> np.ndarray:
    """Upstream's ``resize_foreground``: crop to the alpha box, pad square, pad to ``margin`` x its side."""
    ys, xs = np.where(rgba[..., 3] > 0)
    if len(xs) == 0:
        raise ValueError(
            "No object found in the image: its alpha channel is empty after background removal."
        )
    # Upstream slices [min:max], leaving out the last row and column of the box.
    fg = rgba[ys.min() : ys.max(), xs.min() : xs.max()]
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    square = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)))
    new_size = int(square.shape[0] / (1.0 / margin))
    pad0 = (new_size - size) // 2
    pad1 = new_size - size - pad0
    return np.pad(square, ((pad0, pad1), (pad0, pad1), (0, 0)))


def preprocess_image(image: Image.Image, birefnet, margin: float) -> Image.Image:
    """Return the RGB image the encoder sees, before its resize to 512 px."""
    rgba = np.array(_to_rgba(image, birefnet).convert("RGBA"))
    framed = frame_object(rgba, margin).astype(np.float32) / 255.0
    rgb = framed[:, :, :3] * framed[:, :, 3:4] + (1 - framed[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """RGB PIL image -> [1, 3, size, size] in [0, 1], resized as upstream's ``ImagePreprocessor``."""
    pixels = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(
        pixels, (size, size), mode="bilinear", align_corners=False, antialias=True
    )
