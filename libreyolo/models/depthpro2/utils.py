#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Preprocessing, normalization, EXIF focal length, and intrinsics helpers for Depth Pro 2."""

from __future__ import annotations

import numbers
from functools import cache

import numpy as np
import torch
from PIL import ExifTags, Image

from .nn import IMAGENET_MEAN, IMAGENET_STD, INPUT_MULTIPLE


@cache
def _normalization(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(3, 1, 1)
    return mean, std


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int | None = None
) -> tuple[np.ndarray, float]:
    """RGB HWC uint8 to a CHW float32 array in ``[0, 1]`` at native resolution.

    Depth Pro 2 operates at the input's native resolution, padding internally
    to multiples of 32 and cropping back. ImageNet mean/std normalization runs
    inside the model forward path so validation tensors from ``DepthDataset``
    (which are already ``[0, 1]`` RGB float32) share the exact same transform.
    """
    del input_size
    image = np.asarray(img_rgb_hwc)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    chw = image.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(chw), 1.0


def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    """Normalize ``[0, 1]`` float RGB tensor ``(3, H, W)`` or ``(B, 3, H, W)`` with ImageNet stats."""
    mean, std = _normalization(x.device)
    x = x.to(torch.float32)
    return (x - mean) / std


def fpx_from_f35(width: float, height: float, f_mm: float) -> float:
    """Convert a 35mm-equivalent focal length in mm to pixels for a given image size."""
    return float(f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2))


def extract_exif_f_px(
    image: Image.Image, width: int, height: int
) -> float | None:
    """Read ``FocalLengthIn35mmFilm`` from PIL EXIF metadata, returning focal length in pixels."""
    try:
        exif = image.getexif()
        if not exif:
            return None
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        f_35mm = exif_ifd.get(ExifTags.Base.FocalLengthIn35mmFilm)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None
    if isinstance(f_35mm, numbers.Real) and float(f_35mm) > 0:
        return fpx_from_f35(width, height, float(f_35mm))
    return None


def pixel_intrinsics(
    f_px: float | torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Build a ``(3, 3)`` OpenCV pixel-space intrinsics matrix for a centered pinhole camera.

    Follows ADR 0006's pixel-center convention: ``cx = (W - 1) / 2``,
    ``cy = (H - 1) / 2``, square pixels ``fx = fy = f_px``.
    """
    focal = float(f_px.detach().cpu().item()) if isinstance(f_px, torch.Tensor) else float(f_px)
    return torch.tensor(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_MULTIPLE",
    "extract_exif_f_px",
    "fpx_from_f35",
    "normalize_rgb",
    "pixel_intrinsics",
    "preprocess_numpy",
]
