"""Preprocessing for the LibreBEN2 family."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 1024
) -> Tuple[np.ndarray, float]:
    """Apply BEN2's fixed-square Lanczos resize and ImageNet normalization."""
    arr = np.asarray(img_rgb_hwc)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    pil = Image.fromarray(arr.astype(np.uint8), mode="RGB").resize(
        (input_size, input_size), Image.Resampling.LANCZOS
    )
    chw = np.asarray(pil, dtype=np.float32) / 255.0
    chw = (chw - _MEAN) / _STD
    return np.ascontiguousarray(chw.transpose(2, 0, 1)), 1.0


def preprocess_image(
    image,
    input_size: int = 1024,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load an image and return ``tensor, RGB image, original size, ratio``."""
    from ...utils.image_loader import ImageLoader

    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), input_size)
    return torch.from_numpy(chw).unsqueeze(0), img, original_size, ratio


__all__ = ["preprocess_image", "preprocess_numpy"]
