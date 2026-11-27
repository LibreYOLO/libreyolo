"""Exact torchvision-compatible preprocessing for FCOS inference.

Transform semantics are derived from pytorch/vision v0.26.0 at commit
336d36e8db990a905498c73933e35231876e28bc (BSD-3-Clause).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_MIN_SIZE = 800
DEFAULT_MAX_SIZE = 1333
SIZE_DIVISIBLE = 32


def max_size_for(input_size: int) -> int:
    """Scale torchvision's 800/1333 resize contract for an explicit override."""
    input_size = int(input_size)
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    return int(round(input_size * DEFAULT_MAX_SIZE / DEFAULT_MIN_SIZE))


def resize_dimensions(
    height: int,
    width: int,
    input_size: int = DEFAULT_MIN_SIZE,
    fixed_canvas: bool = False,
) -> tuple[int, int, float]:
    """Return the unpadded FCOS resize dimensions and nominal uniform scale.

    ``fixed_canvas`` fits the whole image into an ``input_size`` square instead
    (longer side = ``input_size``), for fixed-shape exports such as Axelera.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"image dimensions must be positive, got {(height, width)}")
    if fixed_canvas:
        scale = min(float(input_size) / height, float(input_size) / width)
    else:
        max_size = max_size_for(input_size)
        scale = min(float(input_size) / min(height, width), max_size / max(height, width))
    # F.interpolate with scale_factor and recompute_scale_factor=True floors
    # each spatial dimension independently.
    return max(1, int(height * scale)), max(1, int(width * scale)), scale


def _preprocess_tensor(
    img_rgb_hwc: np.ndarray,
    input_size: int = DEFAULT_MIN_SIZE,
    fixed_canvas: bool = False,
) -> tuple[torch.Tensor, float]:
    image = np.asarray(img_rgb_hwc)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"FCOS expects an RGB HWC image, got {image.shape}")

    tensor = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32).div(255.0)
    mean = tensor.new_tensor(IMAGE_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGE_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    height, width = tensor.shape[-2:]
    _, _, scale = resize_dimensions(height, width, input_size, fixed_canvas)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]

    resized_h, resized_w = tensor.shape[-2:]
    if fixed_canvas:
        padded_h = padded_w = int(input_size)
    else:
        padded_h = int(math.ceil(resized_h / SIZE_DIVISIBLE) * SIZE_DIVISIBLE)
        padded_w = int(math.ceil(resized_w / SIZE_DIVISIBLE) * SIZE_DIVISIBLE)
    tensor = F.pad(tensor, (0, padded_w - resized_w, 0, padded_h - resized_h))
    return tensor, scale


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = DEFAULT_MIN_SIZE,
    fixed_canvas: bool = False,
) -> Tuple[np.ndarray, float]:
    """Normalize, aspect-resize, and bottom/right-pad an RGB uint8 image.

    ``fixed_canvas`` fits the image into one ``input_size`` square and pads to
    exactly that square (fixed-shape exports such as Axelera).
    """
    if isinstance(input_size, (tuple, list)):  # (h, w) from export calibration
        height, width = (int(v) for v in input_size)
        if height != width:
            raise ValueError(f"FCOS needs a square size, got {height}x{width}.")
        input_size = height
    tensor, ratio = _preprocess_tensor(img_rgb_hwc, input_size, fixed_canvas)
    return np.ascontiguousarray(tensor.numpy(), dtype=np.float32), ratio


def preprocess_image(
    image: ImageInput,
    color_format: str = "auto",
    input_size: int = DEFAULT_MIN_SIZE,
    fixed_canvas: bool = False,
) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
    """Load and apply torchvision's FCOS inference transform to one image."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    tensor, ratio = _preprocess_tensor(np.asarray(loaded), input_size, fixed_canvas)
    return tensor.unsqueeze(0), loaded, original_size, ratio


__all__ = [
    "DEFAULT_MAX_SIZE",
    "DEFAULT_MIN_SIZE",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "SIZE_DIVISIBLE",
    "max_size_for",
    "preprocess_image",
    "preprocess_numpy",
    "resize_dimensions",
]
