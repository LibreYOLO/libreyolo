"""Upstream-equivalent preprocessing helpers for RetinaNet inference."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ...postprocess.retinanet import resize_geometry
from ...utils.image_loader import ImageInput, ImageLoader


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
MAX_SIZE_RATIO = 1333 / 800
SIZE_DIVISIBLE = 32


def resize_scale(
    original_size: Tuple[int, int], input_size: int, fixed_canvas: bool = False
) -> float:
    """Return the upstream continuous aspect-preserving resize factor.

    ``fixed_canvas`` fits the whole image into an ``input_size`` square instead
    (longer side = ``input_size``), for fixed-shape exports.
    """
    original_width, original_height = original_size
    if fixed_canvas:
        return min(input_size / original_height, input_size / original_width)
    max_size = round(input_size * MAX_SIZE_RATIO)
    return min(
        input_size / min(original_height, original_width),
        max_size / max(original_height, original_width),
    )


def preprocess_tensor(
    image: torch.Tensor, input_size: int = 800, fixed_canvas: bool = False
) -> torch.Tensor:
    """Normalize, aspect-resize, and bottom/right-pad one RGB CHW tensor.

    ``fixed_canvas`` fits the image into an ``input_size`` square and pads to
    exactly that square (fixed-shape exports such as Axelera).

    ``image`` must be floating point in the [0, 1] range. The interpolation
    call deliberately uses ``scale_factor`` plus ``recompute_scale_factor`` to
    preserve torchvision's exact rounding and sampling behavior.
    """
    if image.ndim != 3:
        raise ValueError(
            f"RetinaNet expects a CHW image tensor, got {tuple(image.shape)}"
        )
    if not image.is_floating_point():
        raise TypeError("RetinaNet preprocessing expects floating-point pixels")
    mean = torch.as_tensor(IMAGE_MEAN, dtype=image.dtype, device=image.device)
    std = torch.as_tensor(IMAGE_STD, dtype=image.dtype, device=image.device)
    normalized = (image - mean[:, None, None]) / std[:, None, None]
    scale = resize_scale(
        (int(image.shape[2]), int(image.shape[1])), input_size, fixed_canvas
    )
    resized = F.interpolate(
        normalized.unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]
    height, width = resized.shape[-2:]
    if fixed_canvas:
        padded_height = padded_width = int(input_size)
    else:
        padded_height = math.ceil(height / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
        padded_width = math.ceil(width / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    return F.pad(resized, (0, padded_width - width, 0, padded_height - height))


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 800, fixed_canvas: bool = False
) -> Tuple[np.ndarray, float]:
    """Return normalized padded CHW float32 pixels and the resize factor."""
    array = np.asarray(img_rgb_hwc)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"RetinaNet expects an RGB HWC image, got {tuple(array.shape)}"
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    if isinstance(input_size, (tuple, list)):  # (h, w) from export calibration
        height, width = (int(v) for v in input_size)
        if height != width:
            raise ValueError(f"RetinaNet needs a square size, got {height}x{width}.")
        input_size = height
    tensor = tensor.to(torch.float32) / 255.0
    processed = preprocess_tensor(tensor, input_size=input_size, fixed_canvas=fixed_canvas)
    ratio = resize_scale((array.shape[1], array.shape[0]), input_size, fixed_canvas)
    return np.ascontiguousarray(processed.numpy()), ratio


def preprocess_image(
    image: ImageInput,
    input_size: int = 800,
    color_format: str = "auto",
    fixed_canvas: bool = False,
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and apply RetinaNet's upstream inference transform once."""
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    image_chw, ratio = preprocess_numpy(np.asarray(loaded), input_size, fixed_canvas)
    return torch.from_numpy(image_chw).unsqueeze(0), loaded, original_size, ratio


def resized_shape(
    original_size: Tuple[int, int], input_size: int = 800
) -> tuple[int, int]:
    """Expose the unpadded upstream H/W for validation diagnostics."""
    height, width, _, _ = resize_geometry(original_size, input_size)
    return height, width


__all__ = [
    "IMAGE_MEAN",
    "IMAGE_STD",
    "preprocess_image",
    "preprocess_numpy",
    "preprocess_tensor",
    "resize_scale",
    "resized_shape",
]
