"""ImageNet evaluation preprocessing for LibreAlexNet."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(
    input_size: int,
    crop_pct: float = 0.875,
) -> transforms.Compose:
    """Resize the shorter side, center-crop, and apply ImageNet normalization."""
    resize_size = int(math.floor(input_size / crop_pct))
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(
    image: ImageInput,
    input_size: int,
    crop_pct: float = 0.875,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Return a normalized batch tensor plus the unchanged source image metadata."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(
    img_rgb_hwc,
    input_size: int,
    crop_pct: float = 0.875,
) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC array into one normalized CHW array.

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        img_rgb_hwc
        if isinstance(img_rgb_hwc, Image.Image)
        else Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
    )
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_eval_transform",
    "preprocess_image",
    "preprocess_numpy",
]
