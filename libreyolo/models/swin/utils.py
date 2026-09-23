"""ImageNet evaluation preprocessing for LibreSwin classifiers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...data.classify_dataset import build_classify_transforms
from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(input_size: int, crop_pct: float = 0.9):
    """Build the released Swin V1 resize, center-crop, and normalization path."""
    return build_classify_transforms(
        imgsz=input_size,
        augment=False,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        crop_pct=crop_pct,
        interpolation="bicubic",
    )


def preprocess_image(
    image: ImageInput,
    input_size: int,
    crop_pct: float = 0.9,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load one image and return the standard classification predict tuple."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(
    image_rgb_hwc, input_size: int, crop_pct: float = 0.9
) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC array or PIL image into normalized CHW array.

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        image_rgb_hwc
        if isinstance(image_rgb_hwc, Image.Image)
        else Image.fromarray(np.asarray(image_rgb_hwc).astype("uint8"))
    )
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_eval_transform",
    "preprocess_image",
    "preprocess_numpy",
]
