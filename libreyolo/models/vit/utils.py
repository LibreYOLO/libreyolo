"""Preprocessing for LibreViT AugReg image classifiers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...data.classify_dataset import build_classify_transforms
from ...utils.image_loader import ImageInput, ImageLoader

VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)


def build_eval_transform(input_size: int, crop_pct: float = 0.9):
    """Return the exact timm AugReg evaluation transform."""
    return build_classify_transforms(
        imgsz=input_size,
        augment=False,
        mean=VIT_MEAN,
        std=VIT_STD,
        crop_pct=crop_pct,
        interpolation="bicubic",
    )


def preprocess_image(
    image: ImageInput,
    input_size: int,
    crop_pct: float = 0.9,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load one image and return the shared classification predict tuple."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(
    img_rgb_hwc, input_size: int, crop_pct: float = 0.9
) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC array (or PIL image) to normalized CHW array.

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        img_rgb_hwc
        if isinstance(img_rgb_hwc, Image.Image)
        else Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
    )
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0
