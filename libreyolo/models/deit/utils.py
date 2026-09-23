"""ImageNet evaluation preprocessing for LibreDeiT."""

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


def build_eval_transform(input_size: int, crop_pct: float) -> transforms.Compose:
    """Build the timm DeiT eval transform for a fixed square input."""
    scale_size = int(math.floor(input_size / crop_pct))
    return transforms.Compose(
        [
            transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(
    image: ImageInput,
    input_size: int,
    crop_pct: float,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load one image and return the standard LibreYOLO preprocess tuple."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(
    img_rgb_hwc, input_size: int, crop_pct: float = 0.9
) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC image to a normalized DeiT CHW array.

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
        if not isinstance(img_rgb_hwc, Image.Image)
        else img_rgb_hwc
    )
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0
