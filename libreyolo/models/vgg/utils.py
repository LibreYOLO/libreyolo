"""ImageNet evaluation preprocessing for LibreVGG."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SIZE = 256


def build_eval_transform(input_size: int = 224) -> transforms.Compose:
    """Resize the short side to 256, center-crop, and ImageNet-normalize."""
    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SIZE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def preprocess_image(
    image: ImageInput,
    input_size: int = 224,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image for VGG classification."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(img_rgb_hwc, input_size: int = 224) -> Tuple[np.ndarray, float]:
    """Convert an RGB HWC array to a normalized CHW array.

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        img_rgb_hwc
        if isinstance(img_rgb_hwc, Image.Image)
        else Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
    )
    return build_eval_transform(input_size)(pil).numpy(), 1.0


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RESIZE_SIZE",
    "build_eval_transform",
    "preprocess_image",
    "preprocess_numpy",
]
