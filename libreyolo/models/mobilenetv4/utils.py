"""Preprocessing for LibreMobileNetV4 classification.

Mirrors timm's ImageNet eval transform (bicubic shorter-side resize ->
center-crop -> ImageNet normalize) so accuracy matches the upstream benchmark.
``crop_pct`` is per-variant (s=0.875, m/l=0.95).
"""

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
    """timm-style eval transform: bicubic shorter-side resize -> center crop -> normalize."""
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
    """Load and preprocess a single image for classification.

    Returns ``(input_tensor[1,3,H,W], original_pil, (orig_w, orig_h), ratio)``.
    ``ratio`` is 1.0 (classification has no box geometry to rescale).
    """
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(img_rgb_hwc, input_size: int, crop_pct: float = 0.875) -> Tuple[np.ndarray, float]:
    """Secondary numpy entry point (RGB HWC -> normalized CHW array).

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.

    The primary predict path uses :meth:`LibreMobileNetV4._preprocess`, which
    supplies the exact per-variant ``crop_pct``; this fallback defaults to 0.875.
    """
    pil = Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8")) if not isinstance(
        img_rgb_hwc, Image.Image
    ) else img_rgb_hwc
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0
