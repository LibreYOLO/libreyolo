"""Preprocessing for LibreResNet classification.

Mirrors timm's ImageNet eval transform (bicubic shorter-side resize ->
center-crop -> ImageNet normalize). The ``resnet*.a1_in1k`` checkpoints use
crop_pct 0.95 at 224; 288/1.0 is the (optional) higher-res eval.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ...data.augment.classify import build_classify_transforms
from ...utils.image_loader import ImageInput, ImageLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(input_size: int, crop_pct: float) -> transforms.Compose:
    """timm-style eval transform: bicubic shorter-side resize -> center crop -> normalize.

    The shared classification eval pipeline (``data/augment/classify.py``),
    so ``predict()`` and ``val()`` run the same code (#886).
    """
    return build_classify_transforms(
        input_size, augment=False, crop_pct=crop_pct, interpolation="bicubic"
    )


def preprocess_image(
    image: ImageInput,
    input_size: int,
    crop_pct: float,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load and preprocess one image. Returns (tensor[1,3,H,W], pil, (w,h), ratio=1.0)."""
    pil = ImageLoader.load(image, color_format=color_format)
    orig_w, orig_h = pil.size
    tensor = build_eval_transform(input_size, crop_pct)(pil).unsqueeze(0)
    return tensor, pil, (orig_w, orig_h), 1.0


def preprocess_numpy(img_rgb_hwc, input_size: int, crop_pct: float = 0.95) -> Tuple[np.ndarray, float]:
    """Secondary numpy entry point (RGB HWC -> normalized CHW array).

    Returns ``(CHW float32 array, 1.0)``, the INT8 calibration contract.
    """
    pil = (
        img_rgb_hwc
        if isinstance(img_rgb_hwc, Image.Image)
        else Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
    )
    return build_eval_transform(input_size, crop_pct)(pil).numpy(), 1.0
