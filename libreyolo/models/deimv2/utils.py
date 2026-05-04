"""LibreDEIMv2 preprocessing and postprocessing helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..deim.utils import postprocess, unwrap_deim_checkpoint

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "postprocess",
    "preprocess_image",
    "preprocess_numpy",
    "unwrap_deim_checkpoint",
]


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
    *,
    imagenet_norm: bool = False,
) -> Tuple[np.ndarray, float]:
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    chw = (np.array(img_resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    if imagenet_norm:
        chw = (chw - IMAGENET_MEAN) / IMAGENET_STD
    return chw.astype(np.float32), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
    *,
    imagenet_norm: bool = False,
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(
        np.array(img), input_size=input_size, imagenet_norm=imagenet_norm
    )
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio
