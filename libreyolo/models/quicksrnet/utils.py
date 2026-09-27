"""Preprocessing helpers for LibreQuickSRNet."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def preprocess_image(
    image: ImageInput,
    *,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load native-resolution RGB input as a contiguous float tensor in [0, 1]."""

    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, img, original_size, 1.0


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int | tuple[int, int]):
    """Exporter calibration helper: convert RGB HWC to CHW float [0, 1]."""

    del input_size
    array = np.asarray(img_rgb_hwc, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return array.transpose(2, 0, 1).astype(np.float32), 1.0


__all__ = ["preprocess_image", "preprocess_numpy"]
