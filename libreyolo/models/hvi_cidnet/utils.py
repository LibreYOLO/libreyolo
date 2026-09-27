"""Preprocessing helpers for LibreHVI-CIDNet."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if not pad_h and not pad_w:
        return tensor
    mode = (
        "reflect"
        if height > 1 and width > 1 and pad_h < height and pad_w < width
        else "replicate"
    )
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


def preprocess_image(
    image: ImageInput,
    *,
    color_format: str = "auto",
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load RGB in ``[0,1]``, apply gamma, and pad right/bottom to /8."""

    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    loaded = ImageLoader.load(image, color_format=color_format)
    original_size = loaded.size
    array = np.asarray(loaded, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = _pad_to_multiple(tensor.pow(float(gamma)), 8)
    return tensor, loaded, original_size, 1.0


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Exporter calibration helper for the default gamma=1 contract."""

    del input_size
    array = np.asarray(img_rgb_hwc, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return _pad_to_multiple(tensor, 8)[0].numpy().astype(np.float32), 1.0


__all__ = ["preprocess_image", "preprocess_numpy"]
