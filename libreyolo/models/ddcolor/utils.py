"""Checkpoint-faithful OpenCV preprocessing for LibreDDColor.

Adapted from ``piddnad/DDColor/ddcolor/pipeline.py`` at commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13`` under Apache-2.0. LibreYOLO
modifies the pipeline to accept its public RGB image inputs, preserve the
original-resolution Lab lightness plane as per-request context, and separate
preprocessing from result construction. See the family ``NOTICE``.
"""

from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


DDCOLOR_ORIGINAL_L_KEY = "ddcolor_original_l"
_OFFICIAL_INPUT_SIZE = 512


def _gray_rgb_and_l(
    image_rgb_uint8: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official BGR -> Lab -> neutral-L RGB transform."""

    if input_size != _OFFICIAL_INPUT_SIZE:
        raise ValueError(
            "DDColor checkpoints use the official fixed 512x512 pipeline; "
            f"got input_size={input_size}."
        )
    if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
        raise ValueError(
            "DDColor preprocessing expects an RGB HWC image with three channels."
        )

    # The upstream public contract starts from OpenCV BGR uint8. ImageLoader
    # gives LibreYOLO one canonical RGB image, so reverse channels first and
    # keep every subsequent operation in the original order and dtype.
    image_bgr = np.ascontiguousarray(image_rgb_uint8[..., ::-1])
    image_bgr_float = (image_bgr / 255.0).astype(np.float32)
    original_l = cv2.cvtColor(image_bgr_float, cv2.COLOR_BGR2Lab)[:, :, :1]

    # cv2.resize defaults to INTER_LINEAR, exactly as the pinned pipeline.
    resized_bgr = cv2.resize(image_bgr_float, (input_size, input_size))
    resized_l = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2Lab)[:, :, :1]
    neutral_lab = np.concatenate(
        (resized_l, np.zeros_like(resized_l), np.zeros_like(resized_l)),
        axis=-1,
    )
    neutral_rgb = cv2.cvtColor(neutral_lab, cv2.COLOR_LAB2RGB)
    return np.ascontiguousarray(neutral_rgb), np.ascontiguousarray(original_l)


def preprocess_image(
    image: ImageInput,
    *,
    input_size: int = _OFFICIAL_INPUT_SIZE,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], dict[str, Any]]:
    """Return official 512-square input plus original-resolution Lab metadata."""

    image_pil = ImageLoader.load(image, color_format=color_format)
    original_size = image_pil.size
    image_rgb = np.asarray(image_pil, dtype=np.uint8)
    neutral_rgb, original_l = _gray_rgb_and_l(image_rgb, int(input_size))
    tensor = (
        torch.from_numpy(neutral_rgb.transpose(2, 0, 1))
        .float()
        .unsqueeze(0)
        .contiguous()
    )
    metadata = {DDCOLOR_ORIGINAL_L_KEY: original_l}
    return tensor, image_pil, original_size, metadata


def preprocess_numpy(
    image_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Produce the checkpoint-faithful network tensor for calibration tools.

    The exported tensor network still emits Lab ``ab``. Full RGB restoration
    also requires the original ``L`` plane and therefore remains a wrapper
    operation rather than an in-graph approximation.
    """

    if isinstance(input_size, tuple):
        if len(input_size) != 2 or input_size[0] != input_size[1]:
            raise ValueError(f"DDColor input_size must be square, got {input_size!r}.")
        input_size = int(input_size[0])
    array = np.asarray(image_rgb_hwc)
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.clip(0.0, 255.0).round().astype(np.uint8)
    neutral_rgb, _ = _gray_rgb_and_l(array, int(input_size))
    return neutral_rgb.transpose(2, 0, 1).astype(np.float32), 1.0


__all__ = [
    "DDCOLOR_ORIGINAL_L_KEY",
    "preprocess_image",
    "preprocess_numpy",
]
