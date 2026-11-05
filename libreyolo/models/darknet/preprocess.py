"""Preprocessing for the Darknet families (YOLOv2/v3/v4).

Darknet detectors take an RGB image normalized to ``[0, 1]``, aspect-preserving
letterbox-resized onto a square canvas. We use top-left padding (like YOLOX /
YOLO9 in LibreYOLO) so the inverse transform is a single divide-by-ratio in
post-processing. The pad fill is ~0.5 (128/255), matching Darknet's
``letterbox_image`` gray fill.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

# 128/255 ≈ 0.502, the nearest uint8 to Darknet's 0.5 letterbox fill.
_PAD_VALUE = 128


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 416,
) -> Tuple[np.ndarray, float]:
    """Letterbox an RGB HWC uint8 image to ``input_size`` and normalize to [0,1].

    Returns ``(CHW float32 RGB in [0,1], ratio)`` where ``ratio =
    min(input/orig_h, input/orig_w)``.
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError(f"Image has an empty dimension: {orig_w}x{orig_h} (WxH).")
    ratio = min(input_size / orig_h, input_size / orig_w)
    # Clamp so extreme aspect ratios (e.g. 1x2000) don't truncate to a 0-px side.
    new_w, new_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (input_size, input_size), (_PAD_VALUE,) * 3)
    padded.paste(img_resized, (0, 0))

    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput, input_size: int = 416, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Preprocess an arbitrary image input for Darknet inference.

    Returns ``(input_tensor[1,3,H,W], original_image, (width, height), ratio)``.
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    chw, ratio = preprocess_numpy(np.array(img), input_size)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor, original_img, original_size, ratio


def preprocess_numpy_stretch(
    img_rgb_hwc: np.ndarray,
    input_size: int = 448,
) -> Tuple[np.ndarray, float]:
    """Stretch (non-aspect-preserving) resize to ``input_size`` and normalize.

    YOLOv1's fully-connected head is not translation-invariant and was trained on
    a plain square resize (Darknet's ``resize_image``, as used by the classic
    ``yolo.c`` test path), NOT the aspect-preserving ``letterbox_image`` the
    convolutional v2/v3/v4 detectors use. Letterboxing shifts the image into the
    top-left of the canvas and the FC head then sees a spatial layout it never
    trained on, wrecking accuracy. So YOLOv1 stretches the whole image onto the
    square canvas. The returned ratio is ``1.0`` (unused; the inverse transform
    is independent x/y scaling done in post-processing, ``letterbox=False``).
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image_stretch(
    image: ImageInput, input_size: int = 448, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Preprocess an image for YOLOv1 inference (square stretch resize)."""
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    chw, ratio = preprocess_numpy_stretch(np.array(img), input_size)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor, original_img, original_size, ratio
