"""YOLOX input preprocessing.

Moved verbatim from ``libreyolo/models/yolox/utils.py``, which re-exports
everything here for backward compatibility. Lives outside ``models/`` so the
ONNX backend can import it without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..utils.image_loader import ImageInput, ImageLoader
from . import as_batched_input


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for YOLOX inference.

    YOLOX-specific: letterbox + RGB to BGR + no normalization (0-255 range).

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of (preprocessed CHW float32 array in BGR 0-255, ratio).
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    if orig_h == 0 or orig_w == 0:
        raise ValueError(f"Image has an empty dimension: {orig_w}x{orig_h} (WxH).")
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
    else:
        input_h = input_w = int(input_size)
    ratio = min(input_w / orig_w, input_h / orig_h)
    # Clamp so extreme aspect ratios (e.g. 1x2000) don't truncate to a 0-px side.
    new_w, new_h = max(1, int(orig_w * ratio)), max(1, int(orig_h * ratio))

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    # Letterbox with gray padding at top-left
    padded = Image.new("RGB", (input_w, input_h), (114, 114, 114))
    padded.paste(img_resized, (0, 0))

    # RGB to BGR, HWC to CHW, keep 0-255
    arr = np.array(padded, dtype=np.float32)[:, :, ::-1].copy()
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
):
    """
    Preprocess image for YOLOX inference with letterboxing.

    YOLOX-specific preprocessing:
    - Letterbox resize maintaining aspect ratio
    - Gray padding (114, 114, 114)
    - NO normalization (keeps 0-255 range as float32)

    Args:
        image: Input image (path, PIL, numpy, tensor, bytes, etc.)
        input_size: Target size for the model (default: 640)
        color_format: Color format hint ("auto", "rgb", "bgr")

    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size, ratio)
    """
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size  # (width, height)
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(np.array(img), input_size)
    return as_batched_input(img_chw), original_img, original_size, ratio


__all__ = ["preprocess_numpy", "preprocess_image"]
