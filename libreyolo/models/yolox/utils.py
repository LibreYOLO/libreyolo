"""
Utility functions for YOLOX.

YOLOX uses different preprocessing and postprocessing:
- Preprocessing: Letterbox with gray padding (114,114,114), NO normalization (0-255 range)
- Postprocessing: Box decoding with exp() for width/height, objectness score

Postprocessing lives in ``libreyolo.postprocess.yolox`` and is re-exported
here for backward compatibility.
"""

import torch
import numpy as np
from typing import Tuple
from PIL import Image

from ...postprocess.yolox import (  # noqa: F401  (backward-compatible re-exports)
    decode_outputs,
    make_grids,
    postprocess,
)
from ...utils.image_loader import ImageLoader, ImageInput


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
    ratio = min(input_size / orig_h, input_size / orig_w)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    # Letterbox with gray padding at top-left
    padded = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    padded.paste(img_resized, (0, 0))

    # RGB to BGR, HWC to CHW, keep 0-255
    arr = np.array(padded, dtype=np.float32)[:, :, ::-1].copy()
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
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
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio
