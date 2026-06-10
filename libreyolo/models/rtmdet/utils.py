"""
RTMDet preprocess and postprocess.

Preprocess: BGR letterbox at pad value 114 with mmdet-style normalization
    mean = [103.53, 116.28, 123.675] (BGR order; same numerical values as ImageNet RGB
    reversed because the input stays BGR per ``bgr_to_rgb=False`` in the config)
    std  = [57.375, 57.12, 58.395]

Postprocessing lives in ``libreyolo.postprocess.rtmdet`` and is re-exported
here for backward compatibility.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.rtmdet import (  # noqa: F401  (backward-compatible re-exports)
    _distance2bbox,
    _make_grid_priors,
    postprocess,
)
from ...utils.image_loader import ImageInput, ImageLoader


# mmdet/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py:
#     mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], bgr_to_rgb=False
_RTMDET_BGR_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
_RTMDET_BGR_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
_RTMDET_PAD_VALUE = 114


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
) -> Tuple[np.ndarray, float]:
    """Letterbox to (input_size, input_size), convert RGB -> BGR, mean/std normalize.

    Args:
        img_rgb_hwc: HWC uint8 RGB image.
        input_size: square target side.

    Returns:
        (CHW float32 normalized, scale_ratio).
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(input_size / orig_h, input_size / orig_w)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (input_size, input_size), (_RTMDET_PAD_VALUE,) * 3)
    padded.paste(img_resized, (0, 0))

    # RGB -> BGR; mean/std applied in BGR space (mmdet config has bgr_to_rgb=False).
    arr = np.array(padded, dtype=np.float32)[:, :, ::-1]
    arr = (arr - _RTMDET_BGR_MEAN) / _RTMDET_BGR_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)), ratio


def preprocess_image(
    image: ImageInput, input_size: int = 640, color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Load image and produce a CHW tensor + metadata for inference."""
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio
