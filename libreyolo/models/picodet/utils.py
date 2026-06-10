"""PICODET preprocessing and postprocessing.

PICODET upstream uses **non-letterbox** simple resize + ImageNet
normalisation (RGB, mean=[123.675, 116.28, 103.53],
std=[58.395, 57.12, 57.375]).

Postprocessing (GFL/DFL decode + NMS) lives in
``libreyolo.postprocess.picodet`` and is re-exported here for backward
compatibility.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...postprocess.picodet import (  # noqa: F401  (backward-compatible re-exports)
    _grid_centers,
    _per_level_filter_topk,
    postprocess,
)
from ...utils.image_loader import ImageInput, ImageLoader


# ImageNet stats Bo's repo uses (shared across all PICODET sizes)
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 320,
) -> Tuple[np.ndarray, float]:
    """Preprocess an RGB HWC uint8 image for PICODET inference.

    Returns ``(chw_float32, ratio)``. ``ratio`` is unused by PICODET's
    non-letterbox resize but kept in the signature so it can flow through
    the same postprocess pipeline as letterbox-based families.
    """
    # Upstream PaddleDetection / Bo's port resize with cv2.INTER_LINEAR.
    # PIL's bilinear kernel differs and drifts ~0.3-0.5 mAP on COCO, so match cv2.
    arr = cv2.resize(
        img_rgb_hwc, (input_size, input_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    arr -= np.array(IMAGENET_MEAN, dtype=np.float32)
    arr /= np.array(IMAGENET_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 320,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio
