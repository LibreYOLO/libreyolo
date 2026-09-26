"""RF-DETR input preprocessing.

Moved verbatim from ``libreyolo/models/rfdetr/utils.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | Tuple[int, int] = 560,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for RF-DETR inference.

    Simple resize + ImageNet normalization. The resize is bilinear without
    antialiasing, matching the cv2 resize used in training and upstream
    RF-DETR's ``predict()`` since v1.9.0 (roboflow/rf-detr#1206). Antialiasing
    drifts boxes and scores on inputs much larger than the model canvas.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model, square ``int`` or
            ``(height, width)``. Upstream ``predict(shape=(h, w))`` resizes
            the same way; boxes are normalized, so no ratio is needed.

    Returns:
        Tuple of (preprocessed CHW float32 array with ImageNet norm, ratio).
    """
    import cv2

    arr = np.asarray(img_rgb_hwc, dtype=np.float32) / 255.0
    if isinstance(input_size, (list, tuple)):
        height, width = int(input_size[0]), int(input_size[1])
    else:
        height = width = int(input_size)
    arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1), 1.0


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "preprocess_numpy"]
