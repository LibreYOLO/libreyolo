"""Inference-side preprocess for LibreYOLO RF-DETR.

Behavior matches upstream RF-DETR (https://github.com/roboflow/rf-detr) so weights
load and produce numerically equivalent detections.

Postprocessing lives in ``libreyolo.postprocess.rfdetr`` and is re-exported
here for backward compatibility.
"""

import numpy as np
from typing import Tuple
from PIL import Image

from ...postprocess.rfdetr import postprocess  # noqa: F401  (backward-compatible re-export)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 560,
) -> Tuple[np.ndarray, float]:
    """
    Preprocess RGB HWC uint8 image for RF-DETR inference.

    Simple resize + ImageNet normalization.

    Args:
        img_rgb_hwc: Input image as RGB HWC uint8 numpy array.
        input_size: Target size for the model.

    Returns:
        Tuple of (preprocessed CHW float32 array with ImageNet norm, ratio).
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1), 1.0
