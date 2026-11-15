"""Preprocessing helpers for the UniMatch (GMFlow) family."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def inference_size(height: int, width: int, padding_factor: int) -> Tuple[int, int]:
    """Return upstream's ``nearest_size``: each side rounded up to the factor."""
    return (
        int(math.ceil(height / padding_factor)) * padding_factor,
        int(math.ceil(width / padding_factor)) * padding_factor,
    )


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 640
) -> Tuple[np.ndarray, float]:
    """Stretch-resize one RGB frame to a square ``input_size`` canvas.

    Used by fixed-canvas consumers (export, validation). Returns a ``(3, H, W)``
    float32 array in ``[0, 255]``; ImageNet normalization happens inside the
    network, as upstream. The ratio is ``1.0`` because flow vectors are rescaled
    per axis in postprocess, not by a single factor.
    """
    image = np.asarray(img_rgb_hwc)[..., :3]
    resized = cv2.resize(
        image, (input_size, input_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)), 1.0
