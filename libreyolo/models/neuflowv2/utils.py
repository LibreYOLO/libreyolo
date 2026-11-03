"""Preprocessing for the NeuFlow v2 family."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: Tuple[int, int] = (432, 768)
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Stretch-resize one RGB frame to the NeuFlow v2 canvas.

    Reproduces upstream ``infer.py``: ``cv2.resize`` with the default bilinear
    filter straight to ``(width, height)``, no aspect preservation and no
    padding. The frame stays RGB (the order upstream trains and evaluates on,
    via PIL); upstream ``infer.py`` feeds OpenCV BGR, which LibreYOLO does not
    copy. Returns a ``(3, H, W)`` float32 array in ``[0, 1]`` and the
    ``(sx, sy)`` factors that map canvas pixels back to original pixels.
    """
    input_h, input_w = int(input_size[0]), int(input_size[1])
    img = np.asarray(img_rgb_hwc)
    orig_h, orig_w = img.shape[:2]
    if (orig_h, orig_w) != (input_h, input_w):
        img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    chw = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw), (orig_w / input_w, orig_h / input_h)
