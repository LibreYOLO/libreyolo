"""RiO-DETRv2 numpy preprocessing (export calibration and other numpy callers)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size) -> tuple[np.ndarray, float]:
    """Same transform as ``predict``: keep-ratio bilinear resize, zero pad right/bottom.

    Accepts an int or ``(height, width)`` size and returns one ``(3, H, W)``
    float32 image in ``[0, 1]`` plus the resize ratio.
    """
    if isinstance(input_size, int):
        target_h = target_w = input_size
    else:
        target_h, target_w = int(input_size[0]), int(input_size[1])
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, round(orig_w * ratio))
    new_h = max(1, round(orig_h * ratio))
    resized = Image.fromarray(np.ascontiguousarray(img_rgb_hwc, dtype=np.uint8)).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = np.asarray(resized)
    chw = np.ascontiguousarray(canvas.astype(np.float32).transpose(2, 0, 1) / 255.0)
    return chw, ratio
