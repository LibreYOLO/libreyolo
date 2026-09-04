"""U-Net numpy preprocess: stretch-resize to the native rectangle."""

from __future__ import annotations

import cv2
import numpy as np


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = (1024, 2048),
) -> tuple[np.ndarray, float]:
    """Direct-resize an RGB image to the checkpoint canvas as CHW ``[0, 1]``.

    The mmseg Cityscapes test pipeline feeds whole 1024x2048 frames with no
    padding; ``cv2.INTER_LINEAR`` matches its ``mmcv.imrescale`` kernel, so a
    Cityscapes-aspect frame yields the same tensor upstream sees. ImageNet
    standardization lives inside ``LibreUNetNet``.
    """
    input_h, input_w = _input_size_hw(input_size)
    resized = cv2.resize(img_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32), 1.0


__all__ = ["_input_size_hw", "preprocess_numpy"]
