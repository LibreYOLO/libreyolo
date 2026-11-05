"""PP-DocLayoutV3 input preprocessing (torch-free).

The released model resizes the page to a fixed square canvas without keeping
the aspect ratio (cv2 bicubic, as the upstream Paddle pipeline) and feeds RGB
in ``[0, 1]`` with no mean/std normalization.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 800
) -> Tuple[np.ndarray, float]:
    """Stretch-resize an RGB uint8 image to the square canvas, scale to [0, 1]."""
    import cv2

    resized = cv2.resize(
        np.ascontiguousarray(img_rgb_hwc),
        (input_size, input_size),
        interpolation=cv2.INTER_CUBIC,
    )
    array = resized.astype(np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)), 1.0
