"""DEIM input preprocessing.

Moved verbatim from ``libreyolo/models/deim/utils.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..utils.image_size import imgsz_to_hw

def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | Tuple[int, int] = 640,
) -> Tuple[np.ndarray, float]:
    """Preprocess an RGB HWC uint8 array to DEIM input layout.

    Plain resize to ``input_size`` (square ``int`` or ``(height, width)``,
    like upstream ``eval_spatial_size``), no letterbox, no
    ImageNet normalization — just ``uint8 / 255``. Ratio is always 1.0
    because there's no padding.
    """
    height, width = imgsz_to_hw(input_size)
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


__all__ = ["preprocess_numpy"]
