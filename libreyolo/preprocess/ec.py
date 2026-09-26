"""EC input preprocessing.

Moved verbatim from ``libreyolo/models/ec/postprocess.py``, which re-exports it for backward
compatibility. Lives outside ``models/`` so the ONNX backend can import it
without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..utils.image_size import imgsz_to_hw

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int | Tuple[int, int] = 640
) -> Tuple[np.ndarray, float]:
    """EC preprocess: resize + /255 + ImageNet (mean, std).

    ``input_size`` is a square ``int`` or ``(height, width)``, like upstream
    ``eval_spatial_size``.

    Mirrors upstream val transforms (`Resize -> ConvertPILImage(scale=True) ->
    Normalize(IMAGENET)`). The ImageNet normalization is what distinguishes
    EC's preprocess from D-FINE's; missing it costs ~2 mAP on COCO val.
    """
    height, width = imgsz_to_hw(input_size)
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1), 1.0


__all__ = ["_IMAGENET_MEAN", "_IMAGENET_STD", "preprocess_numpy"]
