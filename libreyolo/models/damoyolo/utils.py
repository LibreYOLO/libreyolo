"""DAMO-YOLO preprocessing + postprocessing helpers.

Preprocessing matches upstream's inference pipeline
(``damo/utils/demo_utils.py::transform_img`` + ``test_transform`` defaults):

- RGB, no normalization, float32 in [0, 255]
- resize to (640, 640) via ``cv2.INTER_LINEAR``, *no* keep-ratio (image stretched)

Postprocessing lives in ``libreyolo.postprocess.damoyolo`` and is re-exported
here for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from ...postprocess.damoyolo import (  # noqa: F401  (backward-compatible re-exports)
    multiclass_nms,
    postprocess_predictions,
)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 640
) -> Tuple[np.ndarray, float]:
    """Stretch-resize an RGB HWC uint8 array to ``(input_size, input_size)``.

    Returns ``(chw_float32_in_0_255, ratio)``. Since DAMO-YOLO uses non-keep-ratio
    resize, ``ratio`` is reported as ``1.0`` (caller scales x and y separately
    via the original_size hint).
    """
    img = cv2.resize(img_rgb_hwc, (input_size, input_size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    return chw, 1.0


def preprocess_image(
    image_path: str | Path,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess one image.

    Returns ``(tensor, (orig_w, orig_h))``. ``tensor`` is float32 (3, H, W),
    range [0, 255], RGB.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    orig_h, orig_w = img.shape[:2]
    # cv2 reads BGR; upstream loads via PIL.convert("RGB") so we mirror that.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_t, w_t = input_size
    img = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    return tensor, (orig_w, orig_h)
