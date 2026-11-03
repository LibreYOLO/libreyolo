"""Preprocessing helpers for the DPFlow family."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def preprocess_numpy(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """RGB uint8 ``(H, W, 3)`` to float32 CHW in ``[0, 1]`` at native size.

    DPFlow runs at the input resolution: the network pads to its own output
    stride and chooses its pyramid depth from the image diagonal, so no resize
    happens here and the ratio is always 1.0. The ``[-1, 1]`` scaling and the
    padding live inside the vendored network.
    """
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return chw, 1.0
