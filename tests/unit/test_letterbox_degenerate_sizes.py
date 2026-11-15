"""Family-local letterbox preprocessing on degenerate image sizes.

Extreme aspect ratios must not truncate a resized side to 0 px (Pillow rejects
that), and empty images must fail with a clear ValueError instead of a
ZeroDivisionError. Mirrors the clamp in ``libreyolo.preprocess.letterbox``.
"""

from __future__ import annotations

import numpy as np
import pytest

from libreyolo.models.darknet.preprocess import preprocess_numpy as darknet_pre
from libreyolo.models.rtmdet.utils import preprocess_numpy as rtmdet_pre
from libreyolo.models.yolo7.utils import preprocess_numpy as yolo7_pre
from libreyolo.preprocess.yolox import preprocess_numpy as yolox_pre

pytestmark = pytest.mark.unit

PREPROCESSORS = pytest.mark.parametrize(
    "preprocess",
    [darknet_pre, rtmdet_pre, yolo7_pre, yolox_pre],
    ids=["darknet", "rtmdet", "yolo7", "yolox"],
)


@PREPROCESSORS
@pytest.mark.parametrize("shape", [(1, 2000, 3), (3000, 2, 3)])
def test_extreme_aspect_ratio(preprocess, shape):
    chw, ratio = preprocess(np.zeros(shape, np.uint8), 640)
    assert chw.shape == (3, 640, 640)
    assert ratio > 0


@PREPROCESSORS
@pytest.mark.parametrize("shape", [(0, 640, 3), (640, 0, 3)])
def test_empty_image_rejected(preprocess, shape):
    with pytest.raises(ValueError, match="empty dimension"):
        preprocess(np.zeros(shape, np.uint8), 640)
