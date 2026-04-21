"""
val_128: Validation sanity check for all catalog pretrained models.

Runs model.val() on coco128.yaml (128 images) and checks mAP50-95 >= 0.18.
Purpose: catch catastrophic regressions (broken preprocessing, wrong class
mapping, etc.) — NOT exact mAP benchmarking.

Usage:
    pytest tests/e2e/test_val_coco128.py -v -m e2e
    pytest tests/e2e/test_val_coco128.py::test_val_coco128[yolox-n] -v
    pytest tests/e2e/test_val_coco128.py -k "rfdetr" -v
"""

import pytest

from libreyolo import LibreYOLO
from .conftest import (
    ALL_MODELS_WITH_WEIGHTS,
    cuda_cleanup,
    make_ids,
    require_test_weights,
)

pytestmark = pytest.mark.e2e

MIN_MAP = 0.18  # Uniform threshold for all models


@pytest.mark.parametrize(
    "family,size,weights",
    ALL_MODELS_WITH_WEIGHTS,
    ids=make_ids(ALL_MODELS_WITH_WEIGHTS),
)
def test_val_coco128(family, size, weights):
    """Validate a pretrained model on coco128 and check mAP >= 0.18."""
    weights = require_test_weights(weights)
    model = LibreYOLO(weights, size=size)

    results = model.val(data="coco128.yaml", batch=16, conf=0.001, iou=0.6)

    map50_95 = results["metrics/mAP50-95"]
    map50 = results["metrics/mAP50"]

    print(f"\n  {weights} (size={size}): mAP50-95={map50_95:.4f}, mAP50={map50:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"mAP50-95={map50_95:.4f} below threshold {MIN_MAP} — "
        f"model may be broken (wrong preprocessing, class mapping, etc.)"
    )

    cuda_cleanup()
