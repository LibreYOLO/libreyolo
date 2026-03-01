"""
val_128: Validation sanity check for all 15 pretrained models.

Runs model.val() on coco128.yaml (128 images) and checks mAP50-95 >= 0.18.
Purpose: catch catastrophic regressions (broken preprocessing, wrong class
mapping, etc.) — NOT exact mAP benchmarking.

Usage:
    pytest tests/e2e/test_val_coco128.py -v -m e2e
    pytest tests/e2e/test_val_coco128.py::test_val_coco128[yolox-nano] -v
    pytest tests/e2e/test_val_coco128.py -k "rfdetr" -v
"""

import pytest
import torch

from libreyolo import LibreYOLO

MIN_MAP = 0.18  # Uniform threshold for all models

# (weights, size)
MODELS = [
    # YOLOX
    ("LibreYOLOXn.pt",    "nano"),
    ("LibreYOLOXt.pt",    "tiny"),
    ("LibreYOLOXs.pt",    "s"),
    ("LibreYOLOXm.pt",    "m"),
    ("LibreYOLOXl.pt",    "l"),
    ("LibreYOLOXx.pt",    "x"),
    # YOLOv9
    ("LibreYOLO9t.pt",    "t"),
    ("LibreYOLO9s.pt",    "s"),
    ("LibreYOLO9m.pt",    "m"),
    ("LibreYOLO9c.pt",    "c"),
    # RF-DETR
    ("LibreRFDETRn.pth",  "n"),
    ("LibreRFDETRs.pth",  "s"),
    ("LibreRFDETRm.pth",  "m"),
    ("LibreRFDETRl.pth",  "l"),
]

IDS = [
    "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x",
    "v9-t", "v9-s", "v9-m", "v9-c",
    "rfdetr-n", "rfdetr-s", "rfdetr-m", "rfdetr-l",
]


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size", MODELS, ids=IDS)
def test_val_coco128(weights, size):
    """Validate a pretrained model on coco128 and check mAP >= 0.18."""
    model = LibreYOLO(weights, size=size)

    results = model.val(data="coco128.yaml", batch=16, conf=0.001, iou=0.6)

    map50_95 = results["metrics/mAP50-95"]
    map50 = results["metrics/mAP50"]

    print(f"\n  {weights} (size={size}): mAP50-95={map50_95:.4f}, mAP50={map50:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"mAP50-95={map50_95:.4f} below threshold {MIN_MAP} — "
        f"model may be broken (wrong preprocessing, class mapping, etc.)"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
