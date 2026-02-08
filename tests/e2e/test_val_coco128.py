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

from libreyolo import LIBREYOLO

MIN_MAP = 0.18  # Uniform threshold for all models

# (weights, size)
MODELS = [
    # YOLOX
    ("libreyoloXnano.pt",    "nano"),
    ("libreyoloXtiny.pt",    "tiny"),
    ("libreyoloXs.pt",       "s"),
    ("libreyoloXm.pt",       "m"),
    ("libreyoloXl.pt",       "l"),
    ("libreyoloXx.pt",       "x"),
    # YOLOv9
    ("libreyolo9t.pt",       "t"),
    ("libreyolo9s.pt",       "s"),
    ("libreyolo9m.pt",       "m"),
    ("libreyolo9c.pt",       "c"),
    # RF-DETR
    ("librerfdetrnano.pth",  "n"),
    ("librerfdetrsmall.pth", "s"),
    ("librerfdetrbase.pth",  "b"),
    ("librerfdetrmedium.pth","m"),
    ("librerfdetrlarge.pth", "l"),
]

IDS = [
    "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x",
    "v9-t", "v9-s", "v9-m", "v9-c",
    "rfdetr-n", "rfdetr-s", "rfdetr-b", "rfdetr-m", "rfdetr-l",
]


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size", MODELS, ids=IDS)
def test_val_coco128(weights, size):
    """Validate a pretrained model on coco128 and check mAP >= 0.18."""
    model = LIBREYOLO(weights, size=size)

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
