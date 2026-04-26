"""End-to-end validation sanity check for D-FINE on coco128.

Hoare flagged that ``DFINEValPreprocessor`` (inheriting StandardValPreprocessor)
might apply incorrect target rescaling for D-FINE's plain-resize pipeline.
Empirical mAP on coco128 confirms the math is correct.

Skipped if the N checkpoint or coco128 are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_dfine_n_validation_mAP_on_coco128():
    """LibreDFINE-N must produce sane mAP on coco128 (regression for Hoare's concern)."""
    from libreyolo import LibreDFINE

    ckpt = Path("weights/dfine_n_coco.pth")
    if not ckpt.exists():
        pytest.skip(f"{ckpt} not present")

    m = LibreDFINE(str(ckpt), size="n", device="cpu")
    metrics = m.val(data="coco128.yaml", batch=4, conf=0.001, iou=0.6, verbose=False)

    # D-FINE-N reports mAP50-95 = 0.428 on full COCO val2017. coco128 is a
    # small friendly subset so we expect higher numbers. Lower bound 0.45
    # gives a comfortable margin while still failing if a future change
    # silently breaks target rescaling (which would collapse mAP).
    assert metrics["metrics/mAP50-95"] > 0.45, (
        f"mAP50-95 = {metrics['metrics/mAP50-95']:.3f} — too low; suggests "
        "validation preprocessor target rescaling may be broken"
    )
    assert metrics["metrics/mAP50"] > 0.65, (
        f"mAP50 = {metrics['metrics/mAP50']:.3f} — too low"
    )
