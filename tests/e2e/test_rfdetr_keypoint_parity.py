"""Weights-gated parity test for the RF-DETR GroupPose keypoint model.

Compares LibreYOLO's RF-DETR keypoint prediction on ``parkour.jpg`` against a
small golden fixture generated from the official keypoint oracle
(``official_kp_parity.pt`` blob ``["final"]``). The golden values (boxes xyxy,
kp_xy, kp_confidence) live in ``tests/data/rfdetr_keypoint_parkour_golden.json``
(~5 KB, committed) so the test needs no oracle download.

The converted LibreYOLO checkpoint is NOT committed. The test loads it from
``$LIBREYOLO_RFDETR_KP_CKPT`` (or the default local download path) and SKIPS when
neither is present.

    pytest tests/e2e/test_rfdetr_keypoint_parity.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.rfdetr,
    pytest.mark.slow,
    pytest.mark.external_data,
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _REPO_ROOT / "tests" / "data" / "rfdetr_keypoint_parkour_golden.json"
_PARKOUR = _REPO_ROOT / "tests" / "data" / "parkour.jpg"
_DEFAULT_CKPT = _REPO_ROOT / "downloads" / "LibreRFDETRx-pose.pt"

# Tolerances. The oracle and LibreYOLO run the same weights, so agreement is
# sub-pixel in practice; keep generous-but-meaningful bounds.
_KP_XY_MAX_ABS_PX = 2.0
_CONF_MAX_ABS = 0.05
_BOX_IOU_MIN = 0.9


def _resolve_checkpoint() -> Path | None:
    env = os.environ.get("LIBREYOLO_RFDETR_KP_CKPT")
    if env:
        path = Path(env)
        return path if path.is_file() else None
    return _DEFAULT_CKPT if _DEFAULT_CKPT.is_file() else None


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def test_rfdetr_keypoint_parity_matches_oracle():
    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        pytest.skip(
            "RF-DETR keypoint checkpoint not available "
            "(set LIBREYOLO_RFDETR_KP_CKPT or stage the local download)."
        )
    if not _PARKOUR.is_file():
        pytest.skip(f"missing sample image {_PARKOUR}")

    assert _GOLDEN.is_file(), f"golden fixture missing: {_GOLDEN}"
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    golden_boxes = np.asarray(golden["boxes_xyxy"], dtype=np.float64)  # (4, 4)
    golden_kp_xy = np.asarray(golden["kp_xy"], dtype=np.float64)       # (4, 17, 2)
    golden_conf = np.asarray(golden["kp_confidence"], dtype=np.float64)  # (4, 17)
    n_golden = golden_boxes.shape[0]

    from libreyolo import LibreYOLO

    model = LibreYOLO(str(checkpoint))
    results = model.predict(str(_PARKOUR), conf=golden["conf"], device="cpu")
    result = results[0] if isinstance(results, (list, tuple)) else results

    pred_boxes = np.asarray(result.boxes.xyxy, dtype=np.float64)
    keypoints = result.keypoints
    pred_kp_xy = np.asarray(keypoints.xy, dtype=np.float64)
    pred_conf = np.asarray(keypoints.conf, dtype=np.float64)

    assert pred_boxes.shape[0] == n_golden, (
        f"expected {n_golden} detections at conf={golden['conf']}, got {pred_boxes.shape[0]}"
    )

    # Greedily match each golden detection to its best-IoU prediction.
    available = list(range(pred_boxes.shape[0]))
    kp_diffs = []
    conf_diffs = []
    for g_idx in range(n_golden):
        best_iou, best_pred = -1.0, None
        for p_idx in available:
            iou = _box_iou(golden_boxes[g_idx], pred_boxes[p_idx])
            if iou > best_iou:
                best_iou, best_pred = iou, p_idx
        assert best_pred is not None
        assert best_iou >= _BOX_IOU_MIN, (
            f"golden detection {g_idx} only matched IoU {best_iou:.3f} (< {_BOX_IOU_MIN})"
        )
        available.remove(best_pred)

        kp_diffs.append(np.abs(pred_kp_xy[best_pred] - golden_kp_xy[g_idx]).max())
        conf_diffs.append(np.abs(pred_conf[best_pred] - golden_conf[g_idx]).max())

    max_kp_diff = float(max(kp_diffs))
    max_conf_diff = float(max(conf_diffs))
    print(
        f"\nRF-DETR keypoint parity: n={n_golden} "
        f"max_kp_xy_diff={max_kp_diff:.4f}px max_conf_diff={max_conf_diff:.4f}"
    )

    assert max_kp_diff < _KP_XY_MAX_ABS_PX, (
        f"keypoint xy max abs diff {max_kp_diff:.4f}px exceeds {_KP_XY_MAX_ABS_PX}px"
    )
    assert max_conf_diff < _CONF_MAX_ABS, (
        f"keypoint confidence max abs diff {max_conf_diff:.4f} exceeds {_CONF_MAX_ABS}"
    )
