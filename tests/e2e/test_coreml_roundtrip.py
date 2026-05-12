"""End-to-end CoreML export + load roundtrip. macOS only.

Asserts numerical-ish parity: the CoreML model must produce the same number
of detections as the source PyTorch model on the bundled sample image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.coreml, pytest.mark.e2e]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def _macos_only():
    if sys.platform != "darwin":
        pytest.skip("CoreML tests require macOS")
    pytest.importorskip("coremltools")


def _load_or_skip(filename: str):
    """Load a model via the LibreYOLO factory (which auto-downloads on miss).

    Skips the test only if the factory itself errors — e.g. network failure
    or unknown filename pattern.
    """
    from libreyolo import LibreYOLO

    try:
        return LibreYOLO(filename)
    except Exception as e:
        pytest.skip(f"Could not load {filename}: {e}")


def _load_yolox_nano():
    return _load_or_skip("LibreYOLOXn.pt")


def _load_yolo9_tiny():
    return _load_or_skip("LibreYOLO9t.pt")


def _assert_parity(pt_res, cm_res, *, conf_tol: float = 1e-3) -> None:
    """Strong numerical parity check, robust to borderline-threshold detections.

    Compares the top-min(N_pt, N_cm) detections by confidence: each pair must
    agree to within ``conf_tol``. Allows a single missing detection (uint8
    quantization can knock a barely-above-threshold detection under).
    """
    pt_conf = sorted([float(s) for s in pt_res.boxes.conf], reverse=True)
    cm_conf = sorted([float(s) for s in cm_res.boxes.conf], reverse=True)

    assert abs(len(pt_conf) - len(cm_conf)) <= 1, (
        f"detection counts differ too much: pt={len(pt_conf)} cm={len(cm_conf)}"
    )

    n = min(len(pt_conf), len(cm_conf))
    assert n >= 1, "expected at least one matched detection"
    for i in range(n):
        assert abs(pt_conf[i] - cm_conf[i]) < conf_tol, (
            f"confidence mismatch at rank {i}: pt={pt_conf[i]} cm={cm_conf[i]}"
        )


def test_yolox_export_fp32_parity(tmp_path):
    """YOLOX nano: CoreML detections must numerically match PyTorch."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    pt_res = pt_model(SAMPLE_IMAGE)
    assert len(pt_res.boxes) >= 1, "PT produced 0 detections — sample image issue"

    out_path = tmp_path / "model.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path))
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    cm_res = coreml_model(SAMPLE_IMAGE)
    _assert_parity(pt_res, cm_res)


def test_yolox_export_fp16(tmp_path):
    """fp16 export should still produce a non-empty detection set."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    out_path = tmp_path / "model_fp16.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path), half=True)
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    assert len(coreml_model(SAMPLE_IMAGE).boxes) >= 1


def test_yolox_export_with_embedded_nms(tmp_path):
    """nms=True should produce a loadable CoreML pipeline."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolox_nano()
    out_path = tmp_path / "model_nms.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path), nms=True)
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    assert len(coreml_model(SAMPLE_IMAGE).boxes) >= 1


def test_yolo9_export_fp32_parity(tmp_path):
    """YOLO9 tiny: CoreML detections must numerically match PyTorch."""
    from libreyolo import SAMPLE_IMAGE, LibreYOLO

    pt_model = _load_yolo9_tiny()
    pt_res = pt_model(SAMPLE_IMAGE)
    assert len(pt_res.boxes) >= 1

    out_path = tmp_path / "model.mlpackage"
    pt_model.export(format="coreml", output_path=str(out_path))
    assert out_path.is_dir()

    coreml_model = LibreYOLO(str(out_path))
    cm_res = coreml_model(SAMPLE_IMAGE)
    _assert_parity(pt_res, cm_res)


def test_compute_units_kwarg_accepted(tmp_path):
    from libreyolo import LibreYOLO

    model = _load_yolox_nano()
    out_path = tmp_path / "model_cpu.mlpackage"
    model.export(
        format="coreml",
        output_path=str(out_path),
        compute_units="cpu_only",
    )
    assert out_path.is_dir()


def test_rfdetr_nms_true_raises(tmp_path):
    # RF-DETR auto-detection requires the rfdetr extra to be installed
    # (the LibreYOLORFDETR wrapper is registered lazily via _ensure_rfdetr).
    # Without it, the factory can't resolve size from the filename and the
    # download path fails. Install with: pip install -e ".[coreml,rfdetr]"
    pytest.importorskip("rfdetr")
    rfdetr = _load_or_skip("LibreRFDETRn.pt")
    with pytest.raises(NotImplementedError, match="RF-DETR"):
        rfdetr.export(
            format="coreml",
            output_path=str(tmp_path / "rfdetr.mlpackage"),
            nms=True,
        )
