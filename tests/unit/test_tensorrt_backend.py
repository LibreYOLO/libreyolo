from __future__ import annotations

import pytest

from libreyolo.backends.tensorrt import TensorRTBackend


pytestmark = pytest.mark.unit


def _bare_tensorrt_backend(path: str, model_family: str | None = None):
    backend = TensorRTBackend.__new__(TensorRTBackend)
    backend.model_path = path
    backend.model_family = model_family
    backend.output_names = ["pred_logits", "pred_boxes"]
    backend.output_shapes = {
        "pred_logits": (1, 300, 80),
        "pred_boxes": (1, 300, 4),
    }
    return backend


@pytest.mark.parametrize(
    ("path", "family"),
    [
        ("LibreDEIMv2_s.engine", "deimv2"),
        ("LibreEC_s.engine", "ec"),
        ("LibreDFINE_s.engine", "dfine"),
        ("LibreDEIM_s.engine", "deim"),
        ("LibreRTDETR_s.engine", "rtdetr"),
        ("LibreRFDETR_s.engine", "rfdetr"),
    ],
)
def test_tensorrt_sidecarless_detr_family_detection_uses_filename(path, family):
    backend = _bare_tensorrt_backend(path)

    assert backend._detect_model_family() == family


def test_tensorrt_backend_reports_deimv2_model_name():
    backend = _bare_tensorrt_backend("LibreDEIMv2_s.engine", model_family="deimv2")

    assert backend._get_model_name() == "deimv2"
