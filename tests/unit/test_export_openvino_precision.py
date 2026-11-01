"""OpenVINO FP32 and INT8 exports keep full-precision weights.

``openvino.save_model`` compresses weights to FP16 unless
``compress_to_fp16=False`` is passed, so the FP32 branch used to write FP16
weights.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "half, int8, compressed", [(True, False, True), (False, False, False), (False, True, False)]
)
def test_save_model_compression_follows_the_precision(
    monkeypatch, tmp_path, half, int8, compressed
):
    from libreyolo.export import openvino as ov_export

    saved = {}
    fake = types.ModuleType("openvino")
    fake.convert_model = lambda path: object()
    fake.save_model = lambda model, path, **kwargs: saved.update(kwargs)
    monkeypatch.setitem(sys.modules, "openvino", fake)
    monkeypatch.setattr(ov_export, "check_openvino_available", lambda: None)
    monkeypatch.setattr(ov_export, "_quantize_int8", lambda model, data, verbose: model)

    ov_export.export_openvino(
        str(tmp_path / "m.onnx"),
        str(tmp_path / "m_openvino"),
        half=half,
        int8=int8,
        calibration_data=object() if int8 else None,
    )
    assert saved.get("compress_to_fp16", True) is compressed
