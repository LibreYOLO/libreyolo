"""Unit tests for TFLite export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from libreyolo.export.exporter import BaseExporter, TFLiteExporter

pytestmark = pytest.mark.unit


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_wrapper(model_name="yolo9", size="t", input_size=32):
    wrapper = MagicMock()
    wrapper.model = _TinyModel()
    wrapper.model.eval()
    wrapper.size = size
    wrapper.nb_classes = 4
    wrapper.names = {i: f"class_{i}" for i in range(wrapper.nb_classes)}
    wrapper.device = torch.device("cpu")
    wrapper._get_model_name.return_value = model_name
    wrapper._get_input_size.return_value = input_size
    wrapper.task = "detect"
    wrapper.SUPPORTED_TASKS = ("detect",)
    wrapper.DEFAULT_TASK = "detect"
    return wrapper


def _mock_onnx_available(monkeypatch):
    import libreyolo.export.exporter as exporter_module

    original_find_spec = exporter_module.importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "onnx":
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(exporter_module.importlib.util, "find_spec", fake_find_spec)


def test_tflite_format_registered():
    assert "tflite" in BaseExporter._registry
    assert TFLiteExporter.suffix == ".tflite"
    assert TFLiteExporter.requires_onnx is True
    assert TFLiteExporter.supports_int8 is False
    assert TFLiteExporter.supports_fp16 is False
    assert TFLiteExporter.apply_model_half is False


def test_tflite_family_support_scaffold():
    from libreyolo.export.tflite import (
        ensure_tflite_family_supported,
        supported_tflite_exports,
    )

    assert supported_tflite_exports() == (("yolo9", "detect"),)
    ensure_tflite_family_supported("yolo9", "detect")
    with pytest.raises(NotImplementedError, match="task 'segment'"):
        ensure_tflite_family_supported("yolo9", "segment")
    with pytest.raises(NotImplementedError, match="RF-DETR"):
        ensure_tflite_family_supported("rfdetr", "detect")


def test_tflite_rejects_dynamic_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="static input shapes"):
        exporter(dynamic=True)


def test_tflite_rejects_int8_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="INT8"):
        exporter(output_path="unused.tflite", int8=True, data="coco8")


def test_tflite_rejects_fp16_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="FP16"):
        exporter(output_path="unused.tflite", half=True)


@pytest.mark.parametrize("family", ["rfdetr", "yolox", "yolo9_e2e", "dfine"])
def test_tflite_blocks_unvalidated_families_before_onnx_export(family):
    exporter = TFLiteExporter(_make_wrapper(model_name=family))

    with pytest.raises(NotImplementedError, match="currently supports"):
        exporter(output_path="unused.tflite")


def test_tflite_blocks_yolo9_segment_before_onnx_export():
    wrapper = _make_wrapper(model_name="yolo9")
    wrapper.task = "segment"
    exporter = TFLiteExporter(wrapper)

    with pytest.raises(NotImplementedError, match="task 'segment'"):
        exporter(output_path="unused.tflite")


def test_tflite_export_copies_float32_output(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    onnx_path = tmp_path / "model.onnx"
    fp32_dst = tmp_path / "model.tflite"
    onnx_path.write_bytes(b"fake onnx")

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)
    monkeypatch.setattr(tflite_module, "_onnx2tf_command", lambda: ["onnx2tf"])

    def fake_run(cmd, capture_output, text):
        output_dir = Path(cmd[cmd.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model_float32.tflite").write_bytes(b"fp32")
        (output_dir / "model_float16.tflite").write_bytes(b"fp16")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = tflite_module.export_tflite(
        str(onnx_path),
        str(fp32_dst),
        onnx2tf_args=["--flatbuffer_direct_allow_custom_ops"],
        metadata={"model_family": "yolo9", "size": "t", "nc": 4},
    )

    assert result == str(fp32_dst)
    assert fp32_dst.read_bytes() == b"fp32"
    sidecar = json.loads(Path(str(fp32_dst) + ".json").read_text())
    assert sidecar["model_family"] == "yolo9"


def test_tflite_export_reports_converter_failure(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)
    monkeypatch.setattr(tflite_module, "_onnx2tf_command", lambda: ["onnx2tf"])

    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 2, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="onnx2tf failed"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
        )


def test_tflite_export_rejects_float16_helper(tmp_path):
    from libreyolo.export import tflite as tflite_module

    with pytest.raises(ValueError, match="FP16"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
            half=True,
        )


def test_check_tflite_export_raises_helpful_error_when_missing():
    if sys.version_info >= (3, 12):
        try:
            import onnx2tf  # noqa: F401

            pytest.skip("TFLite export dependencies are installed")
        except ImportError:
            pass

    from libreyolo.export.tflite import check_tflite_export_available

    with pytest.raises(ImportError) as exc_info:
        check_tflite_export_available()

    error = str(exc_info.value)
    assert "TFLite" in error
    assert "pip install libreyolo[tflite]" in error


def test_tflite_exporter_runs_static_onnx_then_helper(monkeypatch, tmp_path):
    import libreyolo.export.exporter as exporter_module

    _mock_onnx_available(monkeypatch)
    wrapper = _make_wrapper()
    exporter = TFLiteExporter(wrapper)
    output_path = tmp_path / "model.tflite"
    captured = {}

    def fake_export_onnx(_nn_model, _dummy, **kwargs):
        captured["onnx"] = kwargs
        captured["onnx_dummy_shape"] = tuple(_dummy.shape)
        Path(kwargs["output_path"]).write_bytes(b"onnx")
        return kwargs["output_path"]

    def fake_export_tflite(**kwargs):
        captured["tflite"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"tflite")
        return kwargs["output_path"]

    monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available",
        lambda: None,
    )
    monkeypatch.setattr("libreyolo.export.tflite.export_tflite", fake_export_tflite)

    result = exporter(
        output_path=str(output_path),
        imgsz=(16, 32),
        simplify=False,
        onnx2tf_args=["--flatbuffer_direct_allow_custom_ops"],
    )

    assert result == str(output_path)
    assert captured["onnx"]["dynamic"] is False
    assert captured["onnx_dummy_shape"] == (1, 3, 16, 32)
    assert captured["tflite"]["output_path"] == str(output_path)
    assert captured["tflite"]["metadata"]["model_family"] == "yolo9"
    assert captured["tflite"]["metadata"]["imgsz_h"] == 16
    assert captured["tflite"]["metadata"]["imgsz_w"] == 32
    assert captured["tflite"]["onnx2tf_args"] == ["--flatbuffer_direct_allow_custom_ops"]
    assert not Path(captured["tflite"]["onnx_path"]).exists()


def test_intermediate_onnx_removed_when_tflite_helper_fails(monkeypatch, tmp_path):
    import libreyolo.export.exporter as exporter_module

    _mock_onnx_available(monkeypatch)
    wrapper = _make_wrapper()
    exporter = TFLiteExporter(wrapper)
    output_path = tmp_path / "model.tflite"
    captured = {}

    def fake_export_onnx(_nn_model, _dummy, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"onnx")
        return kwargs["output_path"]

    def fake_export_tflite(**kwargs):
        captured["onnx_path"] = kwargs["onnx_path"]
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available",
        lambda: None,
    )
    monkeypatch.setattr("libreyolo.export.tflite.export_tflite", fake_export_tflite)

    with pytest.raises(RuntimeError, match="conversion failed"):
        exporter(output_path=str(output_path), simplify=False)

    assert not Path(captured["onnx_path"]).exists()
