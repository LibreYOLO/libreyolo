"""Unit tests for TFLite export."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from libreyolo.export.exporter import BaseExporter, TFLiteExporter
from libreyolo.export.support import get_support

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
    assert TFLiteExporter.supports_int8 is True
    assert TFLiteExporter.supports_fp16 is False
    assert TFLiteExporter.apply_model_half is False
    assert TFLiteExporter.default_int8_calibration_data is False


def test_tflite_family_support_scaffold():
    from libreyolo.export.tflite import (
        ensure_tflite_family_supported,
        supported_tflite_exports,
    )

    exports = supported_tflite_exports()
    assert ("yolo9", "detect") in exports
    assert ("swinir", "restore") in exports
    assert ("rfdetr", "detect") not in exports
    assert ("rfdetr", "segment") not in exports
    assert ("rfdetr", "pose") not in exports
    ensure_tflite_family_supported("yolo9", "detect")
    ensure_tflite_family_supported("swinir", "restore")
    with pytest.raises(NotImplementedError, match="STRIDED_SLICE"):
        ensure_tflite_family_supported("rfdetr", "detect")
    with pytest.raises(NotImplementedError, match="Einsum"):
        ensure_tflite_family_supported("rfdetr", "segment")
    with pytest.raises(NotImplementedError, match="timebox"):
        ensure_tflite_family_supported("rfdetr", "pose")
    with pytest.raises(NotImplementedError, match="task 'segment'"):
        ensure_tflite_family_supported("yolo9", "segment")


def test_tflite_rejects_dynamic_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="static input shapes"):
        exporter(dynamic=True)


def test_tflite_int8_requires_calibration_data():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="requires calibration data"):
        exporter(output_path="unused.tflite", int8=True)


def test_tflite_rejects_fp16_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="FP16"):
        exporter(output_path="unused.tflite", half=True)


@pytest.mark.parametrize("family", ["yolo9_e2e", "dfine"])
def test_tflite_blocks_unvalidated_families_before_onnx_export(family):
    exporter = TFLiteExporter(_make_wrapper(model_name=family))

    with pytest.raises(NotImplementedError, match="currently supports"):
        exporter(output_path="unused.tflite")


def test_tflite_accepts_parity_validated_yolox():
    assert get_support("yolox", "detect", "tflite").tier == "validated"


def test_tflite_blocks_yolo9_segment_before_onnx_export():
    wrapper = _make_wrapper(model_name="yolo9")
    wrapper.task = "segment"
    exporter = TFLiteExporter(wrapper)

    with pytest.raises(NotImplementedError, match="YOLO9 segmentation export"):
        exporter(output_path="unused.tflite")


def test_yolox_tflite_raw_parity(tmp_path):
    if (
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None
    ):
        pytest.skip("onnx2tf and ai-edge-litert are required")

    import numpy as np

    from libreyolo import LibreYOLO, LibreYOLOX

    model = LibreYOLOX(None, size="n", nb_classes=3, device="cpu")
    model.model.eval()
    image = np.random.default_rng(29).integers(
        0, 256, size=(80, 100, 3), dtype=np.uint8
    )
    tensor, *_ = model._preprocess(image, input_size=96)
    old_export = model.model.head.export
    model.model.head.export = True
    try:
        with torch.no_grad():
            native = model.model(tensor).numpy()
    finally:
        model.model.head.export = old_export

    artifact = model.export(
        "tflite",
        output_path=str(tmp_path / "yolox.tflite"),
        imgsz=96,
        dynamic=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual = backend._run_inference(tensor.numpy())[0]
    np.testing.assert_allclose(actual, native, rtol=1e-5, atol=1e-5)


def test_tflite_export_copies_float32_output(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    onnx_path = tmp_path / "model.onnx"
    fp32_dst = tmp_path / "model.tflite"
    onnx_path.write_bytes(b"fake onnx")

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)
    monkeypatch.setattr(tflite_module, "_onnx2tf_command", lambda: ["onnx2tf"])

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
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
    assert captured["cmd"][captured["cmd"].index("-tb") + 1] == "flatbuffer_direct"
    assert "-oiqt" not in captured["cmd"]
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
    assert captured["tflite"]["onnx2tf_args"] == [
        "--flatbuffer_direct_allow_custom_ops"
    ]
    assert not Path(captured["tflite"]["onnx_path"]).exists()


class _FakeCalibration:
    """Stand-in for CalibrationDataLoader: NCHW batches, padded final batch."""

    def __init__(self, batches, num_samples):
        self._batches = batches
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self._batches)


def _constant_images(*values):
    return np.concatenate(
        [np.full((1, 3, 2, 2), value, dtype=np.float32) for value in values]
    )


def _int8_converter(monkeypatch, produced):
    """Patch onnx2tf away and record the command it would have been given."""
    from libreyolo.export import tflite as tflite_module

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)
    monkeypatch.setattr(tflite_module, "_onnx2tf_command", lambda: ["onnx2tf"])
    monkeypatch.setattr(tflite_module, "_onnx_input_name", lambda _path: "images")
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = list(cmd)
        output_dir = Path(cmd[cmd.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in produced.items():
            (output_dir / name).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_int8_calibration_npy_is_nhwc_and_drops_batch_padding(tmp_path):
    from libreyolo.export import tflite as tflite_module

    # Second batch repeats its last image, the way CalibrationDataLoader pads.
    loader = _FakeCalibration(
        [_constant_images(1.0, 2.0), _constant_images(3.0, 3.0)],
        num_samples=3,
    )

    npy_path = tflite_module._write_int8_calibration_npy(loader, tmp_path)
    data = np.load(npy_path)

    assert data.shape == (3, 2, 2, 3)
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data[:, 0, 0, 0], [1.0, 2.0, 3.0])


def test_int8_calibration_npy_trims_when_images_are_skipped(tmp_path):
    from libreyolo.export import tflite as tflite_module

    # num_samples counts files on disk; unreadable ones never reach a batch.
    loader = _FakeCalibration([_constant_images(1.0, 2.0)], num_samples=4)

    data = np.load(tflite_module._write_int8_calibration_npy(loader, tmp_path))

    assert data.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(data[:, 0, 0, 0], [1.0, 2.0])


def test_int8_calibration_npy_rejects_empty_dataset(tmp_path):
    from libreyolo.export import tflite as tflite_module

    with pytest.raises(RuntimeError, match="no usable images"):
        tflite_module._write_int8_calibration_npy(
            _FakeCalibration([], num_samples=4), tmp_path
        )


def test_tflite_int8_helper_requires_calibration_data(tmp_path):
    from libreyolo.export import tflite as tflite_module

    with pytest.raises(ValueError, match="requires calibration data"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
            int8=True,
        )


def test_tflite_int8_rejected_for_rfdetr(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)

    with pytest.raises(NotImplementedError, match="RF-DETR"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], num_samples=1),
            metadata={"model_family": "rfdetr"},
        )


def test_tflite_int8_export_selects_full_integer_artifact(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    captured = _int8_converter(
        monkeypatch,
        {
            "model_float32.tflite": b"fp32",
            "model_integer_quant.tflite": b"int8",
            "model_full_integer_quant.tflite": b"full-int8",
        },
    )
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    result = tflite_module.export_tflite(
        str(onnx_path),
        str(dst),
        int8=True,
        calibration_data=_FakeCalibration([_constant_images(1.0, 2.0)], 2),
        metadata={"model_family": "yolo9", "precision": "int8"},
    )

    assert result == str(dst)
    assert dst.read_bytes() == b"full-int8"

    cmd = captured["cmd"]
    # The integer path runs on tf_converter; flatbuffer_direct aborts on any op
    # it cannot keep in int8 end to end.
    assert cmd[cmd.index("-tb") + 1] == "tf_converter"
    assert "-oiqt" in cmd
    cind = cmd.index("-cind")
    assert cmd[cind + 1] == "images"
    assert cmd[cind + 2].endswith(".npy")
    # mean 0 / std 1: the calibration batches are already preprocessed, so a
    # second normalization inside onnx2tf would calibrate the wrong ranges.
    assert cmd[cind + 3 : cind + 5] == ["0.0", "1.0"]

    sidecar = json.loads(Path(str(dst) + ".json").read_text())
    assert sidecar["precision"] == "int8"


def test_tflite_int8_export_falls_back_to_integer_quant(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    _int8_converter(
        monkeypatch,
        {
            "model_float32.tflite": b"fp32",
            "model_integer_quant.tflite": b"int8",
        },
    )
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    tflite_module.export_tflite(
        str(onnx_path),
        str(dst),
        int8=True,
        calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
    )

    assert dst.read_bytes() == b"int8"


def test_tflite_int8_export_never_returns_a_float_artifact(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    _int8_converter(monkeypatch, {"model_float32.tflite": b"fp32"})
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    with pytest.raises(RuntimeError, match="did not produce an INT8"):
        tflite_module.export_tflite(
            str(onnx_path),
            str(dst),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
        )

    assert not dst.exists()


def test_tflite_exporter_forwards_int8_and_calibration(monkeypatch, tmp_path):
    import libreyolo.export.exporter as exporter_module

    _mock_onnx_available(monkeypatch)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    for idx in range(3):
        cv2.imwrite(
            str(image_dir / f"{idx}.jpg"),
            rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8),
        )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images",
                "val: images",
                "nc: 1",
                "names:",
                "  0: object",
            ]
        ),
        encoding="utf-8",
    )

    def preprocess(img_rgb, imgsz):
        h, w = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
        resized = cv2.resize(img_rgb, (w, h)).astype(np.float32) / 255.0
        return np.transpose(resized, (2, 0, 1)), 1.0

    wrapper = _make_wrapper()
    wrapper._get_preprocess_numpy.return_value = preprocess
    exporter = TFLiteExporter(wrapper)
    output_path = tmp_path / "model.tflite"
    captured = {}

    def fake_export_onnx(_nn_model, _dummy, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"onnx")
        return kwargs["output_path"]

    def fake_export_tflite(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"tflite")
        return kwargs["output_path"]

    monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available", lambda: None
    )
    monkeypatch.setattr("libreyolo.export.tflite.export_tflite", fake_export_tflite)

    exporter(
        output_path=str(output_path),
        imgsz=16,
        simplify=False,
        int8=True,
        data=str(data_yaml),
    )

    assert captured["int8"] is True
    assert captured["half"] is False
    assert captured["metadata"]["precision"] == "int8"
    assert captured["calibration_data"].num_samples == 3
    assert next(iter(captured["calibration_data"])).shape[1:] == (3, 16, 16)


def test_tflite_backend_restores_yolonas_channel_first_outputs():
    from libreyolo.backends.tflite import TFLiteBackend

    class _Interpreter:
        def __init__(self):
            self.outputs = {
                1: np.arange(3 * 5, dtype=np.float32).reshape(1, 3, 5),
                2: np.arange(4 * 5, dtype=np.float32).reshape(1, 4, 5),
            }

        def set_tensor(self, index, value):
            self.input = (index, value)

        def invoke(self):
            return None

        def get_tensor(self, index):
            return self.outputs[index]

    backend = TFLiteBackend.__new__(TFLiteBackend)
    backend.interpreter = _Interpreter()
    backend.input_details = [
        {"index": 0, "shape": np.array([1, 3, 8, 8]), "dtype": np.float32}
    ]
    backend.output_details = [
        {"index": 1, "dtype": np.float32},
        {"index": 2, "dtype": np.float32},
    ]
    backend.model_family = "yolonas"
    backend.task = "detect"
    backend.nb_classes = 3

    scores, boxes = backend._run_inference(
        np.zeros((1, 3, 8, 8), dtype=np.float32)
    )

    assert scores.shape == (1, 5, 3)
    assert boxes.shape == (1, 5, 4)


def test_tflite_backend_restores_edge_channel_first_output():
    from libreyolo.backends.tflite import TFLiteBackend

    class _Interpreter:
        def set_tensor(self, index, value):
            self.input = (index, value)

        def invoke(self):
            return None

        def get_tensor(self, index):
            return np.arange(8 * 8, dtype=np.float32).reshape(1, 8, 8, 1)

    backend = TFLiteBackend.__new__(TFLiteBackend)
    backend.interpreter = _Interpreter()
    backend.input_details = [
        {"index": 0, "shape": np.array([1, 8, 8, 3]), "dtype": np.float32}
    ]
    backend.output_details = [{"index": 1, "dtype": np.float32}]
    backend.model_family = "dexined"
    backend.task = "edge"
    backend.nb_classes = 1

    (edges,) = backend._run_inference(
        np.zeros((1, 3, 8, 8), dtype=np.float32)
    )

    assert edges.shape == (1, 1, 8, 8)


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
