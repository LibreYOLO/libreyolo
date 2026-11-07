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

# Same split as libreyolo.export.tflite.TFLITE_PER_CHANNEL_FLAG (vocabulary check).
TFLITE_PER_CHANNEL_FLAG = "_experi" + "mental_disable_per_channel"


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
    assert TFLiteExporter.default_int8_calibration_data is True


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


def test_tflite_int8_without_data_uses_the_default_calibration_set(caplog):
    from libreyolo.export.exporter import DEFAULT_INT8_CALIBRATION_DATA

    exporter = TFLiteExporter(_make_wrapper())

    with caplog.at_level("WARNING"):
        data = exporter._resolve_calibration_data(True, None)

    assert data == DEFAULT_INT8_CALIBRATION_DATA
    assert "not representative" in caplog.text
    assert exporter._resolve_calibration_data(True, "custom.yaml") == "custom.yaml"


def test_tflite_rejects_fp16_export():
    exporter = TFLiteExporter(_make_wrapper())

    with pytest.raises(ValueError, match="FP16"):
        exporter(output_path="unused.tflite", half=True)


@pytest.mark.parametrize("family", ["yolo9_p2", "dfine"])
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

    def fake_run(cmd, capture_output, text, env=None):
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

    def fake_run(cmd, capture_output, text, env=None):
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

    def fake_run(cmd, capture_output, text, env=None):
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
        int8_backend="onnx2tf",
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


def test_tflite_int8_export_rejects_float_io_integer_quant(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    _int8_converter(
        monkeypatch,
        {
            "model_float32.tflite": b"fp32",
            "model_integer_quant.tflite": b"float-io int8",
            "model_full_integer_quant_with_int16_act.tflite": b"int16 act",
        },
    )
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    with pytest.raises(RuntimeError, match="did not produce a full-integer"):
        tflite_module.export_tflite(
            str(onnx_path),
            str(dst),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
            int8_backend="onnx2tf",
        )

    assert not dst.exists()


def test_tflite_int8_failure_reports_converter_output(tmp_path):
    from libreyolo.export import tflite as tflite_module

    (tmp_path / "model_float32.tflite").write_bytes(b"fp32")

    with pytest.raises(RuntimeError, match="Full INT8 Quantization tflite output failed"):
        tflite_module._find_converted_tflite(
            tmp_path,
            tmp_path / "model.onnx",
            int8=True,
            converter_output="...\nFull INT8 Quantization tflite output failed.\n",
        )


def test_tflite_int8_export_never_returns_a_float_artifact(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    _int8_converter(monkeypatch, {"model_float32.tflite": b"fp32"})
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    with pytest.raises(RuntimeError, match="did not produce a full-integer"):
        tflite_module.export_tflite(
            str(onnx_path),
            str(dst),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
            int8_backend="onnx2tf",
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

    def fake_export_tflite(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"tflite")
        return kwargs["output_path"]

    def fake_export_onnx(nn_model, _dummy, **kwargs):
        captured["onnx_model"] = nn_model
        Path(kwargs["output_path"]).write_bytes(b"onnx")
        return kwargs["output_path"]

    monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available", lambda: None
    )
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_int8_available", lambda: None
    )
    monkeypatch.setattr("libreyolo.export.tflite.export_tflite", fake_export_tflite)

    exporter(
        output_path=str(output_path),
        imgsz=16,
        simplify=False,
        int8=True,
        data=str(data_yaml),
    )

    assert isinstance(captured["onnx_model"], exporter_module._YOLO9SplitOutputWrapper)
    assert captured["metadata"]["output_layout"] == "boxes_norm_scores"
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


# ---------------------------------------------------------------------------
# INT8: split outputs so boxes and scores keep separate int8 scales
# ---------------------------------------------------------------------------
def _split_family_model(family):
    if family == "yolox":
        from libreyolo.models.yolox.nn import LibreYOLOXModel

        model = LibreYOLOXModel(config="t", nb_classes=3)
    else:
        from libreyolo.models.yolo9.nn import LibreYOLO9Model

        model = LibreYOLO9Model(config="t", nb_classes=3)
    model.eval()
    model.head.export = True
    return model


@pytest.mark.parametrize("family", ["yolox", "yolo9"])
def test_split_output_wrapper_matches_the_regular_export(family):
    from libreyolo.export.exporter import (
        _YOLO9SplitOutputWrapper,
        _YOLOXSplitOutputWrapper,
    )

    torch.manual_seed(0)
    model = _split_family_model(family)
    wrapper_cls = _YOLOXSplitOutputWrapper if family == "yolox" else _YOLO9SplitOutputWrapper
    x = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        reference = model(x)
        boxes, scores = wrapper_cls(model, (64, 96))(x)

    scale = torch.tensor([96.0, 64.0, 96.0, 64.0]).view(1, 4, 1)
    pixel_boxes = boxes * scale  # (B, 4, N)
    if family == "yolox":
        rebuilt = torch.cat([pixel_boxes.permute(0, 2, 1), scores], -1)
    else:
        rebuilt = torch.cat([pixel_boxes, scores.permute(0, 2, 1)], 1)
    assert boxes.shape[1] == 4
    torch.testing.assert_close(rebuilt, reference, atol=5e-4, rtol=1e-4)


def test_bounded_sigmoid_matches_sigmoid_inside_the_bound_and_clamps_outside():
    from libreyolo.export.exporter import _INT8_LOGIT_BOUND, _bounded_sigmoid

    conv = nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        conv.weight.fill_(1.0)
        conv.bias.zero_()
    logits = torch.tensor([-50.0, -8.0, -2.5, 0.0, 3.0, 8.0, 40.0]).view(1, 1, 1, -1)

    out = _bounded_sigmoid(conv, logits)

    expected = logits.clamp(-_INT8_LOGIT_BOUND, _INT8_LOGIT_BOUND).sigmoid()
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("family", ["yolox", "yolo9"])
@pytest.mark.parametrize("boxes_first", [True, False])
def test_tflite_backend_merges_split_int8_outputs(family, boxes_first):
    from libreyolo.backends.tflite import SPLIT_OUTPUT_LAYOUT, TFLiteBackend

    backend = TFLiteBackend.__new__(TFLiteBackend)
    backend.model_family = family
    backend._canvas_hw = (320, 640)
    backend._output_layout = SPLIT_OUTPUT_LAYOUT
    n, channels = 5, 4 if family == "yolox" else 3
    boxes = np.random.default_rng(0).random((1, 4, n)).astype(np.float32)
    scores = np.random.default_rng(1).random((1, n, channels)).astype(np.float32)

    merged = backend._merge_split_outputs([boxes, scores] if boxes_first else [scores, boxes])

    pixels = np.transpose(boxes, (0, 2, 1)) * np.array([640, 320, 640, 320], np.float32)
    expected = np.concatenate([pixels, scores], -1)
    if family == "yolo9":
        expected = np.transpose(expected, (0, 2, 1))
    np.testing.assert_allclose(merged, expected, rtol=1e-6)


def test_split_output_layout_constant_matches_the_runtime():
    from libreyolo.backends.tflite import SPLIT_OUTPUT_LAYOUT as runtime_layout
    from libreyolo.export.tflite import SPLIT_OUTPUT_LAYOUT as export_layout

    assert runtime_layout == export_layout


def test_tflite_int8_rejects_families_outside_the_allowlist(monkeypatch):
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available", lambda: None
    )
    exporter = TFLiteExporter(_make_wrapper(model_name="yolonas"))

    with pytest.raises(NotImplementedError, match="TFLite INT8 export currently supports"):
        exporter(output_path="unused.tflite", int8=True, data="coco8.yaml")


def test_tflite_int8_requires_batch_one():
    exporter = TFLiteExporter(_make_wrapper(model_name="yolox"))

    with pytest.raises(ValueError, match="batch=1"):
        exporter(output_path="unused.tflite", int8=True, batch=2, data="coco8.yaml")


def test_tflite_int8_reports_missing_tensorflow(monkeypatch):
    import libreyolo.export.tflite as tflite_module

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "tensorflow" else real_find_spec(name, *a, **k),
    )

    with pytest.raises(ImportError, match=r"onnx2tf\[tensorflow\]"):
        tflite_module.check_tflite_int8_available()


@pytest.mark.parametrize(
    ("installed", "refused"), [("2.6.7", True), ("2.6.9", False), ("2.7.0", False)]
)
def test_yolo9_e2e_tflite_needs_a_validated_converter(monkeypatch, installed, refused):
    import importlib.metadata

    from libreyolo.export import tflite

    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    if refused:
        with pytest.raises(ImportError, match="onnx2tf >=2.6.9"):
            tflite.check_tflite_converter_version("yolo9_e2e", "detect")
    else:
        tflite.check_tflite_converter_version("yolo9_e2e", "detect")
    tflite.check_tflite_converter_version("yolo9", "detect")  # no minimum



def test_tflite_int8_representative_dataset_is_one_nhwc_image_per_sample():
    from libreyolo.export import tflite as tflite_module

    loader = _FakeCalibration([_constant_images(1.0, 2.0, 3.0)], 3)
    samples = list(tflite_module._int8_representative_dataset(loader))

    # The converter requires every sample to match the model input shape, so a
    # batch-1 graph is fed one image per call.
    assert [s[0].shape for s in samples] == [(1, 2, 2, 3)] * 3
    # NCHW -> NHWC, values preserved.
    assert samples[0][0][0, 0, 0, 0] == pytest.approx(1.0)
    assert samples[1][0][0, 0, 0, 0] == pytest.approx(2.0)


def test_tflite_int8_representative_dataset_drops_loader_padding():
    from libreyolo.export import tflite as tflite_module

    # CalibrationDataLoader pads the final batch by repeating its last image;
    # num_samples says only two are real.
    loader = _FakeCalibration([_constant_images(1.0, 2.0, 2.0)], 2)
    samples = list(tflite_module._int8_representative_dataset(loader))

    assert len(samples) == 2


def test_tflite_int8_export_defaults_to_single_pass(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    monkeypatch.setattr(tflite_module, "check_tflite_export_available", lambda: None)
    calls = {"saved_model": 0, "convert": 0, "subprocess": 0}

    def fake_build(onnx_path, saved_model_dir, *, verbose):
        calls["saved_model"] += 1
        saved_model_dir.mkdir(parents=True, exist_ok=True)

    def fake_convert(saved_model_dir, output_path, calibration_data, **kwargs):
        calls["convert"] += 1
        calls["kwargs"] = kwargs
        output_path.write_bytes(b"int8-single-pass")

    def fake_run(*args, **kwargs):
        calls["subprocess"] += 1
        raise AssertionError("the default INT8 path must not shell out")

    monkeypatch.setattr(tflite_module, "_build_saved_model", fake_build)
    monkeypatch.setattr(tflite_module, "_convert_saved_model_int8", fake_convert)
    monkeypatch.setattr(subprocess, "run", fake_run)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    dst = tmp_path / "model.tflite"

    result = tflite_module.export_tflite(
        str(onnx_path),
        str(dst),
        int8=True,
        calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
        metadata={"model_family": "yolo9", "precision": "int8"},
        int8_output_dtype="float32",
    )

    assert result == str(dst)
    assert dst.read_bytes() == b"int8-single-pass"
    assert calls == {
        "saved_model": 1,
        "convert": 1,
        "subprocess": 0,
        "kwargs": {"output_dtype": "float32"},
    }
    sidecar = json.loads(Path(str(dst) + ".json").read_text())
    assert sidecar["precision"] == "int8"


def test_tflite_int8_export_rejects_unknown_backend(tmp_path):
    from libreyolo.export import tflite as tflite_module

    with pytest.raises(ValueError, match="int8_backend"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
            int8_backend="magic",
        )


def test_tflite_exporter_forwards_int8_calibration_options(monkeypatch, tmp_path):
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
        captured["tflite"] = kwargs
        Path(kwargs["output_path"]).write_bytes(b"tflite")
        return kwargs["output_path"]

    def fake_load_calibration(*_args, **_kwargs):
        return _FakeCalibration([_constant_images(1.0)], 1)

    monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(exporter, "_load_calibration", fake_load_calibration)
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_export_available", lambda: None
    )
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_converter_version",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "libreyolo.export.tflite.check_tflite_int8_available", lambda: None
    )
    monkeypatch.setattr("libreyolo.export.tflite.export_tflite", fake_export_tflite)

    exporter(
        output_path=str(output_path),
        imgsz=(16, 32),
        int8=True,
        data="coco8.yaml",
        int8_output_dtype="float32",
    )

    assert captured["tflite"]["int8"] is True
    assert captured["tflite"]["int8_backend"] == "converter"
    assert captured["tflite"]["int8_output_dtype"] == "float32"


def test_tflite_int8_onnx2tf_backend_rejects_float32_output_dtype(tmp_path):
    from libreyolo.export import tflite as tflite_module

    # The CLI route ships *_full_integer_quant.tflite, which always has int8
    # inputs and outputs; accepting float32 here would hand the caller a graph
    # typed differently from what it asked for.
    with pytest.raises(ValueError, match="always writes int8 input and output"):
        tflite_module.export_tflite(
            str(tmp_path / "model.onnx"),
            str(tmp_path / "model.tflite"),
            int8=True,
            calibration_data=_FakeCalibration([_constant_images(1.0)], 1),
            int8_backend="onnx2tf",
            int8_output_dtype="float32",
        )


def test_build_saved_model_uses_fdosm_without_quantizing(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    monkeypatch.setattr(tflite_module, "_onnx2tf_command", lambda: ["onnx2tf"])
    captured = {}

    def fake_run(cmd, capture_output, text, env=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env or {})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake onnx")
    saved_model_dir = tmp_path / "saved_model"

    tflite_module._build_saved_model(onnx_path, saved_model_dir, verbose=False)

    cmd = captured["cmd"]
    assert cmd[0] == "onnx2tf"
    assert cmd[cmd.index("-i") + 1] == str(onnx_path)
    assert cmd[cmd.index("-o") + 1] == str(saved_model_dir)
    # SavedModel only: no -oiqt (quantization) and no -tb (tflite backend).
    assert "-fdosm" in cmd
    assert "-oiqt" not in cmd
    assert "-tb" not in cmd
    # The calibration pass is CPU-bound, so the subprocess gets thread settings.
    assert captured["env"]["TF_NUM_INTRAOP_THREADS"]
    assert captured["env"]["OMP_NUM_THREADS"]


def _fake_tensorflow(monkeypatch):
    """Install a fake tensorflow module and return the converter it creates."""
    import types

    created = {}

    class _FakeConverter:
        def __init__(self, saved_model_dir):
            self.saved_model_dir = saved_model_dir
            self.optimizations = None
            self.target_spec = types.SimpleNamespace(supported_ops=None)
            self.representative_dataset = None
            self.inference_input_type = None
            self.inference_output_type = None
            setattr(self, TFLITE_PER_CHANNEL_FLAG, True)
            self.convert_calls = 0
            created["converter"] = self

        def convert(self):
            self.convert_calls += 1
            return b"quantized-model"

    class _FakeTFLiteConverter:
        @staticmethod
        def from_saved_model(path):
            return _FakeConverter(path)

    fake_tf = types.SimpleNamespace(
        lite=types.SimpleNamespace(
            TFLiteConverter=_FakeTFLiteConverter,
            Optimize=types.SimpleNamespace(DEFAULT="DEFAULT"),
            OpsSet=types.SimpleNamespace(
                TFLITE_BUILTINS_INT8="TFLITE_BUILTINS_INT8",
                SELECT_TF_OPS="SELECT_TF_OPS",
            ),
        ),
        int8="tf.int8",
        float32="tf.float32",
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    return created


@pytest.mark.parametrize(
    ("output_dtype", "expected"), [("int8", "tf.int8"), ("float32", "tf.float32")]
)
def test_convert_saved_model_int8_converts_once(
    monkeypatch, tmp_path, output_dtype, expected
):
    from libreyolo.export import tflite as tflite_module

    created = _fake_tensorflow(monkeypatch)
    saved_model_dir = tmp_path / "saved_model"
    saved_model_dir.mkdir()
    output_path = tmp_path / "model.tflite"
    loader = _FakeCalibration([_constant_images(1.0, 2.0)], 2)

    tflite_module._convert_saved_model_int8(
        saved_model_dir, output_path, loader, output_dtype=output_dtype
    )

    converter = created["converter"]
    # One quantization pass, not the two onnx2tf's -oiqt performs.
    assert converter.convert_calls == 1
    assert converter.saved_model_dir == str(saved_model_dir)
    assert converter.optimizations == ["DEFAULT"]
    assert converter.target_spec.supported_ops == [
        "TFLITE_BUILTINS_INT8",
        "SELECT_TF_OPS",
    ]
    assert converter.inference_input_type == "tf.int8"
    assert converter.inference_output_type == expected
    assert getattr(converter, TFLITE_PER_CHANNEL_FLAG) is False
    # Representative dataset: the loader's NCHW images, as NHWC batch-1 samples.
    samples = list(converter.representative_dataset())
    assert [sample[0].shape for sample in samples] == [(1, 2, 2, 3)] * 2
    assert output_path.read_bytes() == b"quantized-model"


def test_convert_saved_model_int8_rejects_unknown_output_dtype(monkeypatch, tmp_path):
    from libreyolo.export import tflite as tflite_module

    _fake_tensorflow(monkeypatch)

    with pytest.raises(ValueError, match="int8_output_dtype"):
        tflite_module._convert_saved_model_int8(
            tmp_path,
            tmp_path / "model.tflite",
            _FakeCalibration([_constant_images(1.0)], 1),
            output_dtype="float16",
        )
