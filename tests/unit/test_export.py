"""Unit tests for the unified Exporter module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import libreyolo.export.exporter as exporter_module
import libreyolo.export.onnx as onnx_module
from libreyolo.export.exporter import (
    BaseExporter,
    CoreMLExporter,
    ExecuTorchExporter,
    NcnnExporter,
    OnnxExporter,
    OpenVINOExporter,
    PaddleExporter,
    TensorRTExporter,
    TFLiteExporter,
    TorchScriptExporter,
)
from libreyolo.export.onnx import export_onnx

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal model for export tests (no real weights needed)."""

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


def test_onnx_simplify_skip_targets_known_macos_arm64_combo(monkeypatch):
    versions = {"onnx": "1.22.0", "onnxsim": "0.6.5"}
    monkeypatch.setattr(onnx_module.importlib_metadata, "version", versions.__getitem__)
    monkeypatch.setattr(onnx_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(onnx_module.platform, "machine", lambda: "arm64")
    assert onnx_module._should_skip_onnx_simplify()

    versions["onnxsim"] = "0.6.6"
    assert not onnx_module._should_skip_onnx_simplify()

    versions.update(onnx="1.21.0", onnxsim="0.6.5")
    assert not onnx_module._should_skip_onnx_simplify()

    monkeypatch.setattr(onnx_module.platform, "system", lambda: "Linux")
    assert not onnx_module._should_skip_onnx_simplify()


class _TinyRFDETRExport(nn.Module):
    """Small RF-DETR-shaped export module for ONNX schema tests."""

    def __init__(self, *, segmentation=False, obb=False):
        super().__init__()
        self.segmentation = segmentation
        self.obb = obb
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        batch = x.shape[0]
        signal = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
        boxes = signal.reshape(batch, 1, 1).expand(batch, 3, 4)
        logits = signal.reshape(batch, 1, 1).expand(batch, 3, 2)
        if self.segmentation:
            masks = signal.expand(batch, 3, 8, 8)
            return boxes, logits, masks
        if self.obb:
            angles = signal.reshape(batch, 1, 1).expand(batch, 3, 1)
            return boxes, logits, angles
        return boxes, logits


class _TinyRTDETRExport(nn.Module):
    """Small RT-DETR-shaped module that returns the native output dict."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        batch = x.shape[0]
        signal = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
        logits = signal.reshape(batch, 1, 1).expand(batch, 3, 2)
        boxes = signal.reshape(batch, 1, 1).expand(batch, 3, 4)
        return {"pred_logits": logits, "pred_boxes": boxes}


def _make_wrapper(nb_classes=4, model_name="TESTYOLO", size="s", input_size=32):
    """Build a mock BaseModel-like wrapper around _TinyModel."""
    wrapper = MagicMock()
    wrapper.model = _TinyModel()
    wrapper.model.eval()
    wrapper.size = size
    wrapper.nb_classes = nb_classes
    wrapper.names = {i: f"class_{i}" for i in range(nb_classes)}
    wrapper.device = torch.device("cpu")
    wrapper.model_path = None
    wrapper.FILENAME_PREFIX = ""
    wrapper._get_model_name.return_value = model_name
    wrapper._get_input_size.return_value = input_size
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExporterFormats:
    def test_expected_keys(self):
        assert "onnx" in BaseExporter._registry
        assert "torchscript" in BaseExporter._registry
        assert "executorch" in BaseExporter._registry
        assert "tensorrt" in BaseExporter._registry
        assert "openvino" in BaseExporter._registry
        assert "paddle" in BaseExporter._registry
        assert "ncnn" in BaseExporter._registry
        assert "tflite" in BaseExporter._registry

    def test_suffix_present(self):
        for cls in BaseExporter._registry.values():
            assert cls.suffix.startswith(".") or cls.suffix.startswith("_")

    def test_subclass_attributes(self):
        assert OnnxExporter.suffix == ".onnx"
        assert OnnxExporter.supports_int8 is True
        assert OnnxExporter.supports_embedded_nms is True
        assert CoreMLExporter.supports_embedded_nms is True
        assert TensorRTExporter.requires_onnx is True
        assert TorchScriptExporter.apply_model_half is True
        assert TorchScriptExporter.supports_embedded_nms is False
        assert ExecuTorchExporter.suffix == ".pte"
        assert ExecuTorchExporter.supports_fp16 is False
        assert ExecuTorchExporter.supports_int8 is False
        assert NcnnExporter.supports_int8 is False
        assert TFLiteExporter.requires_onnx is True
        assert TFLiteExporter.supports_fp16 is False
        assert PaddleExporter.suffix == "_paddle"
        assert PaddleExporter.requires_onnx is True
        assert PaddleExporter.supports_fp16 is False

    def test_unsupported_exporter_rejects_embedded_nms(self):
        exporter = TorchScriptExporter(_make_wrapper())

        with pytest.raises(NotImplementedError, match="TORCHSCRIPT embedded NMS"):
            exporter._preflight(half=False, int8=False, data=None, nms=True)

    def test_coreml_exporter_accepts_embedded_nms_preflight(self):
        wrapper = _make_wrapper(model_name="yolo9")
        wrapper.task = "detect"
        exporter = CoreMLExporter(wrapper)

        exporter._preflight(half=False, int8=False, data=None, nms=True)

    def test_coreml_embedded_nms_preflight_rejects_rtdetr(self):
        wrapper = _make_wrapper(model_name="rtdetr")
        wrapper.task = "detect"
        exporter = CoreMLExporter(wrapper)

        with pytest.raises(NotImplementedError, match="YOLOX and YOLO9"):
            exporter._preflight(half=False, int8=False, data=None, nms=True)

    def test_coreml_embedded_nms_preflight_rejects_yolo9_segment(self):
        wrapper = _make_wrapper(model_name="yolo9")
        wrapper.task = "segment"
        exporter = CoreMLExporter(wrapper)

        with pytest.raises(NotImplementedError, match="YOLO9 detection"):
            exporter._preflight(half=False, int8=False, data=None, nms=True)

    def test_export_rejects_yolo9_segment(self):
        wrapper = _make_wrapper(model_name="yolo9")
        wrapper.task = "segment"
        exporter = OnnxExporter(wrapper)

        with pytest.raises(NotImplementedError, match="YOLO9 segmentation export"):
            exporter(output_path="unused.onnx")

    def test_direct_onnx_export_rejects_yolo9_segment_metadata(self):
        wrapper = _make_wrapper(model_name="yolo9")

        with pytest.raises(NotImplementedError, match="YOLO9 segmentation ONNX export"):
            export_onnx(
                wrapper.model,
                torch.zeros(1, 3, 32, 32),
                output_path="unused.onnx",
                opset=13,
                simplify=False,
                dynamic=False,
                half=False,
                metadata={
                    "model_family": "yolo9",
                    "task": "segment",
                    "segmentation": "true",
                },
            )

    def test_coreml_embedded_nms_preflight_rejects_max_det(self):
        wrapper = _make_wrapper(model_name="yolo9")
        wrapper.task = "detect"
        exporter = CoreMLExporter(wrapper)

        with pytest.raises(NotImplementedError, match="does not support max_det"):
            exporter._preflight(half=False, int8=False, data=None, nms=True, max_det=12)

    def test_onnx_embedded_nms_preflight_rejects_non_yolo9_detect(self):
        exporter = OnnxExporter(_make_wrapper(model_name="yolox"))

        with pytest.raises(NotImplementedError, match="YOLO9 detection"):
            exporter._preflight(half=False, int8=False, data=None, nms=True)

    def test_unwired_point_export_fails_before_artifact_creation(self, tmp_path):
        wrapper = _make_wrapper(model_name="unwired")
        wrapper.task = "point"

        with pytest.raises(NotImplementedError, match="not wired.*point"):
            OnnxExporter(wrapper)(output_path=str(tmp_path / "point.onnx"))

        assert not (tmp_path / "point.onnx").exists()

    def test_unwired_semantic_export_fails_before_artifact_creation(self, tmp_path):
        wrapper = _make_wrapper(model_name="unwired")
        wrapper.task = "semantic"

        with pytest.raises(NotImplementedError, match="not wired.*semantic"):
            OnnxExporter(wrapper)(output_path=str(tmp_path / "sem.onnx"))

        assert not (tmp_path / "sem.onnx").exists()

    def test_metadata_includes_task_contract(self):
        wrapper = _make_wrapper()
        wrapper.task = "segment"
        wrapper.SUPPORTED_TASKS = ("detect", "segment")
        wrapper.DEFAULT_TASK = "detect"

        metadata = TensorRTExporter(wrapper)._build_metadata(
            precision="fp32",
            dynamic=False,
            onnx_path=None,
        )

        assert metadata["task"] == "segment"
        assert metadata["supported_tasks"] == ["detect", "segment"]
        assert metadata["default_task"] == "detect"

    def test_metadata_includes_obb_task_contract(self):
        wrapper = _make_wrapper()
        wrapper.task = "obb"
        wrapper.SUPPORTED_TASKS = ("detect", "segment", "obb")
        wrapper.DEFAULT_TASK = "detect"

        metadata = TensorRTExporter(wrapper)._build_metadata(
            precision="fp32",
            dynamic=False,
            onnx_path=None,
        )
        onnx_metadata = OnnxExporter(wrapper)._build_onnx_metadata(
            dynamic=False,
            half=False,
        )

        assert metadata["task"] == "obb"
        assert metadata["obb"] is True
        assert metadata["supported_tasks"] == ["detect", "segment", "obb"]
        assert onnx_metadata["task"] == "obb"
        assert onnx_metadata["obb"] == "true"

    def test_rfdetr_export_metadata_is_single_task(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.task = "segment"
        wrapper.SUPPORTED_TASKS = ("detect", "segment")
        wrapper.DEFAULT_TASK = "detect"

        metadata = TensorRTExporter(wrapper)._build_metadata(
            precision="fp32",
            dynamic=False,
            onnx_path=None,
        )

        assert metadata["task"] == "segment"
        assert metadata["supported_tasks"] == ["segment"]
        assert metadata["default_task"] == "segment"

    def test_rfdetr_pose_onnx_metadata_preserves_grouppose_schema(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.task = "pose"
        wrapper.SUPPORTED_TASKS = ("detect", "pose")
        wrapper.DEFAULT_TASK = "detect"
        wrapper.num_keypoints = 17
        wrapper.keypoint_dim = 3
        wrapper.num_keypoints_per_class = [0, 17, 4]

        metadata = OnnxExporter(wrapper)._build_onnx_metadata(
            dynamic=False,
            half=False,
        )

        assert metadata["task"] == "pose"
        assert metadata["num_keypoints"] == "17"
        assert metadata["keypoint_dim"] == "8"
        assert json.loads(metadata["num_keypoints_per_class"]) == [0, 17, 4]

    def test_ec_pose_onnx_metadata_describes_exported_xy_keypoints(self):
        wrapper = _make_wrapper(model_name="ec")
        wrapper.task = "pose"
        wrapper.SUPPORTED_TASKS = ("detect", "pose")
        wrapper.DEFAULT_TASK = "detect"
        wrapper.num_keypoints = 17
        wrapper.keypoint_dim = 3

        metadata = OnnxExporter(wrapper)._build_onnx_metadata(
            dynamic=False,
            half=False,
        )

        assert metadata["task"] == "pose"
        assert metadata["num_keypoints"] == "17"
        assert metadata["keypoint_dim"] == "2"

    def test_tensorrt_export_forwards_dynamic_batch_profile(
        self, monkeypatch, tmp_path
    ):
        wrapper = _make_wrapper(model_name="rfdetr")
        captured = {}

        def fake_export_tensorrt(**kwargs):
            captured.update(kwargs)
            return str(tmp_path / "model.engine")

        monkeypatch.setattr(
            "libreyolo.export.tensorrt.export_tensorrt",
            fake_export_tensorrt,
        )

        metadata = {"model_family": "rfdetr"}
        TensorRTExporter(wrapper)._export(
            wrapper.model,
            torch.zeros(1, 3, 32, 32),
            output_path=str(tmp_path / "model.engine"),
            precision="fp16",
            metadata=metadata,
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            half=True,
            int8=False,
            dynamic=True,
            verbose=False,
            min_batch=2,
            opt_batch=4,
            max_batch=16,
        )

        assert captured["min_batch"] == 2
        assert captured["opt_batch"] == 4
        assert captured["max_batch"] == 16
        assert captured["metadata"]["trt_min_batch"] == 2
        assert captured["metadata"]["trt_opt_batch"] == 4
        assert captured["metadata"]["trt_max_batch"] == 16
        assert "trt_max_batch" not in metadata

    def test_rfdetr_export_defaults_to_cpu(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.device = torch.device("cuda")

        imgsz, device, output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device=None,
            half=False,
            int8=False,
        )

        assert imgsz == (32, 32)
        assert device == torch.device("cpu")
        assert output_path.endswith(".onnx")

    def test_rfdetr_export_auto_device_defaults_to_cpu(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.device = torch.device("cuda")

        _imgsz, device, _output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device="auto",
            half=False,
            int8=False,
        )

        assert device == torch.device("cpu")

    @pytest.mark.parametrize("imgsz", [384, (384, 384)])
    def test_rfdetr_obb_executorch_accepts_native_canvas_forms(
        self, monkeypatch, imgsz
    ):
        from libreyolo.models.base.model import BaseModel
        from libreyolo.models.rfdetr.model import LibreRFDETR

        model = object.__new__(LibreRFDETR)
        model.task = "obb"
        model._validate_imgsz = lambda value, **_: value
        model._get_input_size = lambda: 384
        captured = {}

        def fake_export(self, format="onnx", **kwargs):
            captured.update(format=format, **kwargs)
            return "model.pte"

        monkeypatch.setattr(BaseModel, "export", fake_export)

        assert model.export("executorch", imgsz=imgsz) == "model.pte"
        assert captured["imgsz"] == imgsz

    @pytest.mark.parametrize("device_arg", ["0", 0])
    def test_export_normalizes_bare_numeric_device(self, device_arg):
        wrapper = _make_wrapper(model_name="yolo9")
        wrapper.device = torch.device("cpu")

        _imgsz, device, _output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device=device_arg,
            half=False,
            int8=False,
        )

        assert device == torch.device("cuda:0")

    @pytest.mark.parametrize("family", ["rfdetr", "rtdetr", "rtdetrv2"])
    def test_detr_export_auto_opset_is_17(self, monkeypatch, tmp_path, family):
        captured = {}
        wrapper = _make_wrapper(model_name=family)
        if family == "rfdetr":
            wrapper.model = _TinyRFDETRExport(segmentation=False)
        elif family in {"rtdetr", "rtdetrv2"}:
            wrapper.model = _TinyRTDETRExport()
        wrapper.task = "detect"
        wrapper.SUPPORTED_TASKS = ("detect",)
        wrapper.DEFAULT_TASK = "detect"

        def fake_export_onnx(_nn_model, _dummy, **kwargs):
            captured.update(kwargs)
            if family in {"rtdetr", "rtdetrv2"}:
                output = _nn_model(_dummy)
                captured["output_is_tuple"] = isinstance(output, tuple)
                captured["output_len"] = len(output)
            Path(kwargs["output_path"]).write_bytes(b"onnx")
            return kwargs["output_path"]

        monkeypatch.setattr("libreyolo.export.exporter.export_onnx", fake_export_onnx)
        output_path = tmp_path / f"{family}.onnx"

        exported = OnnxExporter(wrapper)(
            output_path=str(output_path),
            simplify=False,
            dynamic=False,
            device="cpu",
        )

        assert exported == str(output_path)
        assert captured["opset"] == 17
        if family in {"rtdetr", "rtdetrv2"}:
            assert captured["output_is_tuple"] is True
            assert captured["output_len"] == 2

    @pytest.mark.parametrize(
        ("task", "segmentation", "obb", "expected_outputs"),
        [
            ("detect", False, False, ["dets", "labels"]),
            ("segment", True, False, ["dets", "labels", "masks"]),
            ("obb", False, True, ["dets", "labels", "angles"]),
        ],
    )
    def test_rfdetr_onnx_uses_upstream_io_names(
        self, tmp_path, task, segmentation, obb, expected_outputs
    ):
        onnx = pytest.importorskip("onnx")
        output_path = tmp_path / "rfdetr.onnx"

        export_onnx(
            _TinyRFDETRExport(segmentation=segmentation, obb=obb),
            torch.zeros(1, 3, 32, 32),
            output_path=str(output_path),
            opset=17,
            simplify=False,
            dynamic=False,
            half=False,
            metadata={
                "model_family": "rfdetr",
                "task": task,
                "segmentation": "true" if segmentation else "false",
            },
        )

        proto = onnx.load(output_path)
        assert [i.name for i in proto.graph.input] == ["input"]
        assert [o.name for o in proto.graph.output] == expected_outputs

    @pytest.mark.parametrize("family", ["rtdetr", "rtdetrv2", "rtdetrv4"])
    def test_rtdetr_onnx_uses_detr_io_names(self, tmp_path, family):
        onnx = pytest.importorskip("onnx")
        wrapper = _make_wrapper(model_name=family, input_size=32)
        wrapper.model = _TinyRTDETRExport()
        wrapper.task = "detect"
        wrapper.SUPPORTED_TASKS = ("detect",)
        wrapper.DEFAULT_TASK = "detect"
        output_path = tmp_path / f"{family}.onnx"

        OnnxExporter(wrapper)(
            output_path=str(output_path),
            simplify=False,
            dynamic=False,
            device="cpu",
        )

        proto = onnx.load(output_path)
        assert [i.name for i in proto.graph.input] == ["images"]
        assert [o.name for o in proto.graph.output] == ["pred_logits", "pred_boxes"]

    def test_onnx_metadata_uses_export_imgsz_override(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        wrapper = _make_wrapper(model_name="TESTYOLO", input_size=32)
        output_path = tmp_path / "custom_imgsz.onnx"

        OnnxExporter(wrapper)(
            output_path=str(output_path),
            imgsz=48,
            simplify=False,
            dynamic=False,
        )

        proto = onnx.load(output_path)
        meta = {p.key: p.value for p in proto.metadata_props}
        assert meta["imgsz"] == "48"

        from libreyolo.backends.onnx import OnnxBackend

        assert OnnxBackend._read_onnx_metadata(str(output_path), 4)[-1] == 48

    def test_onnx_backend_reads_runtime_metadata_without_onnx_load(self):
        from libreyolo.backends.onnx import OnnxBackend

        metadata = {
            "model_family": "yolo9",
            "task": "detect",
            "nb_classes": "1",
            "imgsz": "64",
            "nms": "true",
            "nms_conf": "0.25",
            "nms_iou": "0.45",
            "max_det": "300",
            "nms_raw_output": "true",
        }

        parsed = OnnxBackend._read_onnx_metadata(
            "metadata-from-onnxruntime.onnx",
            80,
            runtime_metadata=metadata,
        )

        assert parsed[0] == "yolo9"
        assert parsed[2] == "detect"
        assert parsed[5] == {0: "class_0"}
        assert parsed[6] is True
        assert parsed[7] == 64

    def test_onnx_backend_reads_rectangular_static_input_imgsz(self):
        from libreyolo.backends.onnx import OnnxBackend

        assert OnnxBackend._read_static_input_imgsz([1, 3, 32, 64]) == (32, 64)
        assert OnnxBackend._read_static_input_imgsz([1, 3, -1, -1]) is None

    @pytest.mark.parametrize(
        "exporter_cls",
        [
            OnnxExporter,
            TorchScriptExporter,
            TensorRTExporter,
            OpenVINOExporter,
            NcnnExporter,
            TFLiteExporter,
            CoreMLExporter,
        ],
    )
    def test_rectangular_imgsz_supported_for_yolo9_export_formats(self, exporter_cls):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)

        imgsz, device, _output_path = exporter_cls(wrapper)._resolve_params(
            output_path=None,
            imgsz=(32, 64),
            device="cpu",
            half=False,
            int8=False,
        )

        assert imgsz == (32, 64)
        assert device == torch.device("cpu")

    def test_rectangular_imgsz_is_limited_to_yolo9_family(self):
        wrapper = _make_wrapper(model_name="yolox", input_size=32)

        with pytest.raises(NotImplementedError, match="YOLO9-family"):
            OnnxExporter(wrapper)._resolve_params(
                output_path=None,
                imgsz=(32, 64),
                device="cpu",
                half=False,
                int8=False,
            )

    @pytest.mark.parametrize(
        ("model_name", "exporter_cls"),
        [
            ("yolox", OnnxExporter),
            ("rtmdet", TorchScriptExporter),
            ("yolo9", ExecuTorchExporter),
            ("yolo9", PaddleExporter),
        ],
    )
    def test_restored_rect_imgsz_falls_back_to_square_by_default(
        self, model_name, exporter_cls, caplog
    ):
        # #899: a square-native fine-tune trained at (h, w) reloads with that
        # size; a family/format pair without rectangular export keeps exporting
        # by default, at max(h, w) instead of raising.
        wrapper = _make_wrapper(model_name=model_name, input_size=(32, 64))
        wrapper._get_task_input_sizes.return_value = {"s": 640}

        with caplog.at_level("WARNING", logger="libreyolo.export.exporter"):
            imgsz, _device, _output_path = exporter_cls(wrapper)._resolve_params(
                output_path=None,
                imgsz=None,
                device="cpu",
                half=False,
                int8=False,
            )

        assert imgsz == (64, 64)
        assert "exporting at the square imgsz=64" in caplog.text

    def test_restored_rect_imgsz_is_kept_where_export_supports_it(self):
        wrapper = _make_wrapper(model_name="yolo9", input_size=(32, 64))
        wrapper._get_task_input_sizes.return_value = {"s": 640}

        imgsz, _device, _output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device="cpu",
            half=False,
            int8=False,
        )

        assert imgsz == (32, 64)

    def test_natively_rect_family_keeps_rejecting_unsupported_format(self):
        wrapper = _make_wrapper(model_name="unet", input_size=(32, 64))
        wrapper._get_task_input_sizes.return_value = {"s": (32, 64)}

        with pytest.raises(NotImplementedError, match="not validated for format"):
            PaddleExporter(wrapper)._resolve_params(
                output_path=None,
                imgsz=None,
                device="cpu",
                half=False,
                int8=False,
            )

    def test_semantic_export_rejects_unaligned_imgsz(self):
        wrapper = _make_wrapper(model_name="segformer", input_size=32)
        wrapper.task = "semantic"
        wrapper.semantic_imgsz_divisor = 32

        with pytest.raises(
            ValueError,
            match=r"Semantic export imgsz must be divisible.*stride 32",
        ):
            OnnxExporter(wrapper)._resolve_params(
                output_path=None,
                imgsz=100,
                device="cpu",
                half=False,
                int8=False,
            )

    @pytest.mark.parametrize(
        "family",
        ["dfine", "deim", "deimv2", "ec", "rfdetr", "rtdetr", "rtdetrv2", "rtdetrv4"],
    )
    def test_rectangular_imgsz_rejected_for_fixed_square_families(self, family):
        wrapper = _make_wrapper(model_name=family, input_size=32)

        with pytest.raises(NotImplementedError, match="fixed square"):
            OnnxExporter(wrapper)._resolve_params(
                output_path=None,
                imgsz=(32, 64),
                device="cpu",
                half=False,
                int8=False,
            )

    def test_deimv2_accepts_non_native_square_imgsz(self):
        wrapper = _make_wrapper(model_name="deimv2", input_size=320)

        imgsz, _, _ = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=(640, 640),
            device="cpu",
            half=False,
            int8=False,
        )
        assert imgsz == (640, 640)

    def test_rectangular_onnx_export_writes_shape_metadata_without_onnx(
        self, monkeypatch, tmp_path
    ):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        output_path = tmp_path / "rectangular.onnx"
        captured = {}

        def fake_export_onnx(_nn_model, dummy, **kwargs):
            captured["dummy_shape"] = tuple(dummy.shape)
            captured["metadata"] = kwargs["metadata"]
            return kwargs["output_path"]

        monkeypatch.setattr("libreyolo.export.exporter.export_onnx", fake_export_onnx)

        exported = OnnxExporter(wrapper)(
            output_path=str(output_path),
            imgsz=(16, 32),
            simplify=False,
            dynamic=False,
            device="cpu",
        )

        assert exported == str(output_path)
        assert captured["dummy_shape"] == (1, 3, 16, 32)
        assert captured["metadata"]["imgsz"] == "32"
        assert captured["metadata"]["imgsz_h"] == "16"
        assert captured["metadata"]["imgsz_w"] == "32"

    def test_onnx_int8_missing_data_uses_default_calibration(
        self, monkeypatch, tmp_path, caplog
    ):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        wrapper.task = "detect"
        exporter = OnnxExporter(wrapper)
        output_path = tmp_path / "model_int8.onnx"
        captured = {}

        monkeypatch.setattr(exporter, "_preflight", lambda **kwargs: None)

        def fake_load_calibration(
            data,
            imgsz,
            batch,
            fraction,
            allow_download_scripts=False,
        ):
            captured["data"] = data
            captured["imgsz"] = imgsz
            captured["batch"] = batch
            return object()

        def fake_export(nn_model, dummy, *, output_path, calibration_data, **kwargs):
            captured["dummy_dtype"] = dummy.dtype
            captured["param_dtype"] = next(nn_model.parameters()).dtype
            captured["calibration_data"] = calibration_data
            captured["half"] = kwargs["half"]
            captured["int8"] = kwargs["int8"]
            return output_path

        monkeypatch.setattr(exporter, "_load_calibration", fake_load_calibration)
        monkeypatch.setattr(exporter, "_export", fake_export)

        with caplog.at_level("WARNING", logger="libreyolo.export.exporter"):
            result = exporter(
                output_path=str(output_path),
                int8=True,
                half=True,
                batch=2,
                device="cpu",
                simplify=False,
                dynamic=False,
            )

        assert result == str(output_path)
        assert captured["data"] == "coco8.yaml"
        assert captured["imgsz"] == (32, 32)
        assert captured["batch"] == 2
        assert captured["dummy_dtype"] == torch.float32
        assert captured["param_dtype"] == torch.float32
        assert captured["half"] is False
        assert captured["int8"] is True
        assert captured["calibration_data"] is not None
        assert "8-image fallback is not representative" in caplog.text

    @pytest.mark.parametrize(
        ("family", "task"),
        [("rfdetr", "detect"), ("yolo9", "segment"), ("yolo9_e2e", "detect")],
    )
    def test_onnx_int8_scope_is_yolo9_detect_only(self, family, task):
        wrapper = _make_wrapper(model_name=family, input_size=32)
        wrapper.task = task

        with pytest.raises(NotImplementedError, match="YOLO9 detection"):
            OnnxExporter(wrapper)._preflight(
                half=False,
                int8=True,
                data="data.yaml",
            )

    def test_onnx_int8_export_uses_fp32_temp_then_quantizes(
        self, monkeypatch, tmp_path
    ):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        wrapper.task = "detect"
        exporter = OnnxExporter(wrapper)
        output_path = tmp_path / "model_int8.onnx"
        captured = {}

        def fake_export_onnx(_nn_model, _dummy, **kwargs):
            captured["fp32_export"] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"fp32")
            return kwargs["output_path"]

        def fake_quantize_onnx_int8(fp32_path, quant_output_path, **kwargs):
            captured["quantize"] = {
                "fp32_path": fp32_path,
                "output_path": quant_output_path,
                **kwargs,
            }
            Path(quant_output_path).write_bytes(b"int8")
            return quant_output_path

        monkeypatch.setattr(exporter_module, "export_onnx", fake_export_onnx)
        monkeypatch.setattr(
            exporter_module, "quantize_onnx_int8", fake_quantize_onnx_int8
        )

        result = exporter._export(
            wrapper.model,
            torch.zeros(1, 3, 32, 32),
            output_path=str(output_path),
            metadata={},
            calibration_data=object(),
            half=False,
            int8=True,
            dynamic=False,
            opset=13,
            simplify=False,
        )

        assert result == str(output_path)
        assert Path(captured["fp32_export"]["output_path"]).name == "model_fp32.onnx"
        assert captured["fp32_export"]["half"] is False
        assert captured["quantize"]["output_path"] == str(output_path)
        assert captured["quantize"]["metadata"]["precision"] == "int8"
        assert captured["quantize"]["metadata"]["half"] == "False"

    def test_onnx_backend_reads_rectangular_metadata(self, tmp_path):
        pytest.importorskip("onnx")
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        output_path = tmp_path / "rectangular.onnx"

        OnnxExporter(wrapper)(
            output_path=str(output_path),
            imgsz=(16, 32),
            simplify=False,
            dynamic=False,
            device="cpu",
        )

        from libreyolo.backends.onnx import OnnxBackend

        assert OnnxBackend._read_onnx_metadata(str(output_path), 4)[-1] == (16, 32)

    @pytest.mark.parametrize(
        "metadata,error_match",
        [
            (
                {"model_family": "yolo9", "imgsz": "32", "imgsz_h": "16"},
                "both imgsz_h and imgsz_w",
            ),
            (
                {
                    "model_family": "yolo9",
                    "imgsz": "32",
                    "imgsz_h": "bad",
                    "imgsz_w": "32",
                },
                "invalid imgsz_h/imgsz_w",
            ),
        ],
    )
    def test_onnx_metadata_rejects_malformed_rectangular_imgsz(
        self, tmp_path, metadata, error_match
    ):
        pytest.importorskip("onnx")
        output_path = tmp_path / "malformed_rectangular.onnx"

        export_onnx(
            _TinyModel(),
            torch.zeros(1, 3, 16, 32),
            output_path=str(output_path),
            opset=13,
            simplify=False,
            dynamic=True,
            half=False,
            metadata=metadata,
        )

        from libreyolo.backends.onnx import OnnxBackend

        with pytest.raises(ValueError, match=error_match):
            OnnxBackend._read_onnx_metadata(str(output_path), 4)

    def test_onnx_backend_prefers_rectangular_static_shape_over_legacy_scalar(
        self, tmp_path
    ):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
        output_path = tmp_path / "rectangular_stale_scalar.onnx"

        export_onnx(
            _TinyModel(),
            torch.zeros(1, 3, 16, 32),
            output_path=str(output_path),
            opset=13,
            simplify=False,
            dynamic=False,
            half=False,
            metadata={
                "model_family": "yolo9",
                "imgsz": "32",
                "nc": "4",
            },
        )

        from libreyolo.backends.onnx import OnnxBackend

        backend = OnnxBackend(str(output_path), nb_classes=4)
        assert backend.imgsz == (16, 32)

    def test_torchscript_backend_reads_rectangular_metadata(self, tmp_path):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        output_path = tmp_path / "rectangular.torchscript"

        TorchScriptExporter(wrapper)(
            output_path=str(output_path),
            imgsz=(16, 32),
            device="cpu",
        )

        from libreyolo.backends.torchscript import TorchScriptBackend

        backend = TorchScriptBackend(str(output_path), device="cpu")
        assert backend.imgsz == (16, 32)

    def test_torchscript_backend_preserves_classification_preprocessing(self, tmp_path):
        wrapper = _make_wrapper(model_name="resnet", input_size=32)
        wrapper.task = "classify"
        wrapper.SUPPORTED_TASKS = ("classify",)
        wrapper.DEFAULT_TASK = "classify"
        wrapper.crop_pct = 0.95
        wrapper.interpolation = "bicubic"
        output_path = tmp_path / "classifier.torchscript"

        TorchScriptExporter(wrapper)(
            output_path=str(output_path),
            imgsz=32,
            device="cpu",
        )

        from libreyolo.backends.torchscript import TorchScriptBackend

        backend = TorchScriptBackend(str(output_path), device="cpu")
        assert backend.task == "classify"
        assert backend.crop_pct == pytest.approx(0.95)
        assert backend.interpolation == "bicubic"

    def test_rectangular_int8_calibration_receives_tuple_imgsz(
        self, monkeypatch, tmp_path
    ):
        wrapper = _make_wrapper(model_name="yolo9", input_size=32)
        exporter = TensorRTExporter(wrapper)
        output_path = tmp_path / "model.engine"
        captured = {}

        monkeypatch.setattr(exporter, "_preflight", lambda **kwargs: None)
        monkeypatch.setattr(
            exporter,
            "_export_intermediate_onnx",
            lambda *args, **kwargs: str(tmp_path / "model.onnx"),
        )

        def fake_load_calibration(
            data,
            imgsz,
            batch,
            fraction,
            allow_download_scripts=False,
        ):
            captured["imgsz"] = imgsz
            captured["batch"] = batch
            return object()

        def fake_export(nn_model, dummy, *, output_path, calibration_data, **kwargs):
            captured["dummy_shape"] = tuple(dummy.shape)
            captured["calibration_data"] = calibration_data
            return output_path

        monkeypatch.setattr(exporter, "_load_calibration", fake_load_calibration)
        monkeypatch.setattr(exporter, "_export", fake_export)

        result = exporter(
            output_path=str(output_path),
            imgsz=(16, 32),
            int8=True,
            data="data.yaml",
            batch=2,
            device="cpu",
            simplify=False,
            dynamic=False,
        )

        assert result == str(output_path)
        assert captured["imgsz"] == (16, 32)
        assert captured["batch"] == 2
        assert captured["dummy_shape"] == (2, 3, 16, 32)
        assert captured["calibration_data"] is not None


class TestExporterValidation:
    def test_invalid_format_raises(self):
        wrapper = _make_wrapper()
        with pytest.raises(ValueError, match="Unsupported export format"):
            BaseExporter.create("badformat", wrapper)

    def test_invalid_format_case_insensitive(self):
        wrapper = _make_wrapper()
        # Should NOT raise — format names are lowered via create()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = BaseExporter.create("TORCHSCRIPT", wrapper)
            path = exporter(
                output_path=str(Path(tmpdir) / "model.torchscript"),
            )
            assert Path(path).exists()

    def test_litert_alias_resolves_to_tflite(self):
        wrapper = _make_wrapper()
        exporter = BaseExporter.create("litert", wrapper)
        assert exporter.format_name == "tflite"
        assert exporter.suffix == ".tflite"


class TestExportLoraOrdering:
    """Live LoRA adapters are folded into dense weights for export.

    That merge is destructive (adapters are folded and removed), so every
    request rejection — unknown format, quantized recipe/format gates, the
    exporter's own option preflight — must raise *before* it happens.
    Otherwise a rejected export silently strips the adapters from a model
    the user may keep training (v1.4.0 release blocker).
    """

    @staticmethod
    def _record_lora_merges(monkeypatch, on_merge):
        from libreyolo.training import lora as lora_helpers

        monkeypatch.setattr(lora_helpers, "module_has_lora", lambda m: True)
        monkeypatch.setattr(lora_helpers, "merge_lora_adapters", on_merge)

    def test_invalid_format_raises_before_lora_merge(self, monkeypatch):
        from types import SimpleNamespace

        from libreyolo.models.base.model import BaseModel

        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        stub = SimpleNamespace(model=nn.Linear(2, 2))
        with pytest.raises(ValueError, match="Unsupported export format"):
            BaseModel.export(stub, format="not-a-format")
        assert merged == []  # adapters untouched by the failed export

    def test_preflight_rejection_leaves_lora_untouched(self, monkeypatch):
        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        # Embedded-NMS ONNX export is YOLO9-only; the preflight rejection
        # must fire before the destructive adapter merge.
        wrapper = _make_wrapper()
        with pytest.raises(NotImplementedError, match="NMS"):
            OnnxExporter(wrapper)(nms=True)
        assert merged == []

    def test_quantized_export_rejection_leaves_lora_untouched(self, monkeypatch):
        from types import SimpleNamespace

        from libreyolo.models.base.model import BaseModel
        from libreyolo.quant import QuantizationError

        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        stub = SimpleNamespace(model=nn.Linear(2, 2))
        stub._quant_manifest = {"recipe": "int8"}
        # int8 recipe only exports to onnx/pt; torchscript must be rejected
        # before the destructive adapter merge.
        with pytest.raises(QuantizationError, match="format='onnx'"):
            BaseModel.export(stub, format="torchscript")
        assert merged == []

    def test_pt_remainder_rejection_leaves_lora_untouched(self, monkeypatch):
        from types import SimpleNamespace

        from libreyolo.models.base.model import BaseModel
        from libreyolo.quant import QuantizationError

        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        stub = SimpleNamespace(model=nn.Linear(2, 2))
        stub._quant_manifest = {"recipe": "int8"}
        with pytest.raises(QuantizationError, match="remainder"):
            BaseModel.export(stub, format="pt", remainder="int4")
        assert merged == []

    def test_quantized_pt_export_merges_on_a_copy(self, monkeypatch, tmp_path):
        from libreyolo.models.base.model import BaseModel

        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        wrapper = _make_wrapper()
        wrapper.task = "detect"
        wrapper.model_path = None
        wrapper._quant_manifest = {"recipe": "int8", "state": "prepared"}
        out = BaseModel.export(wrapper, format="pt", out=str(tmp_path / "q.pt"))
        assert Path(out).exists()
        # The fold ran exactly once, on the checkpoint's deep copy — the
        # live model keeps its adapters trainable.
        assert len(merged) == 1
        assert merged[0] is not wrapper.model

    def test_quantized_onnx_rejection_keeps_finalized_state(self, monkeypatch):
        from libreyolo.quant.api import quantized_export

        merged = []
        self._record_lora_merges(monkeypatch, merged.append)

        wrapper = _make_wrapper()  # not yolo9 -> embedded NMS is rejected
        wrapper._quant_manifest = {
            "recipe": "int8",
            "state": "finalized",
            "calibrated": True,
        }
        with pytest.raises(NotImplementedError, match="NMS"):
            quantized_export(wrapper, format="onnx", nms=True)
        # reprepare_model never ran: the packed finalized state is intact.
        assert wrapper._quant_manifest["state"] == "finalized"
        assert merged == []

    def test_successful_export_merges_after_preflight(self, monkeypatch, tmp_path):
        events = []
        self._record_lora_merges(monkeypatch, lambda m: events.append("merge"))

        original_preflight = BaseExporter._preflight

        def spy_preflight(self, **kwargs):
            events.append("preflight")
            return original_preflight(self, **kwargs)

        monkeypatch.setattr(BaseExporter, "_preflight", spy_preflight)

        wrapper = _make_wrapper()
        path = TorchScriptExporter(wrapper)(
            output_path=str(tmp_path / "model.torchscript")
        )
        assert Path(path).exists()
        assert events == ["preflight", "merge"]


class TestOutputPathGeneration:
    def test_auto_path_torchscript(self):
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        wrapper.model_path = "weights/LibreYOLO9t.pt"
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter()
                assert path == str(Path("weights") / "LibreYOLO9t.torchscript")
                assert Path(path).exists()
            finally:
                os.chdir(orig)

    def test_auto_path_prefers_weight_file_stem(self):
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        wrapper.FILENAME_PREFIX = "LibreYOLO9"
        wrapper.model_path = "runs/train/best.pt"
        exporter = OnnxExporter(wrapper)

        assert exporter._auto_output_path(half=False, int8=False) == str(
            Path("weights") / "best.onnx"
        )

    def test_auto_path_includes_segmentation_task(self):
        wrapper = _make_wrapper(model_name="rfdetr", size="n")
        wrapper.FILENAME_PREFIX = "LibreRFDETR"
        wrapper.task = "segment"
        exporter = OnnxExporter(wrapper)
        assert exporter._auto_output_path(half=False, int8=False) == str(
            Path("weights") / "LibreRFDETRn-seg.onnx"
        )
        assert exporter._auto_output_path(half=True, int8=False) == str(
            Path("weights") / "LibreRFDETRn-seg_fp16.onnx"
        )

    def test_auto_path_includes_obb_task(self):
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        wrapper.FILENAME_PREFIX = "LibreYOLO9"
        wrapper.task = "obb"
        exporter = OnnxExporter(wrapper)

        assert exporter._auto_output_path(half=False, int8=False) == str(
            Path("weights") / "LibreYOLO9t-obb.onnx"
        )
        assert exporter._auto_output_path(half=True, int8=False) == str(
            Path("weights") / "LibreYOLO9t-obb_fp16.onnx"
        )

    def test_auto_path_includes_rfdetr_obb_task(self):
        wrapper = _make_wrapper(model_name="rfdetr", size="n")
        wrapper.FILENAME_PREFIX = "LibreRFDETR"
        wrapper.task = "obb"
        exporter = OnnxExporter(wrapper)

        assert exporter._auto_output_path(half=False, int8=False) == str(
            Path("weights") / "LibreRFDETRn-obb.onnx"
        )
        assert exporter._auto_output_path(half=True, int8=False) == str(
            Path("weights") / "LibreRFDETRn-obb_fp16.onnx"
        )

    def test_auto_path_rejects_unknown_task(self):
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        wrapper.FILENAME_PREFIX = "LibreYOLO9"
        wrapper.task = "bad-task"
        exporter = OnnxExporter(wrapper)

        with pytest.raises(ValueError, match="Unsupported task for auto output naming"):
            exporter._auto_output_path(half=False, int8=False)

    def test_explicit_path(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "custom.torchscript")
            path = exporter(output_path=out)
            assert path == out
            assert Path(out).exists()


class TestTorchScriptExport:
    def test_basic_torchscript(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "model.torchscript")
            path = exporter(output_path=out)
            assert Path(path).exists()

            # Verify the file is loadable
            loaded = torch.jit.load(out)
            dummy = torch.randn(1, 3, 32, 32)
            result = loaded(dummy)
            assert result.shape == (1, 4)

    def test_rfdetr_position_embedding_dim_buffer_not_checkpointed(self):
        from libreyolo.models.rfdetr.backbone import PositionEmbeddingSine

        module = PositionEmbeddingSine(num_pos_feats=8, normalize=True)

        assert "dim_t" not in module.state_dict()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_position_embedding_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.backbone import PositionEmbeddingSine

        module = PositionEmbeddingSine(num_pos_feats=8, normalize=True)
        module.export()
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        traced = torch.jit.trace(module, mask)
        path = tmp_path / "position_embedding.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(mask.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_proposal_grid_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.transformer import gen_encoder_output_proposals

        class ProposalModule(nn.Module):
            def forward(self, memory):
                _, proposals = gen_encoder_output_proposals(
                    memory,
                    spatial_shapes=[(2, 2)],
                    unsigmoid=False,
                )
                return proposals

        module = ProposalModule()
        memory = torch.zeros(1, 4, 8)
        traced = torch.jit.trace(module, memory)
        path = tmp_path / "proposal_grid.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(memory.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_sine_embedding_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.transformer import gen_sineembed_for_position

        class SineModule(nn.Module):
            def forward(self, pos):
                return gen_sineembed_for_position(pos, 128.0)

        module = SineModule()
        pos = torch.rand(2, 3, 4)
        traced = torch.jit.trace(module, pos)
        path = tmp_path / "sine_embedding.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(pos.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_seg_depthwise_block_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.segmentation import DepthwiseConvBlock

        module = DepthwiseConvBlock(4)
        module.export()
        x = torch.randn(1, 4, 8, 8)
        traced = torch.jit.trace(module, x)
        path = tmp_path / "seg_depthwise_block.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(x.to("cuda"))

        assert out.device.type == "cuda"


class TestModelStateRestored:
    def test_model_stays_on_original_device(self):
        wrapper = _make_wrapper()
        original_device = next(wrapper.model.parameters()).device

        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(output_path=str(Path(tmpdir) / "test.torchscript"))

        current_device = next(wrapper.model.parameters()).device
        assert current_device == original_device

    def test_half_restored_to_float32(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(
                output_path=str(Path(tmpdir) / "test.torchscript"),
                half=True,
            )

        param = next(wrapper.model.parameters())
        assert param.dtype == torch.float32


# ---------------------------------------------------------------------------
# TensorRT Export Tests
# ---------------------------------------------------------------------------


class TestTensorRTFormat:
    """Test TensorRT format registration and validation."""

    def test_tensorrt_format_registered(self):
        """Verify TensorRT is in registry."""
        assert "tensorrt" in BaseExporter._registry

    def test_tensorrt_format_config(self):
        """Verify TensorRT format configuration."""
        assert TensorRTExporter.suffix == ".engine"
        assert TensorRTExporter.requires_onnx is True


class TestTensorRTValidation:
    """Test TensorRT export parameter validation."""

    def test_int8_without_data_requires_calibration(self):
        """TensorRT INT8 export requires explicit calibration data."""
        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)

        with pytest.raises(ValueError, match="requires calibration data"):
            exporter(int8=True)

    def test_int8_with_data_no_immediate_error(self, monkeypatch):
        """INT8 with data parameter should not raise validation error.

        Note: Will fail later due to missing TensorRT (or ONNX), but validation should pass.
        """
        try:
            import tensorrt  # noqa: F401

            pytest.skip("TensorRT is installed, skipping missing TensorRT test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)
        monkeypatch.setattr(
            exporter,
            "_load_calibration",
            lambda *args, **kwargs: pytest.fail("calibration should not load"),
        )

        # Should fail with ImportError (missing onnx or tensorrt), not ValueError
        with pytest.raises(ImportError):
            exporter(int8=True, data="unused-local-calibration.yaml")

    def test_int8_with_data_missing_onnx_does_not_load_calibration(self, monkeypatch):
        """Missing ONNX should fail before calibration data is resolved."""
        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)
        original_find_spec = exporter_module.importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "onnx":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(exporter_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(
            exporter,
            "_load_calibration",
            lambda *args, **kwargs: pytest.fail("calibration should not load"),
        )

        with pytest.raises(ImportError, match="ONNX export requires"):
            exporter(int8=True, data="unused-local-calibration.yaml")


class TestTensorRTImportCheck:
    """Test TensorRT availability checking."""

    def test_check_tensorrt_raises_helpful_error(self):
        """Verify helpful error message when TensorRT not installed."""
        # Skip if TensorRT is actually installed
        try:
            import tensorrt  # noqa: F401

            pytest.skip("TensorRT is installed, skipping missing TensorRT test")
        except ImportError:
            pass

        from libreyolo.export.tensorrt import check_tensorrt_available

        with pytest.raises(ImportError) as exc_info:
            check_tensorrt_available()

        error_msg = str(exc_info.value)
        assert "tensorrt" in error_msg.lower()
        assert "pip install" in error_msg


class TestCalibrationDataLoader:
    """Test calibration data loader for INT8 quantization."""

    def test_calibration_loader_import(self):
        """Verify calibration module can be imported."""
        from libreyolo.export.calibration import (
            CalibrationDataLoader,
            get_calibration_dataloader,
        )

        assert CalibrationDataLoader is not None
        assert get_calibration_dataloader is not None

    def test_calibration_loader_properties(self):
        """Test calibration loader with mock data would have correct properties."""
        from libreyolo.export.calibration import CalibrationDataLoader

        # Check that dtype and shape properties are defined
        assert hasattr(CalibrationDataLoader, "shape")
        assert hasattr(CalibrationDataLoader, "dtype")

    def test_calibration_loader_shape_accepts_tuple_imgsz(self):
        """Rectangular INT8 calibration batches must match export input H/W."""
        from libreyolo.export.calibration import CalibrationDataLoader

        loader = CalibrationDataLoader.__new__(CalibrationDataLoader)
        loader.batch = 4
        loader.imgsz = (16, 32)

        assert loader.shape == (4, 3, 16, 32)


# ---------------------------------------------------------------------------
# OpenVINO Export Tests
# ---------------------------------------------------------------------------


class TestOpenVINOFormat:
    """Test OpenVINO format registration and validation."""

    def test_openvino_format_registered(self):
        """Verify OpenVINO is in registry."""
        assert "openvino" in BaseExporter._registry

    def test_openvino_format_config(self):
        """Verify OpenVINO format configuration."""
        assert OpenVINOExporter.suffix == "_openvino"
        assert OpenVINOExporter.requires_onnx is True


class TestOpenVINOValidation:
    """Test OpenVINO export parameter validation."""

    def test_int8_without_data_requires_calibration(self):
        """OpenVINO INT8 export requires explicit calibration data."""
        wrapper = _make_wrapper()
        exporter = OpenVINOExporter(wrapper)

        with pytest.raises(ValueError, match="requires calibration data"):
            exporter(int8=True)

    def test_int8_with_data_no_immediate_error(self, monkeypatch):
        """INT8 with data parameter should not raise validation error.

        Note: Will fail later due to missing OpenVINO (or ONNX), but validation should pass.
        """
        try:
            import openvino  # noqa: F401

            pytest.skip("OpenVINO is installed, skipping missing OpenVINO test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = OpenVINOExporter(wrapper)
        monkeypatch.setattr(
            exporter,
            "_load_calibration",
            lambda *args, **kwargs: pytest.fail("calibration should not load"),
        )

        # Should fail with ImportError (missing onnx or openvino), not ValueError
        with pytest.raises(ImportError):
            exporter(int8=True, data="unused-local-calibration.yaml")

    def test_int8_with_data_missing_onnx_does_not_load_calibration(self, monkeypatch):
        """Missing ONNX should fail before calibration data is resolved."""
        wrapper = _make_wrapper()
        exporter = OpenVINOExporter(wrapper)
        original_find_spec = exporter_module.importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "onnx":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(exporter_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(
            exporter,
            "_load_calibration",
            lambda *args, **kwargs: pytest.fail("calibration should not load"),
        )

        with pytest.raises(ImportError, match="ONNX export requires"):
            exporter(int8=True, data="unused-local-calibration.yaml")


class TestOpenVINOImportCheck:
    """Test OpenVINO availability checking."""

    def test_check_openvino_raises_helpful_error(self):
        """Verify helpful error message when OpenVINO not installed."""
        try:
            import openvino  # noqa: F401

            pytest.skip("OpenVINO is installed, skipping missing OpenVINO test")
        except ImportError:
            pass

        from libreyolo.export.openvino import check_openvino_available

        with pytest.raises(ImportError) as exc_info:
            check_openvino_available()

        error_msg = str(exc_info.value)
        assert "openvino" in error_msg.lower()
        assert "pip install" in error_msg


class TestExportPrecisionSuffix:
    """Test output filename generation with precision suffixes."""

    def test_fp16_suffix_in_auto_path(self):
        """FP16 export should include _fp16 in auto-generated filename."""
        wrapper = _make_wrapper(model_name="TESTYOLO", size="s")
        exporter = TorchScriptExporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter(half=True)
                assert "_fp16" in path, f"Expected _fp16 in path, got: {path}"
                assert path == str(Path("weights") / "testyolo_s_fp16.torchscript")
            finally:
                os.chdir(orig)

    def test_half_and_int8_uses_int8(self):
        """When both half and int8 are True, int8 takes precedence."""
        import warnings

        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)

        # Should warn about using INT8 when both specified
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                exporter(half=True, int8=True, data="unused-local-calibration.yaml")
            except ImportError:
                # Expected - TensorRT not installed
                pass
            except Exception:
                # May fail for other reasons if TensorRT is installed but
                # calibration data can't be loaded, etc. That's OK for this test.
                pass

            # Check that a warning was issued about INT8 precedence
            warning_msgs = [str(warning.message) for warning in w]
            assert any("INT8" in msg for msg in warning_msgs)


# ---------------------------------------------------------------------------
# TensorRT Export Config Tests
# ---------------------------------------------------------------------------


class TestTensorRTExportConfig:
    """Test TensorRT export configuration system."""

    def test_default_config(self):
        """Test default configuration values."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig()
        assert config.precision == "fp16"
        assert config.workspace == 4.0
        assert config.verbose is False
        assert config.hardware_compatibility == "none"
        assert config.device == 0
        assert config.dynamic.enabled is False
        assert config.int8_calibration.fraction == 0.1

    def test_config_half_property(self):
        """Test half property for different precisions."""
        from libreyolo.export.config import TensorRTExportConfig

        fp32_config = TensorRTExportConfig(precision="fp32")
        fp16_config = TensorRTExportConfig(precision="fp16")
        int8_config = TensorRTExportConfig(precision="int8")

        assert fp32_config.half is False
        assert fp16_config.half is True
        assert int8_config.half is True  # INT8 includes FP16 fallback

    def test_config_int8_property(self):
        """Test int8 property for different precisions."""
        from libreyolo.export.config import TensorRTExportConfig

        fp32_config = TensorRTExportConfig(precision="fp32")
        fp16_config = TensorRTExportConfig(precision="fp16")
        int8_config = TensorRTExportConfig(precision="int8")

        assert fp32_config.int8 is False
        assert fp16_config.int8 is False
        assert int8_config.int8 is True

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig.from_dict(
            {
                "precision": "int8",
                "workspace": 8.0,
                "hardware_compatibility": "ampere_plus",
                "dynamic": {"enabled": True, "max_batch": 16},
            }
        )

        assert config.precision == "int8"
        assert config.workspace == 8.0
        assert config.hardware_compatibility == "ampere_plus"
        assert config.dynamic.enabled is True
        assert config.dynamic.max_batch == 16

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig(precision="fp32", workspace=2.0)
        data = config.to_dict()

        assert data["precision"] == "fp32"
        assert data["workspace"] == 2.0
        assert "dynamic" in data
        assert "int8_calibration" in data

    def test_config_validation_invalid_precision(self):
        """Test validation rejects invalid precision."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="Invalid precision"):
            TensorRTExportConfig(precision="fp8")

    def test_config_validation_invalid_workspace(self):
        """Test validation rejects invalid workspace."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="workspace must be positive"):
            TensorRTExportConfig(workspace=-1.0)

    def test_config_validation_invalid_hardware_compat(self):
        """Test validation rejects invalid hardware compatibility."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="Invalid hardware_compatibility"):
            TensorRTExportConfig(hardware_compatibility="invalid")

    def test_load_export_config_none(self):
        """Test load_export_config with None returns default."""
        from libreyolo.export.config import load_export_config, TensorRTExportConfig

        config = load_export_config(None)
        assert isinstance(config, TensorRTExportConfig)
        assert config.precision == "fp16"

    def test_load_export_config_dict(self):
        """Test load_export_config with dict."""
        from libreyolo.export.config import load_export_config

        config = load_export_config({"precision": "fp32"})
        assert config.precision == "fp32"

    def test_load_export_config_passthrough(self):
        """Test load_export_config passes through existing config."""
        from libreyolo.export.config import load_export_config, TensorRTExportConfig

        original = TensorRTExportConfig(precision="int8")
        config = load_export_config(original)
        assert config is original

    def test_load_export_config_yaml(self):
        """Test load_export_config from YAML file."""
        from libreyolo.export.config import load_export_config

        config = load_export_config("tensorrt_default.yaml")
        assert config.precision == "fp16"
        assert config.workspace == 4.0
