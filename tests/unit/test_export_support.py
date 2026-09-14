"""Regression tests for the centralized export support matrix."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from libreyolo.export.exporter import ExecuTorchExporter, NcnnExporter, OnnxExporter
from libreyolo.export.support import EXPORT_FORMATS, SUPPORT, get_support
from libreyolo.models.inventory import collect_model_inventory
from libreyolo.tasks import TASKS, task_to_suffix

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_SNAPSHOT = REPO_ROOT / "reports" / "export_inventory.json"


def _wrapper(family: str, task: str = "detect") -> MagicMock:
    model = MagicMock()
    model._get_model_name.return_value = family
    model.task = task
    return model


def test_matrix_keys_use_canonical_registry_values():
    inventory = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    families = set(inventory)
    for family, task, fmt in SUPPORT:
        assert family in families
        assert task in TASKS
        assert fmt in EXPORT_FORMATS


def test_matrix_uses_capability_statuses_only():
    assert {entry.tier for entry in SUPPORT.values()} == {
        "validated",
        "available",
        "blocked",
    }


def test_matrix_rejects_duplicate_explicit_keys():
    from libreyolo.export import support

    key = ("yolo9", "detect", "onnx")
    original = support.SUPPORT[key]
    with pytest.raises(ValueError, match="Duplicate export support entries"):
        support._add("validated", (key[0],), (key[1],), (key[2],))
    assert support.SUPPORT[key] is original


@pytest.mark.parametrize(
    "family",
    ["dfine", "deim", "deimv2", "rtdetr", "rtdetrv2", "rtdetrv4", "rfdetr", "ec"],
)
def test_ncnn_detr_families_fail_in_preflight(family):
    exporter = NcnnExporter(_wrapper(family))
    with pytest.raises(NotImplementedError, match="NCNN"):
        exporter._preflight(half=False, int8=False, data=None)


def test_available_export_proceeds_without_blanket_warning():
    exporter = OnnxExporter(_wrapper("deimv2"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exporter._preflight(half=False, int8=False, data=None)
    assert not caught


@pytest.mark.parametrize(
    "exporter_cls,family,task,reason",
    [
        (
            ExecuTorchExporter,
            "feynobg",
            "matte",
            r"no \.pte artifact was produced",
        ),
        (NcnnExporter, "yolo2", "detect", "integer divide-by-zero"),
    ],
)
def test_non_runnable_exports_are_blocked_in_preflight(
    exporter_cls, family, task, reason
):
    exporter = exporter_cls(_wrapper(family, task))
    with pytest.raises(NotImplementedError, match=reason):
        exporter._preflight(half=False, int8=False, data=None)


@pytest.mark.parametrize(
    ("family", "task"),
    [
        ("ddcolor", "restore"),
        ("hvi_cidnet", "restore"),
        ("lama", "restore"),
        ("vitmatte", "matte"),
    ],
)
def test_guided_restoration_specialists_block_unvalidated_exports(family, task):
    for format_name in EXPORT_FORMATS:
        assert get_support(family, task, format_name).tier == "blocked"


@pytest.mark.parametrize("family", ["yolo9", "yolo9_e2e", "yolonas", "picodet"])
def test_rknn_retained_families_are_available(family):
    entry = get_support(family, "detect", "rknn")
    assert entry.tier == "available"
    assert "RK3588 PC-simulator" in entry.reason


@pytest.mark.parametrize("family", ["rfdetr", "yolo9_p2", "ec", "rtdetr", "deim"])
def test_rknn_failed_candidates_are_blocked(family):
    entry = get_support(family, "detect", "rknn")
    assert entry.tier == "blocked"


def test_vit_onnx_is_parity_validated():
    entry = get_support("vit", "classify", "onnx")
    assert entry.tier == "validated"
    assert entry.constraint == "FP32, fixed 224x224 input"
    assert "test_vit_export.py" in entry.reason


def test_ssd_exports_only_through_its_validated_onnx_contract():
    assert get_support("ssd", "detect", "onnx").tier == "validated"
    for fmt in EXPORT_FORMATS:
        if fmt != "onnx":
            assert get_support("ssd", "detect", fmt).tier == "blocked"


def test_executorch_realtime_support_is_evidence_backed():
    validated = {
        ("clip", "embed"),
        ("convnext", "classify"),
        ("depth_anything", "depth"),
        ("depth_anything3", "depth"),
        ("dexined", "edge"),
        ("dinov2", "classify"),
        ("dinov2", "semantic"),
        ("ec", "detect"),
        ("ec", "pose"),
        ("ec", "segment"),
        ("efficientnetv2", "classify"),
        ("fomo", "point"),
        ("l2cs", "gaze"),
        ("lingbotvision", "semantic"),
        ("moge2", "normal"),
        ("mobilenetv4", "classify"),
        ("nafnet", "restore"),
        ("picodet", "detect"),
        ("pidnet", "semantic"),
        ("realesrgan", "restore"),
        ("resnet", "classify"),
        ("rtdetr", "detect"),
        ("rtdetrv2", "detect"),
        ("rtdetrv4", "detect"),
        ("segformer", "semantic"),
        ("siglip2", "embed"),
        ("rfdetr", "detect"),
        ("rfdetr", "obb"),
        ("rfdetr", "pose"),
        ("rfdetr", "segment"),
        ("teed", "edge"),
        ("yolo1", "detect"),
        ("yolo2", "detect"),
        ("yolo3", "detect"),
        ("yolo4", "detect"),
        ("yolo7", "detect"),
        ("yolo9", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolo9_p2", "detect"),
        ("yolonas", "detect"),
        ("yolonas", "pose"),
        ("yolox", "detect"),
        ("zipdepth", "depth"),
    }
    for family, task in validated:
        assert get_support(family, task, "executorch").tier == "validated"

    assert get_support("rtmdet", "detect", "executorch").tier == "available"
    assert get_support("dfine", "detect", "executorch").tier == "blocked"
    assert get_support("deim", "detect", "executorch").tier == "blocked"
    assert get_support("deimv2", "detect", "executorch").tier == "blocked"
    assert get_support("swinir", "restore", "executorch").tier == "blocked"
    assert get_support("dinov2", "embed", "executorch").tier == "available"
    assert get_support("eomt", "semantic", "executorch").tier == "blocked"
    assert get_support("birefnet", "matte", "executorch").tier == "blocked"
    assert get_support("feynobg", "matte", "executorch").tier == "blocked"


def test_tflite_support_keys_use_canonical_tasks():
    from libreyolo.export.tflite import supported_tflite_exports

    assert all(task in TASKS for _, task in supported_tflite_exports())
    assert get_support("yolo3", "detect", "tflite").tier == "blocked"
    assert get_support("rfdetr", "detect", "tflite").tier == "blocked"
    assert get_support("rfdetr", "segment", "tflite").tier == "blocked"
    assert get_support("dinov2", "embed", "tflite").tier == "validated"
    assert get_support("siglip2", "embed", "tflite").tier == "validated"
    assert get_support("clip", "embed", "tflite").tier == "blocked"
    assert get_support("siglip2", "classify", "tflite").tier == "validated"
    assert get_support("clip", "classify", "tflite").tier == "blocked"


@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_dinov2_classify_routes_to_base_export(monkeypatch, format):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = object.__new__(LibreDINOv2)
    model.task = "classify"
    captured = {}

    def fake_export(self, format="onnx", **kwargs):
        captured.update(format=format, **kwargs)
        return f"dinov2.{format}"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    assert model.export(format, dynamic=False) == f"dinov2.{format}"
    assert captured == {"format": format, "opset": 17, "dynamic": False}


def test_dinov2_semantic_routes_to_shared_export(monkeypatch):
    from libreyolo.models.base.model import BaseModel
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = object.__new__(LibreDINOv2)
    model.task = "semantic"
    captured = {}

    def fake_export(self, format="onnx", **kwargs):
        captured.update(format=format, **kwargs)
        return "dinov2-semantic.onnx"

    monkeypatch.setattr(BaseModel, "export", fake_export)
    assert model.export("onnx", dynamic=False) == "dinov2-semantic.onnx"
    assert captured == {"format": "onnx", "opset": 17, "dynamic": False}


def test_observed_cpu_toolchain_blocks_are_explicit():
    depth_ncnn = get_support("depth_anything", "depth", "ncnn")
    fomo_tflite = get_support("fomo", "point", "tflite")
    assert depth_ncnn.tier == "blocked" and "reshape" in depth_ncnn.reason
    assert fomo_tflite.tier == "blocked" and "DEPTHWISE" in fomo_tflite.reason


@pytest.mark.parametrize(
    ("family", "task", "reason_fragment"),
    [
        ("yolo1", "detect", "ONNX_EINSUM"),
        ("yolo9_e2e", "detect", "top-k"),
        ("yolo9_p2", "detect", "top-k"),
        ("yolonas", "pose", "CONCATENATION"),
        ("rtmdet", "detect", "0.911 IoU"),
        ("picodet", "detect", "19,200"),
        ("dfine", "detect", "GatherElements"),
        ("ec", "detect", "ONNX_LAYERNORMALIZATION"),
        ("rtdetr", "detect", "CONCATENATION"),
    ],
)
def test_round6_tflite_blocks_are_measured(family, task, reason_fragment):
    entry = get_support(family, task, "tflite")
    assert entry.tier == "blocked"
    assert reason_fragment in entry.reason


def test_yolonas_detect_tflite_is_validated():
    assert get_support("yolonas", "detect", "tflite").tier == "validated"


def test_paddle_support_matches_measured_family_task_matrix():
    validated = {
        ("yolo9", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolo9_p2", "detect"),
        ("ec", "detect"),
        ("ec", "pose"),
        ("ec", "segment"),
        ("rtdetrv4", "detect"),
        ("dfine", "detect"),
        ("deim", "detect"),
        ("deimv2", "detect"),
        ("yolonas", "detect"),
        ("yolonas", "pose"),
    }
    for family, task in validated:
        entry = get_support(family, task, "paddle")
        assert entry.tier == "validated"
        assert "PaddlePaddle 2.6.2" in entry.constraint

    for task in ("detect", "segment", "pose", "obb"):
        rfdetr = get_support("rfdetr", task, "paddle")
        assert rfdetr.tier == "blocked"
        assert "GridSample" in rfdetr.reason
    for family in ("rtdetr", "rtdetrv2"):
        assert get_support(family, "detect", "paddle").tier == "blocked"
    dfine_segment = get_support("dfine", "segment", "paddle")
    assert dfine_segment.tier == "blocked"
    assert "0.582" in dfine_segment.reason
    assert get_support("yolox", "detect", "paddle").tier == "blocked"


def test_fcn_semantic_export_support_matches_trained_checkpoint_evidence():
    for format in ("onnx", "torchscript", "openvino", "tensorrt"):
        entry = get_support("fcn", "semantic", format)
        assert entry.tier == "validated"
        assert "parity" in entry.reason
        assert "divisible by 8" in entry.constraint
    for format in ("executorch", "ncnn", "tflite", "coreml", "coreai"):
        assert get_support("fcn", "semantic", format).tier == "blocked"


def test_round7_swinir_fixed_canvas_exports_are_validated():
    for format in ("onnx", "torchscript", "openvino", "tflite"):
        entry = get_support("swinir", "restore", format)
        assert entry.tier == "validated"
        assert "exactly match" in entry.constraint


def test_round8_tensorrt_fp32_parity_promotes_nine_cells():
    validated = {
        ("mobilenetv4", "classify"),
        ("convnext", "classify"),
        ("efficientnetv2", "classify"),
        ("resnet", "classify"),
        ("fomo", "point"),
        ("realesrgan", "restore"),
        ("nafnet", "restore"),
        ("swinir", "restore"),
        ("depth_anything", "depth"),
    }
    for family, task in validated:
        entry = get_support(family, task, "tensorrt")
        assert entry.tier == "validated"
        assert "FP32" in entry.constraint

    pidnet = get_support("pidnet", "semantic", "tensorrt")
    assert pidnet.tier == "available"
    assert "0.9970" in pidnet.reason


def test_swin_classifier_exports_are_backed_by_trained_runtime_parity():
    for format in ("onnx", "torchscript", "openvino", "tensorrt"):
        entry = get_support("swin", "classify", format)
        assert entry.tier == "validated"
        assert "test_swin_export.py" in entry.reason
        assert "224x224" in entry.constraint


def test_round9_promotes_three_parity_cells_and_records_seven_gaps():
    deim = get_support("deim", "detect", "onnx")
    assert deim.tier == "validated"
    assert "unordered set" in deim.constraint

    for family, task in {
        ("dinov2", "semantic"),
        ("eomt", "semantic"),
    }:
        entry = get_support(family, task, "tensorrt")
        assert entry.tier == "validated"
        assert "FP32" in entry.constraint

    lingbot = get_support("lingbotvision", "semantic", "tensorrt")
    assert lingbot.tier == "available"
    assert "0.9842" in lingbot.reason

    zipdepth = get_support("zipdepth", "depth", "tensorrt")
    assert zipdepth.tier == "available"
    assert "30.27 dB" in zipdepth.reason

    expected_gaps = {
        "deimv2": "43.7%",
    }
    for family, measured in expected_gaps.items():
        entry = get_support(family, "detect", "onnx")
        assert entry.tier == "available"
        assert measured in entry.reason

    for family in ("birefnet", "feynobg"):
        entry = get_support(family, "matte", "tensorrt")
        assert entry.tier == "blocked"
        assert "ModulatedDeformConv2d" in entry.reason


def test_round16_promotes_rtdetrv2_and_rtdetrv4_onnx():
    for family in ("rtdetrv2", "rtdetrv4"):
        entry = get_support(family, "detect", "onnx")
        assert entry.tier == "validated"
        assert "published Apache-2.0 trained checkpoint" in entry.constraint
        assert "non-square public predict parity" in entry.constraint


def test_round17_records_ten_ncnn_cpu_fp32_parity_cells():
    cases = (
        ("mobilenetv4", "classify"),
        ("convnext", "classify"),
        ("efficientnetv2", "classify"),
        ("resnet", "classify"),
        ("fomo", "point"),
        ("pidnet", "semantic"),
        ("realesrgan", "restore"),
        ("nafnet", "restore"),
        ("zipdepth", "depth"),
        ("yolo9", "detect"),
    )
    for family, task in cases:
        entry = get_support(family, task, "ncnn")
        assert entry.tier == "validated"
        assert "PNNX/NCNN 20260526 CPU FP32" in entry.constraint
        assert "public predict parity" in entry.constraint


def test_round18_records_nine_ncnn_parity_cells_and_two_holds():
    validated = (
        ("picodet", "detect"),
        ("yolo1", "detect"),
        ("yolo3", "detect"),
        ("yolo4", "detect"),
        ("yolo7", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolox", "detect"),
        ("yolonas", "detect"),
        ("yolonas", "pose"),
    )
    for family, task in validated:
        entry = get_support(family, task, "ncnn")
        assert entry.tier == "validated"
        assert "PNNX/NCNN 20260526 CPU FP32" in entry.constraint
        assert "public predict parity" in entry.constraint

    yolo2 = get_support("yolo2", "detect", "ncnn")
    assert yolo2.tier == "blocked"
    assert "integer divide-by-zero" in yolo2.reason

    yolo9_p2 = get_support("yolo9_p2", "detect", "ncnn")
    assert yolo9_p2.tier == "available"
    assert "no detections above 0.05" in yolo9_p2.reason


def test_round10_promotes_three_tensorrt_detectors():
    for family in ("yolo2", "yolo3", "yolo4"):
        entry = get_support(family, "detect", "tensorrt")
        assert entry.tier == "validated"
        assert "FP32" in entry.constraint


def test_round11_promotes_three_trained_tensorrt_detectors_and_records_holds():
    for family in ("yolo1", "picodet", "rtmdet"):
        entry = get_support(family, "detect", "tensorrt")
        assert entry.tier == "validated"
        assert "TensorRT 10.16 FP32" in entry.constraint

    measured_holds = {
        ("yolo7", "detect"): "trained checkpoint",
        ("yolo9_e2e", "detect"): "trained checkpoint",
        ("yolo9_p2", "detect"): "transfer fixture",
        ("yolox", "detect"): "1.6%",
        ("rtdetr", "detect"): "17% to 38%",
        ("yolonas", "detect"): "4 to 5 times",
        ("yolonas", "pose"): "2 to 6 times",
    }
    for (family, task), reason_fragment in measured_holds.items():
        entry = get_support(family, task, "tensorrt")
        assert entry.tier == "available"
        assert reason_fragment in entry.reason


def test_round12_records_ten_measured_tensorrt_holds():
    measured_holds = {
        ("dfine", "detect"): "top-k class membership",
        ("dfine", "segment"): "top-k class membership",
        ("deim", "detect"): "0.41%",
        ("rtdetrv2", "detect"): "0.231 IoU",
        ("ec", "detect"): "1.2%",
        ("ec", "pose"): "0.920 IoU",
        ("ec", "segment"): "top-k class membership",
        ("rfdetr", "segment"): "top-k class membership",
        ("rfdetr", "pose"): "0.704 IoU",
        ("rfdetr", "obb"): "top-k class membership",
    }
    for (family, task), reason_fragment in measured_holds.items():
        entry = get_support(family, task, "tensorrt")
        assert entry.tier == "available"
        assert reason_fragment in entry.reason


def test_round15_records_rtdetrv4_tensorrt_hold():
    entry = get_support("rtdetrv4", "detect", "tensorrt")
    assert entry.tier == "available"
    assert "50.4-pixel" in entry.reason


def test_round21_records_six_trained_openvino_holds():
    measured_holds = {
        ("deim", "detect"): "17.9x",
        ("rtdetrv2", "detect"): "93.94%",
        ("ec", "pose"): "0.916",
        ("rfdetr", "segment"): "69.0%",
        ("rfdetr", "pose"): "72.75%",
        ("rfdetr", "obb"): "91.25%",
    }
    for (family, task), reason_fragment in measured_holds.items():
        entry = get_support(family, task, "openvino")
        assert entry.tier == "available"
        assert reason_fragment in entry.reason


def test_round22_promotes_nine_edge_gaze_and_classify_cells():
    validated = {
        ("teed", "edge", "torchscript"),
        ("teed", "edge", "openvino"),
        ("teed", "edge", "tensorrt"),
        ("dexined", "edge", "torchscript"),
        ("dexined", "edge", "openvino"),
        ("dexined", "edge", "tensorrt"),
        ("l2cs", "gaze", "openvino"),
        ("l2cs", "gaze", "tensorrt"),
        ("dinov2", "classify", "openvino"),
    }
    for family, task, format in validated:
        entry = get_support(family, task, format)
        assert entry.tier == "validated"
        assert "parity" in entry.reason

    hold = get_support("dinov2", "classify", "tensorrt")
    assert hold.tier == "available"
    assert "2.2x" in hold.reason


def test_round23_promotes_nine_normal_semantic_depth_and_edge_cells():
    validated = {
        ("moge2", "normal", "torchscript"),
        ("moge2", "normal", "openvino"),
        ("moge2", "normal", "tensorrt"),
        ("segformer", "semantic", "executorch"),
        ("segformer", "semantic", "tensorrt"),
        ("dinov2", "semantic", "executorch"),
        ("depth_anything3", "depth", "executorch"),
        ("teed", "edge", "tflite"),
        ("dexined", "edge", "tflite"),
    }
    for family, task, format in validated:
        entry = get_support(family, task, format)
        assert entry.tier == "validated"
        assert "parity" in entry.reason

    hold = get_support("rtmdet", "detect", "executorch")
    assert hold.tier == "available"
    assert "detection parsing" in hold.reason


def test_round24_promotes_six_embedding_and_depth_cells():
    validated = {
        ("dinov2", "embed", "onnx"),
        ("dinov2", "embed", "torchscript"),
        ("depth_anything3", "depth", "onnx"),
        ("depth_anything3", "depth", "torchscript"),
        ("depth_anything3", "depth", "openvino"),
        ("depth_anything3", "depth", "tensorrt"),
    }
    for family, task, format in validated:
        assert get_support(family, task, format).tier == "validated"

    assert get_support("dinov2", "embed", "openvino").tier == "available"
    assert get_support("dinov2", "embed", "tensorrt").tier == "available"
    assert get_support("moge2", "normal", "ncnn").tier == "available"
    assert get_support("moge2", "normal", "tflite").tier == "blocked"


def test_midas_trained_export_cells_are_validated():
    for format in ("onnx", "torchscript", "tensorrt", "openvino"):
        entry = get_support("midas", "depth", format)
        assert entry.tier == "validated"
        assert "parity" in entry.reason
        assert "fixed square" in entry.constraint


def test_round13_records_ten_measured_tflite_holds():
    measured_holds = {
        ("yolo1", "detect"): "ONNX_EINSUM",
        ("yolo9_e2e", "detect"): "top-k class membership",
        ("yolo9_p2", "detect"): "top-k class membership",
        ("rtmdet", "detect"): "0.911 IoU",
        ("picodet", "detect"): "19,200",
        ("dfine", "detect"): "GatherElements",
        ("ec", "detect"): "ONNX_LAYERNORMALIZATION",
        ("rtdetr", "detect"): "CONCATENATION",
        ("yolonas", "pose"): "CONCATENATION",
        ("dfine", "segment"): "GatherElements",
    }
    for (family, task), reason_fragment in measured_holds.items():
        entry = get_support(family, task, "tflite")
        assert entry.tier == "blocked"
        assert reason_fragment in entry.reason


def test_round14_records_ten_measured_tflite_holds():
    measured_holds = {
        ("yolo2", "detect"): "4,225",
        ("yolo3", "detect"): "public-domain trained checkpoint",
        ("yolo4", "detect"): "0 IoU",
        ("fomo", "point"): "16 filter channels",
        ("nafnet", "restore"): "4539",
        ("depth_anything", "depth"): "[1,3,3,32]",
        ("segformer", "semantic"): "1024 input elements",
        ("zipdepth", "depth"): "edge-mode Pad",
        ("rfdetr", "detect"): "STRIDED_SLICE",
        ("rtdetrv4", "detect"): "640x640",
    }
    for (family, task), reason_fragment in measured_holds.items():
        entry = get_support(family, task, "tflite")
        assert entry.tier == "blocked"
        assert reason_fragment in entry.reason


@pytest.mark.parametrize(
    ("family", "task", "format", "reason_fragment"),
    [
        ("swinir", "restore", "ncnn", "5-rank"),
        ("birefnet", "matte", "openvino", "DeformConv-19"),
        ("feynobg", "matte", "openvino", "DeformConv-19"),
        ("dfine", "segment", "tflite", "GatherElements"),
        ("rfdetr", "detect", "tflite", "STRIDED_SLICE"),
        ("rtdetrv4", "detect", "tflite", "640x640"),
    ],
)
def test_round7_measured_blocks_are_explicit(family, task, format, reason_fragment):
    entry = get_support(family, task, format)
    assert entry.tier == "blocked"
    assert reason_fragment in entry.reason


def test_coreai_validated_tier_has_hardware_parity_coverage():
    validated = {
        (family, task)
        for (family, task, fmt), entry in SUPPORT.items()
        if fmt == "coreai" and entry.tier == "validated"
    }
    assert validated == {
        ("clip", "classify"),
        ("convnext", "classify"),
        ("deim", "detect"),
        ("deimv2", "detect"),
        ("depth_anything", "depth"),
        ("dfine", "detect"),
        ("ec", "detect"),
        ("efficientnetv2", "classify"),
        ("fomo", "point"),
        ("lingbotvision", "semantic"),
        ("mobilenetv4", "classify"),
        ("nafnet", "restore"),
        ("picodet", "detect"),
        ("pidnet", "semantic"),
        ("realesrgan", "restore"),
        ("resnet", "classify"),
        ("rfdetr", "detect"),
        ("rtdetr", "detect"),
        ("rtdetrv2", "detect"),
        ("rtdetrv4", "detect"),
        ("rtmdet", "detect"),
        ("siglip2", "classify"),
        ("yolo1", "detect"),
        ("yolo2", "detect"),
        ("yolo3", "detect"),
        ("yolo4", "detect"),
        ("yolo7", "detect"),
        ("yolo9", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolo9_p2", "detect"),
        ("yolonas", "detect"),
        ("yolox", "detect"),
        ("zipdepth", "depth"),
    }


def test_openvino_validated_tier_has_runtime_parity_coverage():
    validated = {
        (family, task)
        for (family, task, fmt), entry in SUPPORT.items()
        if fmt == "openvino" and entry.tier == "validated"
    }
    assert validated == {
        ("alexnet", "classify"),
        ("clip", "classify"),
        ("clip", "embed"),
        ("convnext", "classify"),
        ("deeplabv3", "semantic"),
        ("dekr", "pose"),
        ("deit", "classify"),
        ("depth_anything", "depth"),
        ("depth_anything3", "depth"),
        ("dexined", "edge"),
        ("dfine", "detect"),
        ("dfine", "segment"),
        ("dinov2", "classify"),
        ("dinov2", "semantic"),
        ("ec", "detect"),
        ("ec", "segment"),
        ("efficientdet", "detect"),
        ("efficientnetv2", "classify"),
        ("eomt", "semantic"),
        ("fcn", "semantic"),
        ("fomo", "point"),
        ("hrnet", "pose"),
        ("lingbotvision", "semantic"),
        ("l2cs", "gaze"),
        ("midas", "depth"),
        ("mobilenetv4", "classify"),
        ("moge2", "normal"),
        ("nafnet", "restore"),
        ("picodet", "detect"),
        ("pidnet", "semantic"),
        ("ppliteseg", "semantic"),
        ("ppyoloe", "detect"),
        ("realesrgan", "restore"),
        ("resnet", "classify"),
        ("rfdetr", "detect"),
        ("rtmdet", "detect"),
        ("rtdetr", "detect"),
        ("rtdetrv4", "detect"),
        ("segformer", "semantic"),
        ("siglip2", "classify"),
        ("siglip2", "embed"),
        ("swin", "classify"),
        ("swinir", "restore"),
        ("teed", "edge"),
        ("vgg", "classify"),
        ("yolo1", "detect"),
        ("yolo2", "detect"),
        ("yolo3", "detect"),
        ("yolo4", "detect"),
        ("yolo7", "detect"),
        ("yolo9", "detect"),
        ("yolo9_e2e", "detect"),
        ("yolo9_p2", "detect"),
        ("yolonas", "detect"),
        ("yolonas", "obb"),
        ("yolonas", "pose"),
        ("yolox", "detect"),
        ("zipdepth", "depth"),
    }


def test_fallback_reasons_describe_project_support_not_developer_environment():
    semantic = get_support("unwired_family", "semantic", "onnx")
    tensorrt = get_support("unwired_family", "detect", "tensorrt")
    eomt_segment = get_support("eomt", "segment", "onnx")

    assert "not wired" in semantic.reason
    assert "project has not yet recorded" in tensorrt.reason
    assert "this environment" not in tensorrt.reason
    assert "instance and panoptic" in eomt_segment.reason
    assert "semantic" not in eomt_segment.reason


def test_compat_table_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    from tools import gen_compat_table

    monkeypatch.chdir(tmp_path)
    assert gen_compat_table.INVENTORY_PATH.exists()
    rows, _, _, _ = gen_compat_table._rows()
    assert rows
    # The full matrix lives in docs/export_support.md; the README is curated.
    rendered = gen_compat_table.render_docs()
    assert rendered.startswith("# Export support")
    assert "## Available combinations" in rendered


def test_dump_inventory_refuses_partial_overwrite(tmp_path):
    from tools.dump_model_inventory import write_inventory

    output = tmp_path / "export_inventory.json"
    fake = {"zzz_fake_family": {"tasks": ["detect"]}}
    output.write_text(json.dumps(fake), encoding="utf-8")

    with pytest.raises(SystemExit, match="zzz_fake_family"):
        write_inventory(output)
    assert json.loads(output.read_text(encoding="utf-8")) == fake

    inventory = write_inventory(output, allow_family_removal=True)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert "zzz_fake_family" not in written
    assert written == inventory


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the canonical inventory snapshot includes transformer-backed families",
)
def test_committed_inventory_matches_runtime_inventory():
    committed = json.loads(INVENTORY_SNAPSHOT.read_text(encoding="utf-8"))
    assert committed == collect_model_inventory()


def test_partial_exporters_are_custom_not_blocked():
    """A family that exports some formats and raises for the rest is custom.

    PicoSAM3 ships a validated ONNX export and raises for every other format.
    Reporting it as ``blocked`` would tell inventory consumers to reject an
    export the support matrix marks validated.
    """
    inventory = collect_model_inventory()
    assert inventory["picosam3"]["export_override"] == "custom"
    assert get_support("picosam3", "segment", "onnx").tier == "validated"

    for family, metadata in inventory.items():
        if metadata["export_override"] != "blocked":
            continue
        for task in metadata["tasks"]:
            for format in EXPORT_FORMATS:
                assert get_support(family, task, format).tier != "validated", (
                    f"{family}/{task}/{format} is validated in the support "
                    "matrix but the inventory reports export as blocked"
                )


def test_default_download_urls_keep_task_repo_suffixes():
    from libreyolo.models.base.model import BaseModel

    for metadata in collect_model_inventory().values():
        module_name, class_name = metadata["class"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(cls, BaseModel) or "get_download_url" in cls.__dict__:
            continue
        # Runtime tasks can intentionally share an artifact. In that case the
        # family advertises only the distinct published suffixes through
        # WEIGHT_TASKS (for example classify weights reused by embed).
        weight_tasks = cls.WEIGHT_TASKS or metadata["tasks"]
        for task in weight_tasks:
            sizes = metadata["task_sizes"].get(task) or metadata["default_imgsz"]
            if not sizes or not cls.FILENAME_PREFIX:
                continue
            size = next(iter(sizes))
            suffix = task_to_suffix(task)
            filename = f"{cls.FILENAME_PREFIX}{size}"
            if suffix:
                filename += f"-{suffix}"
            filename += cls.WEIGHT_EXT
            url = BaseModel.get_download_url.__func__(cls, filename)
            assert url is not None
            expected_repo = f"/{cls.FILENAME_PREFIX}{size}"
            if suffix:
                expected_repo += f"-{suffix}"
            assert expected_repo + "/resolve/main/" in url


def test_generated_export_docs_are_current():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "gen_compat_table.py"), "--check"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_docs_expose_validated_constraints():
    from tools.gen_compat_table import render_docs

    docs = render_docs()
    assert "## Validated constraints" in docs
    assert "`yolonas` / `detect` / `coreai`" in docs
    assert "raw-image preprocessing" in docs


def test_shipping_rows_never_claim_a_future_since_version():
    """A validated/available row cannot postdate the library's own version.

    During 1.5.0 development 176 rows were stamped ``since="1.6"`` or
    ``since="1.7"`` for capabilities that ship in the 1.5 line. Nothing
    rendered ``since`` at the time, but the field becomes a lie the day the
    docs or CLI surface it. ``since`` names the major.minor release LINE a
    capability first ships in (see ``SupportEntry``), so rows added between
    the ``v1.5.0`` tag and the next point release legitimately carry "1.5".
    The floor is the source tree's declared major.minor from pyproject;
    installed dist metadata can lag a source checkout, so pyproject wins.
    """
    import re

    match = re.search(
        r'^\[project\][^\[]*?^version\s*=\s*"(\d+)\.(\d+)',
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        import libreyolo

        match = re.match(r"(\d+)\.(\d+)", libreyolo.__version__)
    assert match is not None
    floor = (int(match.group(1)), int(match.group(2)))

    offenders = {}
    for key, entry in SUPPORT.items():
        if entry.tier not in ("validated", "available") or entry.since is None:
            continue
        since = tuple(int(part) for part in entry.since.split("."))
        if since > floor:
            offenders[key] = entry.since
    assert not offenders, (
        f"Rows claim since > {'.'.join(map(str, floor))}: {sorted(offenders.items())}"
    )
