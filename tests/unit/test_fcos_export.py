"""FCOS single-tensor export and backend runtime contracts."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


class _AnchorStub(torch.nn.Module):
    def grid_anchors(self, images, features):
        del features
        anchors = images.new_tensor(
            [
                [0.0, 0.0, 8.0, 8.0],
                [8.0, 0.0, 16.0, 8.0],
                [0.0, 8.0, 8.0, 16.0],
                [8.0, 8.0, 16.0, 16.0],
                [16.0, 16.0, 24.0, 24.0],
            ]
        )
        return anchors.unsqueeze(0).expand(images.shape[0], -1, -1)


class _ModelStub(torch.nn.Module):
    num_classes = 91

    def __init__(self) -> None:
        super().__init__()
        self.anchor_generator = _AnchorStub()

    def forward_head(self, images):
        batch = images.shape[0]
        features = [images.new_zeros(batch, 1, 1, 1) for _ in range(5)]
        logits = images.new_full((batch, 5, 91), -10.0)
        logits[:, :, 1] = 10.0
        return (
            {
                "cls_logits": logits,
                "bbox_regression": images.new_full((batch, 5, 4), 0.5),
                "bbox_ctrness": images.new_full((batch, 5, 1), 10.0),
            },
            features,
        )


def test_export_wrapper_emits_one_mapped_tensor() -> None:
    from libreyolo.models.fcos.nn import FCOSExportWrapper

    output = FCOSExportWrapper(_ModelStub())(torch.zeros(1, 3, 24, 24))
    assert output.shape == (1, 5, 85)
    assert output[0, :, 4].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    torch.testing.assert_close(
        output[0, :, 5],
        torch.full((5,), torch.sigmoid(torch.tensor(10.0))),
    )
    assert bool((output[0, :, 6:] < 1e-2).all())


def test_backend_parser_is_multilabel_and_fcos_stays_nms_based() -> None:
    from libreyolo.backends.base import _is_nms_free_family
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend.model_family = "fcos"
    backend.task = "detect"
    backend.names = {0: "a", 1: "b"}
    output = np.zeros((1, 3, 7), dtype=np.float32)
    output[0, :, :4] = [0.0, 0.0, 8.0, 8.0]
    output[0, :, 4] = 0.0
    output[0, 0, 5] = 0.9
    output[0, 1, 5] = 0.8
    output[0, 2, 6] = 0.7

    boxes, scores, classes = backend._parse_fcos(
        [output], 8, orig_w=8, orig_h=8, conf=0.2
    )
    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(8, 8),
        image_path=None,
        iou=0.6,
        classes=None,
        max_det=300,
    )

    assert _is_nms_free_family("fcos") is False
    assert len(result) == 2
    assert result.boxes.cls.tolist() == [0.0, 1.0]
    torch.testing.assert_close(result.boxes.conf, torch.tensor([0.9, 0.7]))


def test_backend_preprocess_uses_aspect_resize_and_stride_padding() -> None:
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    image = Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8), "RGB")
    tensor, _, original_size, ratio = backend._preprocess_fcos(image, 32, "rgb")

    assert tensor.shape == (1, 3, 32, 64)
    assert original_size == (8, 6)
    assert ratio == 32 / 6


def test_onnx_and_openvino_force_dynamic_spatial_axes(monkeypatch) -> None:
    from libreyolo.models.base import BaseModel
    from libreyolo.models.fcos.model import LibreFCOS

    model = object.__new__(LibreFCOS)
    monkeypatch.setattr(
        BaseModel,
        "export",
        lambda self, format="onnx", **kwargs: (format, kwargs),
    )

    for export_format in ("onnx", "openvino"):
        with pytest.warns(RuntimeWarning, match="forcing dynamic=True"):
            selected, kwargs = model.export(export_format, dynamic=False)
        assert selected == export_format
        assert kwargs["dynamic"] is True
        assert kwargs["opset"] == 18


def test_export_support_is_explicit_for_every_format() -> None:
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    entries = {fmt: get_support("fcos", "detect", fmt) for fmt in EXPORT_FORMATS}
    assert entries["onnx"].tier == "validated"
    assert entries["torchscript"].tier == "validated"
    assert entries["openvino"].tier == "available"
    # Axelera: six compiled parts with the GroupNorm head on the host (docs/axelera.md).
    assert entries["axelera"].tier == "available"
    assert all(
        entry.tier == "blocked"
        for fmt, entry in entries.items()
        if fmt not in {"onnx", "torchscript", "openvino", "axelera"}
    )


@pytest.mark.external_data
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_FCOS_ACCEPTANCE") != "1",
    reason="set LIBREYOLO_FCOS_ACCEPTANCE=1 to run trained export parity",
)
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx and onnxruntime are required",
)
def test_trained_onnx_and_torchscript_runtime_parity(tmp_path) -> None:
    import onnx

    from libreyolo import LibreYOLO
    from libreyolo.models.fcos.nn import FCOSExportWrapper
    from libreyolo.models.fcos.utils import preprocess_image

    checkpoint = os.environ.get("LIBREYOLO_FCOS_ACCEPTANCE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("set LIBREYOLO_FCOS_ACCEPTANCE_CHECKPOINT to LibreFCOSr50.pt")

    model = LibreYOLO(checkpoint, device="cpu")
    image_path = Path(__file__).parents[1] / "fixtures" / "dog.jpg"
    tensor, _, _, _ = preprocess_image(image_path, input_size=128)
    with torch.inference_mode():
        expected_raw = FCOSExportWrapper(model.model).eval()(tensor).numpy()

    onnx_path = model.export(
        "onnx",
        imgsz=128,
        dynamic=True,
        simplify=False,
        device="cpu",
        output_path=str(tmp_path / "LibreFCOSr50.onnx"),
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    assert [value.name for value in graph.graph.output] == ["output"]
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["model_family"] == "fcos"
    assert metadata["model_size"] == "r50"

    onnx_backend = LibreYOLO(onnx_path, device="cpu")
    onnx_raw = onnx_backend._run_inference(tensor.numpy())[0]
    assert onnx_raw.shape[-1] == 85
    np.testing.assert_allclose(onnx_raw, expected_raw, rtol=2e-4, atol=2e-4)

    torchscript_path = model.export(
        "torchscript",
        imgsz=128,
        device="cpu",
        output_path=str(tmp_path / "LibreFCOSr50.torchscript"),
    )
    torchscript_backend = LibreYOLO(torchscript_path, device="cpu")
    torchscript_raw = torchscript_backend._run_inference(tensor.numpy())[0]
    np.testing.assert_allclose(torchscript_raw, expected_raw, rtol=0, atol=0)

    native_result = model.predict(
        image_path, imgsz=128, conf=0.05, iou=0.6, max_det=100
    )
    for backend in (onnx_backend, torchscript_backend):
        converted = backend.predict(
            image_path, imgsz=128, conf=0.05, iou=0.6, max_det=100
        )
        assert len(converted) == len(native_result) > 0
        torch.testing.assert_close(
            converted.boxes.xyxy,
            native_result.boxes.xyxy,
            rtol=0,
            atol=2e-2,
        )
        torch.testing.assert_close(
            converted.boxes.conf,
            native_result.boxes.conf,
            rtol=0,
            atol=2e-4,
        )
        torch.testing.assert_close(
            converted.boxes.cls,
            native_result.boxes.cls,
            rtol=0,
            atol=0,
        )


@pytest.mark.external_data
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_FCOS_ACCEPTANCE") != "1",
    reason="set LIBREYOLO_FCOS_ACCEPTANCE=1 to run trained export parity",
)
@pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None,
    reason="openvino is required",
)
def test_trained_openvino_runtime_parity(tmp_path) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.models.fcos.nn import FCOSExportWrapper
    from libreyolo.models.fcos.utils import preprocess_image

    checkpoint = os.environ.get("LIBREYOLO_FCOS_ACCEPTANCE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("set LIBREYOLO_FCOS_ACCEPTANCE_CHECKPOINT to LibreFCOSr50.pt")

    model = LibreYOLO(checkpoint, device="cpu")
    image_path = Path(__file__).parents[1] / "fixtures" / "dog.jpg"
    tensor, _, _, _ = preprocess_image(image_path, input_size=128)
    with torch.inference_mode():
        expected_raw = FCOSExportWrapper(model.model).eval()(tensor).numpy()

    artifact = model.export(
        "openvino",
        imgsz=128,
        dynamic=True,
        simplify=False,
        half=False,
        device="cpu",
        output_path=str(tmp_path / "LibreFCOSr50_openvino"),
    )
    backend = LibreYOLO(artifact, device="cpu")
    converted_raw = backend._run_inference(tensor.numpy())[0]
    assert converted_raw.shape == expected_raw.shape
    np.testing.assert_allclose(
        converted_raw[..., :4], expected_raw[..., :4], rtol=2e-3, atol=0.2
    )
    np.testing.assert_array_equal(converted_raw[..., 4], expected_raw[..., 4])
    np.testing.assert_allclose(
        converted_raw[..., 5:], expected_raw[..., 5:], rtol=2e-3, atol=2e-4
    )

    native_result = model.predict(image_path, imgsz=128, conf=0.4, iou=0.6, max_det=100)
    converted = backend.predict(image_path, imgsz=128, conf=0.4, iou=0.6, max_det=100)
    assert len(converted) == len(native_result) > 0
    torch.testing.assert_close(
        converted.boxes.xyxy,
        native_result.boxes.xyxy,
        rtol=0,
        atol=1.0,
    )
    torch.testing.assert_close(
        converted.boxes.conf,
        native_result.boxes.conf,
        rtol=0,
        atol=2e-4,
    )
    torch.testing.assert_close(
        converted.boxes.cls,
        native_result.boxes.cls,
        rtol=0,
        atol=0,
    )
