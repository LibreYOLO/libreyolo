"""Faster R-CNN fixed-batch ONNX export and backend parity."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


class _DetectionStub(torch.nn.Module):
    def __init__(self, num_classes: int, labels: list[int]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("test_labels", torch.tensor(labels))

    def forward(self, images):
        del images
        count = len(self.test_labels)
        return [
            {
                "boxes": torch.arange(count * 4).reshape(count, 4).float(),
                "scores": torch.linspace(0.9, 0.7, count),
                "labels": self.test_labels,
            }
        ]


@pytest.mark.parametrize(
    ("num_classes", "source_labels", "expected_labels"),
    [
        (91, [1, 12, 90], [0, 79]),
        (4, [1, 3], [0, 2]),
    ],
)
def test_export_wrapper_emits_contiguous_labels(
    num_classes, source_labels, expected_labels
):
    from libreyolo.models.faster_rcnn.nn import FasterRCNNExportWrapper

    wrapper = FasterRCNNExportWrapper(_DetectionStub(num_classes, source_labels)).eval()
    boxes, scores, labels = wrapper(torch.zeros(1, 3, 8, 8))

    assert boxes.shape == (len(expected_labels), 4)
    assert scores.shape == (len(expected_labels),)
    assert labels.tolist() == expected_labels


def test_onnx_export_rejects_non_unit_batch_and_forces_dynamic(monkeypatch):
    from libreyolo import LibreFasterRCNN
    from libreyolo.models.base import BaseModel

    model = LibreFasterRCNN(None, size="n", device="cpu")
    with pytest.raises(NotImplementedError, match="batch=1"):
        model.export("onnx", batch=2)

    monkeypatch.setattr(
        BaseModel,
        "export",
        lambda self, format="onnx", **kwargs: (format, kwargs),
    )
    with pytest.warns(RuntimeWarning, match="forcing dynamic=True"):
        export_format, kwargs = model.export("onnx", dynamic=False)
    assert export_format == "onnx"
    assert kwargs["dynamic"] is True
    assert kwargs["opset"] == 18


def test_backend_geometry_distinguishes_dynamic_and_fixed_graphs():
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    outputs = [
        np.array([[32.0, 64.0, 288.0, 128.0]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([3], dtype=np.int64),
    ]

    backend._dynamic_spatial_axes = True
    boxes, scores, labels = backend._parse_faster_rcnn(
        outputs, 320, orig_w=640, orig_h=160, conf=0.25
    )
    np.testing.assert_array_equal(boxes, outputs[0])
    np.testing.assert_array_equal(scores, outputs[1])
    np.testing.assert_array_equal(labels, outputs[2])

    backend._dynamic_spatial_axes = False
    boxes, _, _ = backend._parse_faster_rcnn(
        outputs, 320, orig_w=640, orig_h=160, conf=0.25
    )
    np.testing.assert_array_equal(
        boxes,
        np.array([[64.0, 32.0, 576.0, 64.0]], dtype=np.float32),
    )


def test_dynamic_backend_preprocess_keeps_source_shape():
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend._dynamic_spatial_axes = True
    image = Image.fromarray(np.full((19, 31, 3), 127, dtype=np.uint8), "RGB")

    tensor, _, original_size, ratio = backend._preprocess_faster_rcnn(
        image, 320, "rgb"
    )

    assert tensor.shape == (1, 3, 19, 31)
    assert original_size == (31, 19)
    assert ratio == 1.0
    np.testing.assert_allclose(tensor.numpy(), 127 / 255.0, rtol=0, atol=1e-7)


def test_export_support_is_explicit_for_every_format():
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    entries = {fmt: get_support("faster_rcnn", "detect", fmt) for fmt in EXPORT_FORMATS}
    assert entries["onnx"].tier == "validated"
    assert all(
        entry.tier == "blocked" for fmt, entry in entries.items() if fmt != "onnx"
    )


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_FASTER_RCNN_ACCEPTANCE") != "1",
    reason=(
        "set LIBREYOLO_FASTER_RCNN_ACCEPTANCE=1 to run official-checkpoint "
        "ONNX parity"
    ),
)
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx and onnxruntime are required",
)
def test_official_n_onnx_and_backend_parity(tmp_path):
    import onnx
    import onnxruntime as ort
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    from libreyolo import LibreFasterRCNN, LibreYOLO, SAMPLE_IMAGE
    from libreyolo.models.faster_rcnn.nn import FasterRCNNExportWrapper

    upstream = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    ).eval()
    native = LibreFasterRCNN(None, size="n", device="cpu")
    native.model.load_state_dict(upstream.state_dict(), strict=True)
    native.model.eval()

    output_path = tmp_path / "LibreFasterRCNNn.onnx"
    native.export(
        "onnx",
        output_path=str(output_path),
        imgsz=320,
        batch=1,
        dynamic=True,
        device="cpu",
    )

    graph = onnx.load(str(output_path))
    assert [value.name for value in graph.graph.output] == [
        "boxes",
        "scores",
        "labels",
    ]
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["model_family"] == "faster_rcnn"
    assert metadata["model_size"] == "n"
    assert metadata["nc"] == "80"

    with Image.open(SAMPLE_IMAGE) as source:
        image = source.convert("RGB")
    blob = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    blob = np.ascontiguousarray(blob)

    wrapper = FasterRCNNExportWrapper(native.model).eval()
    with torch.inference_mode():
        expected = tuple(value.numpy() for value in wrapper(torch.from_numpy(blob)))
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, "height", "width"]
    actual = tuple(session.run(None, {"images": blob}))

    assert expected[0].shape[0] > 0
    assert [value.shape for value in actual] == [value.shape for value in expected]
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=5e-3)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(actual[2], expected[2])

    exported = LibreYOLO(str(output_path), device="cpu")
    native_result = native.predict(image, conf=0.25, verbose=False)
    exported_result = exported.predict(image, conf=0.25, verbose=False)
    assert len(exported_result) == len(native_result) > 0
    torch.testing.assert_close(
        exported_result.boxes.xyxy,
        native_result.boxes.xyxy,
        rtol=0,
        atol=5e-3,
    )
    torch.testing.assert_close(
        exported_result.boxes.conf,
        native_result.boxes.conf,
        rtol=0,
        atol=2e-5,
    )
    torch.testing.assert_close(
        exported_result.boxes.cls,
        native_result.boxes.cls,
        rtol=0,
        atol=0,
    )
