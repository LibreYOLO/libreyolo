"""Mask R-CNN ONNX schema, backend parsing, and official parity."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


class _MaskDetectionStub(torch.nn.Module):
    def __init__(self, num_classes: int, labels: list[int]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("test_labels", torch.tensor(labels))

    def forward(self, images):
        del images
        count = len(self.test_labels)
        masks = torch.arange(count * 15).reshape(count, 1, 3, 5).float() / 10
        return [
            {
                "boxes": torch.arange(count * 4).reshape(count, 4).float(),
                "scores": torch.linspace(0.9, 0.7, count),
                "labels": self.test_labels,
                "masks": masks,
            }
        ]


def test_export_wrapper_keeps_masks_aligned_with_contiguous_labels():
    from libreyolo.models.mask_rcnn.nn import MaskRCNNExportWrapper

    wrapper = MaskRCNNExportWrapper(
        _MaskDetectionStub(91, [1, 12, 90]),
        include_masks=True,
    ).eval()
    boxes, scores, labels, masks = wrapper(torch.zeros(1, 3, 8, 8))

    assert boxes.shape == (2, 4)
    assert scores.shape == (2,)
    assert labels.tolist() == [0, 79]
    assert masks.shape == (2, 1, 3, 5)
    torch.testing.assert_close(masks[0], torch.arange(15).reshape(1, 3, 5) / 10)
    torch.testing.assert_close(masks[1], torch.arange(30, 45).reshape(1, 3, 5) / 10)


def test_detect_export_wrapper_omits_masks():
    from libreyolo.models.mask_rcnn.nn import MaskRCNNExportWrapper

    wrapper = MaskRCNNExportWrapper(
        _MaskDetectionStub(4, [1, 3]),
        include_masks=False,
    ).eval()
    outputs = wrapper(torch.zeros(1, 3, 8, 8))
    assert len(outputs) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_onnx_mask_paste_keeps_cuda_tensors_on_device():
    from libreyolo.models.mask_rcnn.nn import _onnx_paste_mask_in_image

    mask = torch.ones((4, 4), device="cuda")
    box = torch.tensor([1, 1, 2, 2], dtype=torch.int64, device="cuda")
    image_height = torch.scalar_tensor(4, dtype=torch.int64, device="cuda")
    image_width = torch.scalar_tensor(4, dtype=torch.int64, device="cuda")

    output = _onnx_paste_mask_in_image(
        mask,
        box,
        image_height,
        image_width,
    )

    assert output.device.type == "cuda"
    assert tuple(output.shape) == (4, 4)


def test_onnx_export_rejects_non_unit_batch_and_forces_dynamic(monkeypatch):
    from libreyolo import LibreMaskRCNN
    from libreyolo.models.base import BaseModel

    model = LibreMaskRCNN.__new__(LibreMaskRCNN)
    with pytest.raises(NotImplementedError, match="Mask R-CNN.*batch=1"):
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


def test_backend_parser_preserves_mask_alignment_and_geometry():
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend.task = "segment"
    outputs = [
        np.array(
            [[1.0, 1.0, 7.0, 3.0], [2.0, 0.0, 6.0, 4.0]],
            dtype=np.float32,
        ),
        np.array([0.9, 0.1], dtype=np.float32),
        np.array([3, 5], dtype=np.int64),
        np.array(
            [
                [[[0.0, 0.6, 0.8, 0.0], [0.0, 0.7, 0.9, 0.0]]],
                [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]],
            ],
            dtype=np.float32,
        ),
    ]

    backend._dynamic_spatial_axes = True
    boxes, scores, labels, masks = backend._parse_mask_rcnn(
        outputs, 4, orig_w=8, orig_h=2, conf=0.25
    )
    np.testing.assert_array_equal(boxes, np.array([[1, 1, 7, 2]], np.float32))
    np.testing.assert_array_equal(scores, np.array([0.9], np.float32))
    np.testing.assert_array_equal(labels, np.array([3], np.int64))
    np.testing.assert_array_equal(
        masks,
        np.array([[[False, True, True, False], [False, True, True, False]]]),
    )

    backend._dynamic_spatial_axes = False
    boxes, _, _, masks = backend._parse_mask_rcnn(
        outputs, (2, 4), orig_w=8, orig_h=4, conf=0.25
    )
    np.testing.assert_array_equal(boxes, np.array([[2, 2, 8, 4]], np.float32))
    assert masks.shape == (1, 4, 8)
    assert masks.dtype == np.bool_


def test_export_support_is_explicit_for_both_tasks_and_every_format():
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    for task in ("detect", "segment"):
        entries = {fmt: get_support("mask_rcnn", task, fmt) for fmt in EXPORT_FORMATS}
        assert entries["onnx"].tier == "validated"
        assert all(
            entry.tier == "blocked"
            for fmt, entry in entries.items()
            if fmt != "onnx"
        )


def test_export_metadata_is_single_task():
    from libreyolo.export.exporter import OnnxExporter

    model = type(
        "MaskMetadataStub",
        (),
        {
            "task": "segment",
            "SUPPORTED_TASKS": ("detect", "segment"),
            "DEFAULT_TASK": "segment",
            "size": "r50",
            "nb_classes": 80,
            "names": {0: "person"},
            "_get_model_name": lambda self: "mask_rcnn",
            "_get_input_size": lambda self: 800,
        },
    )()
    metadata = OnnxExporter(model)._build_onnx_metadata(
        dynamic=True,
        half=False,
    )
    assert metadata["task"] == "segment"
    assert json.loads(metadata["supported_tasks"]) == ["segment"]
    assert metadata["default_task"] == "segment"


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("LIBREYOLO_MASK_RCNN_ACCEPTANCE") != "1",
    reason="set LIBREYOLO_MASK_RCNN_ACCEPTANCE=1 to run official ONNX parity",
)
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx and onnxruntime are required",
)
@pytest.mark.parametrize(
    ("task", "output_names"),
    [
        ("detect", ["boxes", "scores", "labels"]),
        ("segment", ["boxes", "scores", "labels", "masks"]),
    ],
)
def test_official_r50_onnx_and_backend_parity(tmp_path, task, output_names):
    import onnx
    import onnxruntime as ort
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    from libreyolo import LibreMaskRCNN, LibreYOLO, SAMPLE_IMAGE
    from libreyolo.models.mask_rcnn.nn import MaskRCNNExportWrapper

    upstream = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).eval()
    native = LibreMaskRCNN(None, size="r50", task=task, device="cpu")
    native.model.load_state_dict(upstream.state_dict(), strict=True)
    native.model.eval()

    output_path = tmp_path / f"LibreMaskRCNNr50-{task}.onnx"
    native.export(
        "onnx",
        output_path=str(output_path),
        imgsz=128,
        batch=1,
        dynamic=True,
        device="cpu",
        simplify=False,
    )

    graph = onnx.load(str(output_path))
    assert [value.name for value in graph.graph.output] == output_names
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["model_family"] == "mask_rcnn"
    assert metadata["model_size"] == "r50"
    assert metadata["task"] == task
    assert json.loads(metadata["supported_tasks"]) == [task]

    with Image.open(SAMPLE_IMAGE) as source:
        image = source.convert("RGB")
        image.thumbnail((256, 256))
    blob = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    blob = np.ascontiguousarray(blob)

    wrapper = MaskRCNNExportWrapper(
        native.model,
        include_masks=task == "segment",
    ).eval()
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
    if task == "segment":
        np.testing.assert_allclose(actual[3], expected[3], rtol=0, atol=2e-4)

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
    if task == "segment":
        torch.testing.assert_close(
            exported_result.masks.data,
            native_result.masks.data,
            rtol=0,
            atol=0,
        )
    else:
        assert exported_result.masks is None
        assert native_result.masks is None
