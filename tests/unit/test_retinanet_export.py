"""RetinaNet ONNX graph and unified-backend parity tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.postprocess.retinanet import level_anchor_counts, resize_geometry


pytestmark = [pytest.mark.unit, pytest.mark.retinanet]


class _OutputStub(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=1, keepdim=True)


def test_export_wrapper_preserves_decoded_output():
    from libreyolo.models.retinanet.nn import RetinaNetExportWrapper

    images = torch.randn(1, 3, 8, 12)
    wrapper = RetinaNetExportWrapper(_OutputStub()).eval()
    torch.testing.assert_close(wrapper(images), images.mean(dim=1, keepdim=True))


def test_onnx_export_rejects_non_unit_batch_and_forces_dynamic(monkeypatch):
    from libreyolo import LibreRetinaNet
    from libreyolo.models.base import BaseModel

    model = LibreRetinaNet(None, size="r50", device="cpu")
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
    assert kwargs["opset"] == 13


def test_backend_candidate_selection_and_class_aware_nms():
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend.model_family = "retinanet"
    backend.task = "detect"
    backend.names = {0: "zero", 1: "one"}
    resized_h, resized_w, _, _ = resize_geometry((64, 64), 64)
    anchors = sum(level_anchor_counts(resized_h, resized_w))
    output = np.zeros((1, anchors, 6), dtype=np.float32)
    output[0, 0, :4] = [8.0, 8.0, 40.0, 40.0]
    output[0, 0, 4:] = [0.9, 0.8]

    boxes, scores, classes = backend._parse_retinanet(
        [output], 64, orig_w=64, orig_h=64, conf=0.5
    )
    result = backend._build_result(
        boxes,
        scores,
        classes,
        orig_shape=(64, 64),
        image_path=None,
        iou=0.5,
        classes=None,
        max_det=300,
    )

    assert len(result.boxes) == 2
    assert result.boxes.cls.tolist() == [0.0, 1.0]


def test_backend_preprocess_uses_upstream_resize_and_padding():
    from libreyolo.backends.onnx import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend.model_family = "retinanet"
    backend.task = "detect"
    image = Image.fromarray(np.full((40, 80, 3), 127, dtype=np.uint8), "RGB")

    tensor, _, original_size, ratio = backend._preprocess(image, 64, "rgb")

    assert tensor.shape == (1, 3, 64, 128)
    assert original_size == (80, 40)
    assert ratio == 107 / 80


def test_export_support_is_explicit_for_every_format():
    from libreyolo.export.support import EXPORT_FORMATS, get_support

    entries = {fmt: get_support("retinanet", "detect", fmt) for fmt in EXPORT_FORMATS}
    assert entries["onnx"].tier == "validated"
    # Axelera compiles a fixed square canvas (supported, measured drop).
    assert entries["axelera"].tier == "available"
    assert all(
        entry.tier == "blocked"
        for fmt, entry in entries.items()
        if fmt not in {"onnx", "axelera"}
    )


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("LIBREYOLO_RETINANET_CHECKPOINT_DIR"),
    reason="set LIBREYOLO_RETINANET_CHECKPOINT_DIR for official ONNX parity",
)
@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx and onnxruntime are required",
)
def test_official_r50_onnx_and_backend_parity(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreRetinaNet, LibreYOLO, SAMPLE_IMAGE
    from libreyolo.models.retinanet.nn import RetinaNetExportWrapper
    from libreyolo.models.retinanet.utils import preprocess_image

    checkpoint_dir = Path(os.environ["LIBREYOLO_RETINANET_CHECKPOINT_DIR"])
    state = torch.load(
        checkpoint_dir / "retinanet_resnet50_fpn_coco-eeacb38b.pth",
        map_location="cpu",
        weights_only=True,
    )
    native = LibreRetinaNet(None, size="r50", device="cpu")
    native.model.load_state_dict(state, strict=True)
    native.model.eval()

    output_path = tmp_path / "LibreRetinaNetr50.onnx"
    native.export(
        "onnx",
        output_path=str(output_path),
        imgsz=800,
        batch=1,
        dynamic=True,
        simplify=False,
        device="cpu",
    )

    graph = onnx.load(str(output_path))
    assert [value.name for value in graph.graph.output] == ["output"]
    metadata = {item.key: item.value for item in graph.metadata_props}
    assert metadata["model_family"] == "retinanet"
    assert metadata["model_size"] == "r50"
    assert metadata["nc"] == "80"

    with Image.open(SAMPLE_IMAGE) as source:
        image = source.convert("RGB")
    blob, _, _, _ = preprocess_image(image, input_size=800, color_format="rgb")

    wrapper = RetinaNetExportWrapper(native.model).eval()
    with torch.inference_mode():
        expected = wrapper(blob).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, "height", "width"]
    actual = session.run(None, {"images": blob.numpy()})[0]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual[:, :, :4], expected[:, :, :4], rtol=0, atol=5e-3)
    np.testing.assert_allclose(actual[:, :, 4:], expected[:, :, 4:], rtol=0, atol=2e-5)

    exported = LibreYOLO(str(output_path), device="cpu")
    native_result = native.predict(image, conf=0.25, verbose=False)
    exported_result = exported.predict(image, conf=0.25, verbose=False)
    assert len(exported_result) == len(native_result) > 0
    torch.testing.assert_close(
        exported_result.boxes.xyxy, native_result.boxes.xyxy, rtol=0, atol=5e-3
    )
    torch.testing.assert_close(
        exported_result.boxes.conf, native_result.boxes.conf, rtol=0, atol=2e-5
    )
    torch.testing.assert_close(
        exported_result.boxes.cls, native_result.boxes.cls, rtol=0, atol=0
    )
