"""DINO-DETR export and exported-backend contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.dinodetr]


def _stub_backend(*, nb_classes: int = 80):
    from libreyolo.backends.base import BaseBackend

    class StubBackend(BaseBackend):
        def _run_inference(self, blob: np.ndarray) -> list:
            del blob
            return []

    return StubBackend(
        model_path="fixture.onnx",
        nb_classes=nb_classes,
        device="cpu",
        imgsz=128,
        model_family="dinodetr",
        model_size="r50s5",
        names={index: f"class_{index}" for index in range(nb_classes)},
        task="detect",
        supported_tasks=("detect",),
        default_task="detect",
    )


def test_export_wrapper_returns_only_final_detection_tensors():
    from libreyolo.models.dinodetr.nn import DINODETRExportWrapper

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            return {
                "pred_logits": x.mean(dim=(-2, -1)),
                "pred_boxes": x.amax(dim=(-2, -1)),
                "aux_outputs": [{"ignored": x}],
            }

    image = torch.rand(1, 3, 8, 8)
    logits, boxes = DINODETRExportWrapper(FakeModel())(image)
    torch.testing.assert_close(logits, image.mean(dim=(-2, -1)))
    torch.testing.assert_close(boxes, image.amax(dim=(-2, -1)))


def test_backend_preprocess_parser_and_validation_adapter():
    from libreyolo.backends.base import _is_nms_free_family
    from libreyolo.models.deformable_detr.utils import preprocess_numpy
    from libreyolo.utils.coco import COCO91_CATEGORY_IDS
    from libreyolo.validation.preprocessors import DeformableDETRValPreprocessor

    backend = _stub_backend()
    image = np.random.default_rng(637).integers(0, 256, (11, 7, 3), dtype=np.uint8)
    tensor, original, original_size = backend._preprocess_deformable_detr(
        Image.fromarray(image), 128, "rgb"
    )
    expected, _ = preprocess_numpy(image, 128)
    np.testing.assert_array_equal(tensor.numpy()[0], expected)
    assert original.size == (7, 11)
    assert original_size == (7, 11)
    assert isinstance(
        backend._get_val_preprocessor(img_size=128), DeformableDETRValPreprocessor
    )

    logits = np.full((1, 8, 91), -10.0, dtype=np.float32)
    boxes = np.full((1, 8, 4), 0.5, dtype=np.float32)
    unused = sorted(set(range(91)) - set(COCO91_CATEGORY_IDS))
    logits[:, :, unused] = 10.0
    for query, category_id in enumerate(COCO91_CATEGORY_IDS[:5]):
        logits[0, query, category_id] = 5.0
    parsed_boxes, scores, classes = backend._parse_deformable_detr(
        [logits, boxes], 100, 80, conf=0.0, max_det=5
    )
    assert _is_nms_free_family("dinodetr") is True
    assert parsed_boxes.shape == (5, 4)
    assert set(classes.tolist()) == {0, 1, 2, 3, 4}
    np.testing.assert_allclose(scores, 1.0 / (1.0 + np.exp(-5.0)), atol=1e-6)


def test_export_support_is_onnx_validated_and_ncnn_blocked():
    from libreyolo.export.onnx import (
        _requires_onnx_opset17,
        _uses_dfine_style_export_wrapper,
    )
    from libreyolo.export.support import get_support

    assert _uses_dfine_style_export_wrapper("dinodetr") is True
    assert _requires_onnx_opset17("dinodetr") is True
    assert get_support("dinodetr", "detect", "onnx").tier == "validated"
    assert get_support("dinodetr", "detect", "ncnn").tier == "blocked"


def test_ncnn_export_is_rejected_before_conversion(tmp_path):
    from libreyolo import LibreDINODETR

    model = LibreDINODETR(None, size="r50", nb_classes=3, device="cpu")
    with pytest.raises(NotImplementedError, match="NCNN export is not supported"):
        model.export(
            format="ncnn",
            imgsz=240,
            output_path=str(tmp_path / "dinodetr_ncnn"),
        )


def test_onnx_export_traces_on_cpu_when_the_model_is_on_cuda(tmp_path):
    # Tracing the two-stage top-k graph on CUDA can kill the process natively.
    from libreyolo import LibreDINODETR
    from libreyolo.export.exporter import OnnxExporter

    model = LibreDINODETR(None, size="r50", nb_classes=3, device="cpu")
    exporter = OnnxExporter(model)
    with pytest.warns(RuntimeWarning, match="DINO-DETR ONNX export is traced on CPU"):
        _, device, _ = exporter._resolve_params(
            str(tmp_path / "dinodetr.onnx"), 128, "cuda", False, False
        )
    assert device == torch.device("cpu")
