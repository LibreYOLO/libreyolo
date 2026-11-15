# SPDX-License-Identifier: Apache-2.0
# Official weights: SenseTime Deformable DETR Hugging Face mirrors (Apache-2.0).
"""ONNX Runtime parity for all five official Deformable DETR variants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.onnx,
    pytest.mark.deformable_detr,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
]

OFFICIAL_CASES = (
    (
        "r50ss",
        "SenseTime/deformable-detr-single-scale",
        "e880a4ca7bbe47b33d37ed90e2948efbbdad0d44",
        "82eeb57bbcdd02408afc53d5f5c874e3a7f27b5034194ae2c4475d06fceaa59b",
    ),
    (
        "r50ssdc5",
        "SenseTime/deformable-detr-single-scale-dc5",
        "c23332913d0ae1a8c98725e308eccba65a5933cc",
        "e71afa5f5900e2e769275156494195508efcadaab4275b0cd4c80f10369dc090",
    ),
    (
        "r50",
        "SenseTime/deformable-detr",
        "83ecd26945199939cb82806f988debdb71e6f43e",
        "caf1e3e61283c6ce35cd2d9adaa7033cf40997d4dfe434003bcdb9085cc8cf9b",
    ),
    (
        "r50refine",
        "SenseTime/deformable-detr-with-box-refine",
        "2e9e461623a8fdc296e19666c46c8a4389a3a6fe",
        "4113700fe8aade398808424b7c5c1304cfbf886adc6450a6ca5d50a702be3373",
    ),
    (
        "r50twostage",
        "SenseTime/deformable-detr-with-box-refine-two-stage",
        "e74bff70d69f3e825f6cefaf179bfba707f92054",
        "411bb4238a834d40fff651b1b5b7d6dd80c2dd28be1747eec7b6918674e85de6",
    ),
)


def _matched_predictions(converted: np.ndarray, native: np.ndarray) -> np.ndarray:
    """Order converted detections to their native geometric/class matches."""
    assert converted.shape == native.shape
    if len(native) == 0:
        return converted

    left = converted[:, None, :4]
    right = native[None, :, :4]
    intersection_xy1 = np.maximum(left[..., :2], right[..., :2])
    intersection_xy2 = np.minimum(left[..., 2:], right[..., 2:])
    intersection_wh = np.clip(intersection_xy2 - intersection_xy1, 0.0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    left_area = np.prod(np.clip(left[..., 2:] - left[..., :2], 0.0, None), axis=-1)
    right_area = np.prod(np.clip(right[..., 2:] - right[..., :2], 0.0, None), axis=-1)
    iou = intersection / np.clip(left_area + right_area - intersection, 1e-9, None)
    class_mismatch = converted[:, None, 5] != native[None, :, 5]
    converted_indices, native_indices = linear_sum_assignment(
        1.0 - iou + class_mismatch * 10.0
    )
    return converted[converted_indices[np.argsort(native_indices)]]


def _aligned_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection_xy1 = np.maximum(left[:, :2], right[:, :2])
    intersection_xy2 = np.minimum(left[:, 2:], right[:, 2:])
    intersection_wh = np.clip(intersection_xy2 - intersection_xy1, 0.0, None)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    left_area = np.prod(np.clip(left[:, 2:] - left[:, :2], 0.0, None), axis=-1)
    right_area = np.prod(np.clip(right[:, 2:] - right[:, :2], 0.0, None), axis=-1)
    return intersection / np.clip(left_area + right_area - intersection, 1e-9, None)


@pytest.mark.parametrize(
    ("size", "repo_id", "revision", "expected_sha256"),
    OFFICIAL_CASES,
    ids=[case[0] for case in OFFICIAL_CASES],
)
def test_official_checkpoint_onnx_predict_parity(
    tmp_path, size, repo_id, revision, expected_sha256
):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    safetensors = pytest.importorskip("safetensors.torch")

    from libreyolo import LibreDeformableDETR, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter
    from libreyolo.models.deformable_detr.conversion import (
        convert_hf_deformable_detr_state_dict,
    )

    source = Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            revision=revision,
            token=False,
        )
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256
    state_dict = convert_hf_deformable_detr_state_dict(
        safetensors.load_file(str(source), device="cpu")
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreDeformableDETR(None, size=size, device=device)
    model.model.load_state_dict(state_dict, strict=True)
    model.model.eval()
    del state_dict

    image = Path("tests/data/parkour.jpg")
    input_tensor, _, _, _ = model._preprocess(image, input_size=800)
    trace_device = "cpu" if size == "r50twostage" else device
    trace_tensor = input_tensor.to(trace_device)
    exporter = OnnxExporter(model)
    with (
        exporter._model_context(trace_device, False, False, 1, (800, 800)) as (
            wrapped,
            _,
        ),
        torch.inference_mode(),
    ):
        native_raw = tuple(
            output.detach().cpu().numpy() for output in wrapped(trace_tensor)
        )

    artifact = model.export(
        format="onnx",
        imgsz=800,
        dynamic=False,
        simplify=False,
        device=device,
        output_path=str(tmp_path / f"LibreDeformableDETR{size}.onnx"),
    )
    graph = onnx.load(artifact)
    onnx.checker.check_model(graph)
    assert [output.name for output in graph.graph.output] == [
        "pred_logits",
        "pred_boxes",
    ]

    backend = LibreYOLO(artifact, device="cpu")
    converted_raw = backend._run_inference(input_tensor.numpy())
    assert len(converted_raw) == len(native_raw) == 2
    logit_error = np.abs(converted_raw[0] - native_raw[0])
    box_error = np.abs(converted_raw[1] - native_raw[1])
    assert float(logit_error.mean()) < 0.01
    assert float(np.quantile(logit_error, 0.99)) < 0.05
    assert float(box_error.mean()) < 0.002
    assert float(np.quantile(box_error, 0.99)) < 0.02

    native = model.predict(image, imgsz=800, conf=0.25, max_det=100).boxes.data
    converted = backend.predict(image, conf=0.25, max_det=100).boxes.data
    native = native.detach().cpu().numpy()
    converted = converted.detach().cpu().numpy()
    assert len(native) > 0
    converted = _matched_predictions(converted, native)
    np.testing.assert_array_equal(converted[:, 5], native[:, 5])
    assert float(_aligned_iou(converted[:, :4], native[:, :4]).min()) > 0.95
    assert float(np.abs(converted[:, 4] - native[:, 4]).mean()) < 0.01
