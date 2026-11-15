# SPDX-License-Identifier: Apache-2.0
# Official weights: IDEA-Research/DINO release (Apache-2.0 implied basis).
"""ONNX Runtime parity for all three official DINO-DETR variants."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from libreyolo.utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.onnx,
    pytest.mark.dinodetr,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.slow,
]

OFFICIAL_CASES = (
    (
        "r50",
        "LibreYOLO/LibreDINODETRr50",
        "462f5afabb53146d933827814199564a9bd6ed93",
        "8b2243075a086e17c898d80ceb939784b1b56d44c5aca26256b4914f3b8d5d03",
        240,
    ),
    (
        "r50s5",
        "LibreYOLO/LibreDINODETRr50s5",
        "7d04c21564296ed31385c2f93db749a568940ab1",
        "8dd59b36fff9750835fac7eb14c07a00f244bc0ec3f205dceac74907f0ef723a",
        128,
    ),
    (
        "swinl",
        "LibreYOLO/LibreDINODETRswinl",
        "3bc6420403413741e224529ff58dd6220e902220",
        "1532135001dff0fa6ba688eac52df9d92af83c2c6bb13a06139fbfcd81574118",
        128,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    ("size", "repo_id", "revision", "expected_sha256", "export_size"),
    OFFICIAL_CASES,
    ids=[case[0] for case in OFFICIAL_CASES],
)
def test_official_checkpoint_onnx_predict_parity(
    tmp_path, size, repo_id, revision, expected_sha256, export_size
):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    huggingface_hub = pytest.importorskip("huggingface_hub")

    from libreyolo import LibreDINODETR, LibreYOLO
    from libreyolo.export.exporter import OnnxExporter

    filename = f"LibreDINODETR{size}.pt"
    source = Path(
        huggingface_hub.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=False,
        )
    )
    assert _sha256(source) == expected_sha256
    checkpoint = load_untrusted_torch_file(source)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    del checkpoint
    gc.collect()

    # The legacy PyTorch exporter can terminate the Windows process while
    # lowering this CUDA graph. CPU traces the same operators and is portable.
    device = "cpu"
    model = LibreDINODETR(str(source), size=size, device=device)
    image = Path("tests/data/parkour.jpg")
    input_tensor, _, _, _ = model._preprocess(image, input_size=export_size)
    exporter = OnnxExporter(model)
    with (
        exporter._model_context(device, False, False, 1, (export_size, export_size))
        as (wrapped, _),
        torch.inference_mode(),
    ):
        native_raw = tuple(
            output.detach().cpu().numpy()
            for output in wrapped(input_tensor.to(device))
        )

    artifact = model.export(
        format="onnx",
        imgsz=export_size,
        dynamic=False,
        simplify=False,
        device=device,
        output_path=str(tmp_path / f"LibreDINODETR{size}.onnx"),
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
    # Swin-L has near-tied, low-scoring encoder proposals. PyTorch and ONNX
    # Runtime can select a handful of different background queries at top-k,
    # so bound the raw error distribution and verify the stricter public
    # prediction contract below instead of requiring every query slot to match.
    assert float(logit_error.mean()) < 0.01
    assert float(np.quantile(logit_error, 0.99)) < 0.05
    assert float(box_error.mean()) < 0.002
    assert float(np.quantile(box_error, 0.99)) < 0.02

    native = model.predict(
        image, imgsz=export_size, conf=0.1, max_det=100
    ).boxes.data
    converted = backend.predict(image, conf=0.1, max_det=100).boxes.data
    native = native.detach().cpu().numpy()
    converted = converted.detach().cpu().numpy()
    assert len(native) > 0
    converted = _matched_predictions(converted, native)
    np.testing.assert_array_equal(converted[:, 5], native[:, 5])
    assert float(_aligned_iou(converted[:, :4], native[:, :4]).min()) > 0.995
    assert float(np.abs(converted[:, 4] - native[:, 4]).max()) < 0.002
