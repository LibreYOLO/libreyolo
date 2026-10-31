"""Real MNN conversion, fresh-load runtime, and detection parity checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import (
    compute_iou,
    match_detections,
    requires_mnn,
    requires_rfdetr,
    run_export_compare_test,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.mnn,
]

_TRAINED_DETECTORS = [
    pytest.param("yolo9", "t", marks=pytest.mark.yolo9),
    pytest.param(
        "rfdetr",
        "n",
        marks=(pytest.mark.rfdetr, requires_rfdetr),
    ),
    pytest.param("yolo9_e2e", "t", marks=pytest.mark.yolo9_e2e),
    pytest.param("ec", "s", marks=pytest.mark.ec),
    pytest.param("rtdetr", "r18", marks=pytest.mark.rtdetr),
    pytest.param("rtdetrv2", "r18", marks=pytest.mark.rtdetrv2),
    pytest.param("rtdetrv4", "s", marks=pytest.mark.rtdetrv4),
    pytest.param("dfine", "n", marks=pytest.mark.dfine),
    pytest.param("deim", "n", marks=pytest.mark.deim),
    pytest.param(
        "deimv2",
        "atto",
        marks=(pytest.mark.deimv2, pytest.mark.extended_backend),
    ),
    pytest.param("yolonas", "s", marks=pytest.mark.yolonas),
]


@requires_mnn
@pytest.mark.external_data
@pytest.mark.parametrize(("family", "size"), _TRAINED_DETECTORS)
def test_mnn_trained_export_runtime_and_detection_parity(
    family, size, sample_image, tmp_path
):
    exported_path, native_results, runtime_results = run_export_compare_test(
        family,
        size,
        sample_image,
        tmp_path,
        format="mnn",
        export_kwargs={"dynamic": False, "simplify": True},
        match_threshold=0.8,
        device="cpu",
    )

    artifact = Path(exported_path)
    sidecar = Path(f"{artifact}.json")
    assert artifact.is_file() and artifact.stat().st_size > 0
    assert sidecar.is_file()
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["model_family"] == family
    assert metadata["size"] == size
    assert metadata["task"] == "detect"
    assert metadata["format"] == "mnn"
    assert metadata["dynamic"] is False
    assert metadata["precision"] == "fp32"
    assert metadata["mnn_backend"] == "cpu"
    assert metadata["mnn_input_names"]
    assert metadata["mnn_output_names"]
    assert metadata["mnn_input_shape"][0] == metadata["mnn_batch"] == 1
    assert len(native_results) > 0
    assert len(runtime_results) == len(native_results)
    match_rate, matched, total = match_detections(native_results, runtime_results)
    assert match_rate >= 0.8, f"matched={matched}/{total}, rate={match_rate:.2%}"


def _strengthen_yolo9_p2_fixture(model) -> None:
    """Give the random P2 fixture stable, input-sensitive detection signal."""
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if "head.anchor_convs" not in name:
                continue
            if name.endswith(".weight"):
                parameter.mul_(32.0)
            elif name.endswith(".bias"):
                parameter.zero_()
        for class_tower in model.model.head.class_convs:
            class_tower[-1].weight[0].mul_(4000.0)
            class_tower[-1].weight[1].zero_()
            class_tower[-1].bias.copy_(torch.tensor([0.0, -20.0]))
    model.model.eval()


def _scalar_result(result):
    return result[0] if isinstance(result, list) else result


def _assert_one_to_one_detection_parity(expected, actual) -> None:
    expected_rows = expected.boxes.data.detach().cpu().numpy()
    actual_rows = actual.boxes.data.detach().cpu().numpy()
    assert len(expected_rows) == len(actual_rows) > 0

    remaining = set(range(len(actual_rows)))
    for expected_row in expected_rows:
        candidates = [
            index
            for index in remaining
            if int(actual_rows[index, 5]) == int(expected_row[5])
        ]
        assert candidates
        match = max(
            candidates,
            key=lambda index: compute_iou(expected_row, actual_rows[index]),
        )
        remaining.remove(match)
        assert compute_iou(expected_row, actual_rows[match]) >= 0.98
        assert abs(float(expected_row[4] - actual_rows[match, 4])) <= 0.02


@requires_mnn
@pytest.mark.yolo9
def test_mnn_yolo9_p2_deterministic_detection_parity(tmp_path):
    """Cover P2 without claiming a trained checkpoint that does not exist."""
    from libreyolo import LibreYOLO, LibreYOLO9P2

    torch.manual_seed(11)
    native = LibreYOLO9P2(None, size="t", nb_classes=2, device="cpu")
    _strengthen_yolo9_p2_fixture(native)
    artifact = native.export(
        "mnn",
        output_path=str(tmp_path / "yolo9_p2.mnn"),
        imgsz=64,
        batch=1,
        dynamic=False,
        simplify=False,
    )
    runtime = LibreYOLO(artifact, device="cpu")

    for seed in (13, 14, 15):
        image = np.random.default_rng(seed).integers(
            0, 256, (64, 64, 3), dtype=np.uint8
        )
        expected = _scalar_result(native.predict(image, imgsz=64, conf=0.1, max_det=20))
        actual = _scalar_result(runtime.predict(image, conf=0.1, max_det=20))
        _assert_one_to_one_detection_parity(expected, actual)

    assert runtime.model_family == "yolo9_p2"
    assert runtime.task == "detect"
    assert runtime.imgsz == 64
