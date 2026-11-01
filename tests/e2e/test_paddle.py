"""End-to-end Paddle export and CPU inference parity for supported cells."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from .conftest import load_model, run_export_compare_test


def _has_validated_paddle_stack() -> bool:
    if sys.version_info >= (3, 12):
        # paddlepaddle 2.6.2 cannot build X2Paddle's static graph on 3.12+.
        return False
    try:
        return (
            importlib.metadata.version("paddlepaddle") == "2.6.2"
            and importlib.metadata.version("x2paddle") == "1.6.0"
            and tuple(
                int(part) for part in importlib.metadata.version("onnx").split(".")[:2]
            )
            <= (1, 17)
        )
    except importlib.metadata.PackageNotFoundError:
        return False


requires_paddle = pytest.mark.skipif(
    not _has_validated_paddle_stack(),
    reason="validated Paddle export stack (Python 3.11 or earlier) is not installed",
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.paddle,
    pytest.mark.slow,
]


@dataclass(frozen=True)
class PaddleCase:
    weights: str
    family: str
    task: str = "detect"
    imgsz: int = 640


TRAINED_G1_CASES = (
    PaddleCase("LibreYOLO9E2Et.pt", "yolo9_e2e", imgsz=320),
    PaddleCase("LibreECs.pt", "ec"),
    PaddleCase("LibreECs-pose.pt", "ec", "pose"),
    PaddleCase("LibreECs-seg.pt", "ec", "segment"),
    PaddleCase("LibreRTDETRv4s.pt", "rtdetrv4"),
    PaddleCase("LibreDFINEn.pt", "dfine"),
    PaddleCase("LibreDEIMn.pt", "deim"),
    PaddleCase("LibreDEIMv2atto.pt", "deimv2"),
)


def _raw_arrays(output):
    if isinstance(output, torch.Tensor):
        return [output.detach().float().cpu().numpy()]
    arrays = []
    for item in output:
        arrays.extend(_raw_arrays(item))
    return arrays


def _match_public_boxes(expected, actual):
    expected_boxes = expected.boxes.data.detach().float().cpu().numpy()
    actual_boxes = actual.boxes.data.detach().float().cpu().numpy()
    matches = []
    used = set()
    for expected_index, expected_row in enumerate(expected_boxes):
        best = None
        for actual_index, actual_row in enumerate(actual_boxes):
            if actual_index in used or expected_row[5] != actual_row[5]:
                continue
            top_left = np.maximum(expected_row[:2], actual_row[:2])
            bottom_right = np.minimum(expected_row[2:4], actual_row[2:4])
            intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
            expected_area = float(
                np.prod(np.maximum(expected_row[2:4] - expected_row[:2], 0.0))
            )
            actual_area = float(
                np.prod(np.maximum(actual_row[2:4] - actual_row[:2], 0.0))
            )
            union = expected_area + actual_area - intersection
            iou = intersection / union if union else 0.0
            if best is None or iou > best[0]:
                best = (iou, actual_index)
        if best is not None and best[0] > 0.9:
            matches.append((expected_index, best[1]))
            used.add(best[1])
    return matches, max(len(expected_boxes), len(actual_boxes))


def _assert_task_public_parity(task, expected, actual, matches):
    if task == "segment":
        assert expected.masks is not None and actual.masks is not None
        expected_masks = expected.masks.data.detach().cpu().numpy() > 0.5
        actual_masks = actual.masks.data.detach().cpu().numpy() > 0.5
        for expected_index, actual_index in matches:
            intersection = np.logical_and(
                expected_masks[expected_index], actual_masks[actual_index]
            ).sum()
            union = np.logical_or(
                expected_masks[expected_index], actual_masks[actual_index]
            ).sum()
            assert not union or float(intersection / union) > 0.95
    elif task == "pose":
        assert expected.keypoints is not None and actual.keypoints is not None
        expected_keypoints = expected.keypoints.data.detach().cpu().numpy()
        actual_keypoints = actual.keypoints.data.detach().cpu().numpy()
        for expected_index, actual_index in matches:
            error = np.linalg.norm(
                expected_keypoints[expected_index, :, :2]
                - actual_keypoints[actual_index, :, :2],
                axis=1,
            )
            assert float(np.max(error)) < 0.01
            if expected_keypoints.shape[-1] > 2:
                confidence_error = np.max(
                    np.abs(
                        expected_keypoints[expected_index, :, 2]
                        - actual_keypoints[actual_index, :, 2]
                    )
                )
                assert float(confidence_error) < 1e-4


def _assert_paddle_parity(
    model,
    sample_image,
    output_path,
    *,
    imgsz,
    minimum_public_match=0.95,
):
    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import PaddleExporter

    artifact = model.export(
        format="paddle",
        output_path=str(output_path),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
        half=False,
        simplify=True,
    )
    paddle = LibreYOLO(artifact, device="cpu")
    exporter = PaddleExporter(model)
    probe = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        generator=torch.Generator().manual_seed(123),
    )
    with (
        exporter._model_context(
            torch.device("cpu"), False, False, 1, (imgsz, imgsz)
        ) as (prepared, _),
        torch.inference_mode(),
    ):
        expected_raw = _raw_arrays(prepared(probe))
    actual_raw = paddle._run_inference(probe.numpy())
    assert len(expected_raw) == len(actual_raw)
    for expected, actual in zip(expected_raw, actual_raw):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=4e-3)

    expected = model(sample_image, imgsz=imgsz, conf=0.0, max_det=25)
    actual = paddle(sample_image, conf=0.0, max_det=25)
    matches, total = _match_public_boxes(expected, actual)
    assert total > 0
    assert len(matches) / total >= minimum_public_match
    _assert_task_public_parity(model.task, expected, actual, matches)
    return Path(artifact)


@requires_paddle
def test_yolo9_paddle_export_and_cpu_parity(sample_image, tmp_path):
    exported_path, pt_results, paddle_results = run_export_compare_test(
        "yolo9",
        "t",
        sample_image,
        tmp_path,
        format="paddle",
        export_kwargs={
            "imgsz": 640,
            "batch": 1,
            "dynamic": False,
            "half": False,
            "simplify": True,
        },
        match_threshold=0.95,
        device="cpu",
    )

    artifact = Path(exported_path)
    assert artifact.is_dir()
    assert (artifact / "model.pdmodel").is_file()
    assert (artifact / "model.pdiparams").is_file()
    assert not list(artifact.glob("*.py")), "converter source leaked into artifact"

    metadata = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert metadata["model_family"] == "yolo9"
    assert metadata["task"] == "detect"
    assert metadata["precision"] == "fp32"
    assert metadata["dynamic"] is False
    assert metadata["imgsz"] == 640
    assert metadata["imgsz_h"] == 640
    assert metadata["imgsz_w"] == 640

    assert len(pt_results) > 0
    assert len(paddle_results) > 0

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import PaddleExporter

    native = load_model("yolo9", "t", device="cpu")
    paddle = LibreYOLO(exported_path, device="cpu")
    exporter = PaddleExporter(native)
    with (
        exporter._model_context(torch.device("cpu"), False, False, 1, (640, 640)) as (
            export_model,
            probe,
        ),
        torch.inference_mode(),
    ):
        expected = export_model(probe).detach().cpu().numpy()
        actual = paddle._run_inference(probe.cpu().numpy())[0]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-3)


@requires_paddle
@pytest.mark.parametrize(
    "case",
    TRAINED_G1_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_trained_g1_paddle_export_and_cpu_parity(case, sample_image, tmp_path):
    from libreyolo import LibreYOLO

    model = LibreYOLO(case.weights, device="cpu")
    assert model.FAMILY == case.family
    assert model.task == case.task
    artifact = _assert_paddle_parity(
        model,
        sample_image,
        tmp_path / f"{case.family}-{case.task}_paddle",
        imgsz=case.imgsz,
    )
    metadata = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert metadata["model_family"] == case.family
    assert metadata["task"] == case.task


@requires_paddle
def test_yolo9_p2_transfer_paddle_export_and_cpu_parity(sample_image, tmp_path):
    from libreyolo import LibreYOLO9P2
    from libreyolo.utils.download import download_weights

    torch.manual_seed(21)
    model = LibreYOLO9P2(None, size="t", device="cpu")
    weights_path = Path(model._resolve_weights_path("LibreYOLO9t.pt"))
    if not weights_path.exists():
        download_weights(str(weights_path), model.size)
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert digest == "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"
    model._load_transfer_weights(weights_path)
    model.model.eval()

    _assert_paddle_parity(
        model,
        sample_image,
        tmp_path / "yolo9-p2_paddle",
        imgsz=320,
        # The P2-only layers are randomly initialized because no trained P2
        # checkpoint exists; near-tied top-k detections are therefore sensitive
        # to sub-millipoint runtime differences despite raw tensor parity.
        minimum_public_match=0.65,
    )


@requires_paddle
@pytest.mark.parametrize("task", ("detect", "pose"))
def test_yolonas_fixture_paddle_export_and_cpu_parity(task, sample_image, tmp_path):
    from libreyolo import LibreYOLONAS

    torch.manual_seed(21)
    if task == "detect":
        model = LibreYOLONAS(None, size="s", nb_classes=2, device="cpu")
    else:
        model = LibreYOLONAS(None, size="n", task="pose", device="cpu")
    with torch.no_grad():
        for head in (
            model.model.heads.head1,
            model.model.heads.head2,
            model.model.heads.head3,
        ):
            head.reg_pred.weight.mul_(10.0)
            head.cls_pred.weight.mul_(100.0)
            head.cls_pred.bias.zero_()
            if task == "pose":
                head.pose_pred.weight.mul_(10.0)
                head.pose_pred.bias.zero_()
    model.model.eval()

    _assert_paddle_parity(
        model,
        sample_image,
        tmp_path / f"yolonas-{task}_paddle",
        imgsz=96,
    )
