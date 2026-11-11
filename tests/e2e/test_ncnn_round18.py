"""Round 18 NCNN parity for the remaining permissive detector claims.

Seven published checkpoints are permissively licensed. YOLO2 and YOLO4 use
public-domain Darknet checkpoints, YOLO9-P2 uses the SHA-pinned MIT YOLO9
transfer fixture, and YOLO-NAS detect/pose use deterministic synthetic
training because the published YOLO-NAS artifacts are proprietary. Synthetic
fixtures validate conversion behavior, not task accuracy.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from .test_ncnn_round17 import _native_outputs

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.ncnn,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("pnnx") is None
        or importlib.util.find_spec("ncnn") is None,
        reason="PNNX and NCNN are required",
    ),
]


@dataclass(frozen=True)
class NCNNRound18Case:
    family: str
    task: str
    imgsz: int
    weights: str | None = None
    fixture: str | None = None


def _network_case(case: NCNNRound18Case):
    return pytest.param(case, marks=pytest.mark.network)


ROUND18_CASES = (
    _network_case(
        NCNNRound18Case("picodet", "detect", 320, weights="LibrePICODETs.pt")
    ),
    _network_case(
        NCNNRound18Case("yolo1", "detect", 448, weights="LibreYOLO1b.pt")
    ),
    pytest.param(
        NCNNRound18Case(
            "yolo2",
            "detect",
            608,
            weights="LibreYOLO2b.pt",
        ),
        marks=(
            pytest.mark.network,
            pytest.mark.skipif(
                sys.platform == "win32",
                reason=(
                    "NCNN 20260526 terminates the Windows runtime with a "
                    "native integer divide-by-zero during output extraction."
                ),
            ),
        ),
    ),
    _network_case(
        NCNNRound18Case("yolo3", "detect", 416, weights="LibreYOLO3t.pt")
    ),
    _network_case(
        NCNNRound18Case("yolo4", "detect", 608, weights="LibreYOLO4b.pt")
    ),
    _network_case(
        NCNNRound18Case("yolo7", "detect", 640, weights="LibreYOLO7b.pt")
    ),
    _network_case(
        NCNNRound18Case(
            "yolo9_e2e",
            "detect",
            640,
            weights="LibreYOLO9E2Et.pt",
        )
    ),
    _network_case(
        NCNNRound18Case("yolox", "detect", 416, weights="LibreYOLOXn.pt")
    ),
    pytest.param(
        NCNNRound18Case(
            "yolo9_p2",
            "detect",
            640,
            fixture="yolo9_transfer",
        ),
        marks=(
            pytest.mark.network,
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "The MIT YOLO9 transfer fixture has raw NCNN parity but "
                    "changes near-noise public top-k classes; it produces no "
                    "detections above 0.05 on the bundled real image."
                ),
            ),
        ),
    ),
    NCNNRound18Case("yolonas", "detect", 96, fixture="synthetic_detect"),
    NCNNRound18Case("yolonas", "pose", 96, fixture="synthetic_pose"),
)


def _build_yolo9_p2_transfer():
    from libreyolo import LibreYOLO9P2
    from libreyolo.utils.download import download_weights

    model = LibreYOLO9P2(None, size="t", device="cpu")
    weights_path = Path(model._resolve_weights_path("LibreYOLO9t.pt"))
    if not weights_path.exists():
        download_weights(str(weights_path), model.size)
    with weights_path.open("rb") as weights_file:
        digest = hashlib.file_digest(weights_file, "sha256").hexdigest()
    assert digest == "b4d7e93f9e0393830fb42e6135c0e3464b2673b05e5ecf4b7f2374ec18e39eb2"
    model._load_transfer_weights(weights_path)
    model.model.eval()
    return model


def _build_synthetic_yolonas_detect(imgsz: int):
    from libreyolo import LibreYOLONAS
    from libreyolo.models.yolonas.loss import PPYoloELoss

    model = LibreYOLONAS(None, size="s", nb_classes=2, device="cpu")
    network = model.model.train()
    loss_fn = PPYoloELoss(num_classes=2)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    for step in range(12):
        images = torch.rand(2, 3, imgsz, imgsz)
        targets = torch.zeros(2, 10, 5)
        targets[0, 0] = torch.tensor(
            [float(step % 2), 36.0 + step, 42.0, 24.0, 30.0]
        )
        targets[1, 0] = torch.tensor(
            [float((step + 1) % 2), 64.0, 52.0, 20.0, 26.0]
        )
        loss, _ = loss_fn(network(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)
    return model


def _build_synthetic_yolonas_pose(imgsz: int):
    from libreyolo import LibreYOLONAS
    from libreyolo.data import default_oks_sigmas
    from libreyolo.models.yolonas.loss import YoloNASPoseLoss

    model = LibreYOLONAS(None, size="n", task="pose", device="cpu")
    network = model.model.train()
    keypoints = model.num_keypoints
    loss_fn = YoloNASPoseLoss(oks_sigmas=default_oks_sigmas(keypoints))
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    for step in range(8):
        images = torch.rand(2, 3, imgsz, imgsz)
        targets = torch.zeros(2, 4, 5 + 3 * keypoints)
        for batch_index in range(2):
            cx = 40.0 + 8.0 * batch_index + float(step % 3)
            cy = 44.0 + 4.0 * batch_index
            targets[batch_index, 0, 1:5] = torch.tensor(
                [cx, cy, 30.0, 36.0]
            )
            for keypoint_index in range(keypoints):
                offset = (keypoint_index - keypoints / 2.0) * 0.8
                start = 5 + 3 * keypoint_index
                targets[batch_index, 0, start : start + 3] = torch.tensor(
                    [cx + offset, cy + offset, 2.0]
                )
        loss, _ = loss_fn(network(images), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        for head in (
            network.heads.head1,
            network.heads.head2,
            network.heads.head3,
        ):
            head.reg_pred.weight.mul_(20.0)
            head.pose_pred.weight.mul_(20.0)
    return model


def _build_model(case: NCNNRound18Case):
    if case.weights is not None:
        from libreyolo import LibreYOLO

        model = LibreYOLO(case.weights, device="cpu")
    elif case.fixture == "yolo9_transfer":
        model = _build_yolo9_p2_transfer()
    elif case.fixture == "synthetic_detect":
        model = _build_synthetic_yolonas_detect(case.imgsz)
    elif case.fixture == "synthetic_pose":
        model = _build_synthetic_yolonas_pose(case.imgsz)
    else:
        raise AssertionError(f"Round 18 case has no fixture: {case}")
    model.model.eval()
    assert model.FAMILY == case.family
    assert model.task == case.task
    return model


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _assert_raw_parity(
    expected_outputs: list[np.ndarray],
    actual_outputs: list[np.ndarray],
) -> None:
    assert len(actual_outputs) == len(expected_outputs)
    for expected, actual in zip(expected_outputs, actual_outputs):
        assert actual.shape == expected.shape
        matches = np.isclose(actual, expected, rtol=2e-3, atol=2e-2)
        assert float(matches.mean()) > 0.95


def _assert_signal_margin(
    expected_first: list[np.ndarray],
    expected_second: list[np.ndarray],
    actual_first: list[np.ndarray],
    actual_second: list[np.ndarray],
) -> None:
    for reference_a, reference_b, converted_a, converted_b in zip(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    ):
        expected_signal = _rms(reference_a - reference_b)
        actual_signal = _rms(converted_a - converted_b)
        parity_error = max(
            _rms(reference_a - converted_a),
            _rms(reference_b - converted_b),
        )
        assert expected_signal > 1e-8
        assert actual_signal > 20.0 * max(parity_error, 1e-12)


def _match_detection_rows(expected_data, actual_data):
    expected_classes = expected_data[:, 5].astype(np.int64)
    actual_classes = actual_data[:, 5].astype(np.int64)
    assert sorted(expected_classes.tolist()) == sorted(actual_classes.tolist())
    matches = []
    for class_id in np.unique(expected_classes):
        expected_indices = np.flatnonzero(expected_classes == class_id)
        actual_indices = np.flatnonzero(actual_classes == class_id)
        expected_rows = expected_data[expected_indices]
        actual_rows = actual_data[actual_indices]
        cost = np.max(
            np.abs(expected_rows[:, None, :4] - actual_rows[None, :, :4]),
            axis=2,
        )
        expected_rows_idx, actual_rows_idx = linear_sum_assignment(cost)
        for expected_index, actual_index in zip(
            expected_rows_idx,
            actual_rows_idx,
        ):
            expected_global = int(expected_indices[expected_index])
            actual_global = int(actual_indices[actual_index])
            np.testing.assert_allclose(
                actual_rows[actual_index, :4],
                expected_rows[expected_index, :4],
                rtol=2e-3,
                atol=1.0,
            )
            assert (
                abs(
                    float(actual_rows[actual_index, 4])
                    - float(expected_rows[expected_index, 4])
                )
                < 0.02
            )
            matches.append((expected_global, actual_global))
    return matches


def _assert_predict_parity(case, model, backend) -> None:
    if case.weights is not None:
        source = str(Path("tests/data/parkour.jpg"))
        confidence = 0.05
    else:
        source = np.random.default_rng(118).integers(
            0,
            256,
            size=(case.imgsz + 8, case.imgsz + 16, 3),
            dtype=np.uint8,
        )
        confidence = 0.0
    expected = model.predict(
        source,
        imgsz=case.imgsz,
        conf=confidence,
        max_det=25,
    )
    actual = backend.predict(source, conf=confidence, max_det=25)
    expected_data = expected.boxes.data.detach().float().cpu().numpy()
    actual_data = actual.boxes.data.detach().float().cpu().numpy()
    assert len(expected_data) == len(actual_data)
    assert len(expected_data) > 0
    matches = _match_detection_rows(expected_data, actual_data)

    if case.task != "pose":
        return
    expected_keypoints = expected.keypoints.data.detach().float().cpu().numpy()
    actual_keypoints = actual.keypoints.data.detach().float().cpu().numpy()
    for expected_index, actual_index in matches:
        coordinate_error = np.linalg.norm(
            expected_keypoints[expected_index, :, :2]
            - actual_keypoints[actual_index, :, :2],
            axis=1,
        )
        assert float(np.max(coordinate_error)) < 2.0
        if expected_keypoints.shape[-1] > 2:
            confidence_error = np.max(
                np.abs(
                    expected_keypoints[expected_index, :, 2]
                    - actual_keypoints[actual_index, :, 2]
                )
            )
            assert float(confidence_error) < 0.02


def _run_case(tmp_path, case: NCNNRound18Case) -> None:
    from libreyolo import LibreYOLO

    torch.manual_seed(18)
    model = _build_model(case)
    generator = torch.Generator().manual_seed(118)
    first = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=generator,
    )
    second = torch.rand(
        1,
        3,
        case.imgsz,
        case.imgsz,
        generator=generator,
    )
    expected_first = _native_outputs(model, first, case.imgsz)
    expected_second = _native_outputs(model, second, case.imgsz)

    artifact = model.export(
        format="ncnn",
        output_path=str(tmp_path / f"{case.family}_{case.task}_ncnn"),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")
    actual_first = backend._run_inference(first.numpy())
    actual_second = backend._run_inference(second.numpy())
    _assert_raw_parity(expected_first, actual_first)
    _assert_raw_parity(expected_second, actual_second)
    _assert_signal_margin(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    )

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(case, model, backend)
    del backend, model
    gc.collect()


@pytest.mark.parametrize(
    "case",
    ROUND18_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_ncnn_round18_raw_and_predict_parity(tmp_path, case):
    _run_case(tmp_path, case)
