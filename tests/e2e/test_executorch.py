"""Real ExecuTorch XNNPACK export/runtime checks for detection flagships."""

from __future__ import annotations

import importlib.resources
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.executorch,
    pytest.mark.export_backend,
]


def _require_executorch(monkeypatch):
    pytest.importorskip("executorch")
    pytest.importorskip("executorch.runtime")
    if shutil.which("flatc") is not None:
        return

    # ExecuTorch Windows wheels bundle flatc but 1.2 does not add its
    # directory to PATH. Keep this host-tool workaround local to the real
    # toolchain test; production emits an actionable error.
    package_root = importlib.resources.files("executorch")
    bundled = Path(str(package_root.joinpath("data", "bin", "flatc.exe")))
    if not bundled.exists():
        pytest.skip("ExecuTorch lowering requires flatc on PATH")
    monkeypatch.setenv("PATH", f"{bundled.parent}{os.pathsep}{os.environ['PATH']}")


def _detections(result) -> np.ndarray:
    """Return postprocessed detection rows for list and scalar Results APIs."""
    if isinstance(result, list):
        result = result[0]
    return result.boxes.data.detach().cpu().numpy()


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def _align_unordered_queries(reference, candidate):
    """Apply one logits-and-box assignment to every query-indexed output."""
    reference_parts = []
    candidate_parts = []
    for expected, actual in zip(reference[:2], candidate[:2]):
        assert expected.ndim >= 3 and actual.ndim >= 3
        scale = max(float(np.abs(expected).max()), 1e-12)
        reference_parts.append(expected[0].reshape(expected.shape[1], -1) / scale)
        candidate_parts.append(actual[0].reshape(actual.shape[1], -1) / scale)
    reference_key = np.concatenate(reference_parts, axis=1)
    candidate_key = np.concatenate(candidate_parts, axis=1)
    cost = np.max(
        np.abs(reference_key[:, None, :] - candidate_key[None, :, :]),
        axis=2,
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    return [output[:, order, ...] for output in candidate]


def _scalar_result(result):
    return result[0] if isinstance(result, list) else result


def _psnr(expected: np.ndarray, actual: np.ndarray, peak: float) -> float:
    error = expected.astype(np.float64) - actual.astype(np.float64)
    mse = float(np.mean(np.square(error)))
    return float("inf") if mse == 0 else 20.0 * np.log10(peak / np.sqrt(mse))


def _match_detection_rows(expected, actual):
    expected_data = expected.boxes.data.detach().float().cpu().numpy()
    actual_data = actual.boxes.data.detach().float().cpu().numpy()
    assert len(expected_data) == len(actual_data)
    assert len(expected_data) > 0

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
        expected_order, actual_order = linear_sum_assignment(cost)
        for expected_index, actual_index in zip(expected_order, actual_order):
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
            matches.append(
                (
                    int(expected_indices[expected_index]),
                    int(actual_indices[actual_index]),
                )
            )
    return matches


def _assert_point_parity(expected, actual):
    expected_data = expected.points.data.detach().float().cpu().numpy()
    actual_data = actual.points.data.detach().float().cpu().numpy()
    assert len(expected_data) == len(actual_data)
    assert len(expected_data) > 0
    expected_classes = expected_data[:, 2].astype(np.int64)
    actual_classes = actual_data[:, 2].astype(np.int64)
    assert sorted(expected_classes.tolist()) == sorted(actual_classes.tolist())
    for class_id in np.unique(expected_classes):
        expected_rows = expected_data[expected_classes == class_id]
        actual_rows = actual_data[actual_classes == class_id]
        cost = np.max(
            np.abs(expected_rows[:, None, :2] - actual_rows[None, :, :2]),
            axis=2,
        )
        expected_order, actual_order = linear_sum_assignment(cost)
        np.testing.assert_allclose(
            actual_rows[actual_order, :2],
            expected_rows[expected_order, :2],
            rtol=1e-3,
            atol=1.0,
        )
        np.testing.assert_allclose(
            actual_rows[actual_order, 3],
            expected_rows[expected_order, 3],
            rtol=2e-3,
            atol=0.02,
        )


def _assert_public_task_parity(case, native, runtime, image, imgsz):
    if case == "l2cs_gaze":
        height, width = image.shape[:2]
        expected = _scalar_result(
            native.predict(
                image,
                face_boxes=[(0, 0, width, height)],
            )
        )
        actual = _scalar_result(runtime.predict(image))
        np.testing.assert_allclose(
            actual.gaze.data.detach().float().cpu().numpy(),
            expected.gaze.data.detach().float().cpu().numpy(),
            rtol=1e-3,
            atol=1e-4,
        )
        return

    confidence = 0.1 if case == "fomo_point" else 0.0
    expected = _scalar_result(
        native.predict(
            image,
            imgsz=imgsz,
            conf=confidence,
            max_det=10,
        )
    )
    actual = _scalar_result(
        runtime.predict(
            image,
            conf=confidence,
            max_det=10,
        )
    )

    if case in {"convnext_classify", "dinov2_classify"}:
        expected_probs = expected.probs.data.detach().float().cpu()
        actual_probs = actual.probs.data.detach().float().cpu()
        cosine = torch.nn.functional.cosine_similarity(
            expected_probs[None],
            actual_probs[None],
        )
        assert float(cosine) > 0.999
        assert int(expected_probs.argmax()) == int(actual_probs.argmax())
        return

    if case in {"nafnet_restore", "realesrgan_restore"}:
        expected_rgb = expected.restored.array
        actual_rgb = actual.restored.array
        assert actual_rgb.shape == expected_rgb.shape
        assert _psnr(expected_rgb, actual_rgb, 255.0) > 40.0
        return

    if case in {
        "depth_anything_depth",
        "depth_anything3_depth",
        "zipdepth_depth",
    }:
        expected_depth = expected.depth_map.data.detach().float().cpu().numpy()
        actual_depth = actual.depth_map.data.detach().float().cpu().numpy()
        assert actual_depth.shape == expected_depth.shape
        peak = max(float(np.max(np.abs(expected_depth))), 1e-6)
        assert _psnr(expected_depth, actual_depth, peak) > 40.0
        return

    if case == "segformer_semantic":
        expected_mask = expected.semantic_mask.data.detach().cpu().numpy()
        actual_mask = actual.semantic_mask.data.detach().cpu().numpy()
        assert actual_mask.shape == expected_mask.shape
        assert float(np.mean(actual_mask == expected_mask)) > 0.95
        return

    if case == "fomo_point":
        _assert_point_parity(expected, actual)
        return

    if case == "rfdetr_obb":
        expected_obb = expected.obb.data.detach().float().cpu().numpy()
        actual_obb = actual.obb.data.detach().float().cpu().numpy()
        assert len(expected_obb) == len(actual_obb)
        expected_classes = expected_obb[:, 6].astype(np.int64)
        actual_classes = actual_obb[:, 6].astype(np.int64)
        assert sorted(expected_classes.tolist()) == sorted(actual_classes.tolist())
        for class_id in np.unique(expected_classes):
            expected_rows = expected_obb[expected_classes == class_id]
            actual_rows = actual_obb[actual_classes == class_id]
            cost = np.max(
                np.abs(expected_rows[:, None, :2] - actual_rows[None, :, :2]),
                axis=2,
            )
            expected_order, actual_order = linear_sum_assignment(cost)
            np.testing.assert_allclose(
                actual_rows[actual_order, :4],
                expected_rows[expected_order, :4],
                rtol=2e-3,
                atol=2e-2,
            )
            for expected_row, actual_row in zip(
                expected_rows[expected_order],
                actual_rows[actual_order],
            ):
                square = abs(float(expected_row[2] - expected_row[3])) < 0.05
                period = np.pi / 2.0 if square else np.pi
                angle_error = (
                    float(actual_row[4] - expected_row[4]) + period / 2.0
                ) % period - period / 2.0
                assert abs(angle_error) < 0.02
            np.testing.assert_allclose(
                actual_rows[actual_order, 5],
                expected_rows[expected_order, 5],
                rtol=2e-3,
                atol=0.01,
            )
        return

    matches = _match_detection_rows(expected, actual)
    if case in {"ec_pose", "rfdetr_pose", "yolonas_pose"}:
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
        return

    if case in {"ec_segment", "rfdetr_segment"}:
        expected_masks = expected.masks.data.detach().float().cpu().numpy()
        actual_masks = actual.masks.data.detach().float().cpu().numpy()
        for expected_index, actual_index in matches:
            expected_mask = expected_masks[expected_index] > 0.5
            actual_mask = actual_masks[actual_index] > 0.5
            union = np.logical_or(expected_mask, actual_mask).sum()
            if union:
                intersection = np.logical_and(expected_mask, actual_mask).sum()
                assert float(intersection / union) > 0.95
        return

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
            [float(step % 2), 22.0 + step % 5, 28.0, 16.0, 20.0]
        )
        targets[1, 0] = torch.tensor(
            [float((step + 1) % 2), 42.0, 36.0, 14.0, 18.0]
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
            head.cls_pred.weight.mul_(100.0)
            head.cls_pred.bias.zero_()
    return model


def _strengthen_yolo9_p2_fixture(model) -> None:
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


def _strengthen_yolonas_pose_fixture(model) -> None:
    with torch.no_grad():
        for head in (
            model.model.heads.head1,
            model.model.heads.head2,
            model.model.heads.head3,
        ):
            head.reg_pred.weight.mul_(10.0)
            head.cls_pred.weight.mul_(100.0)
            head.cls_pred.bias.zero_()
            head.pose_pred.weight.mul_(10.0)
            head.pose_pred.bias.zero_()


def _strengthen_rfdetr_obb_fixture(model) -> None:
    """Make the zero-initialized angle head input-sensitive for conversion parity."""
    with torch.no_grad():
        angle_head = model.model.model.angle_embed.layers[-1]
        torch.nn.init.uniform_(angle_head.weight, -0.02, 0.02)
        torch.nn.init.uniform_(angle_head.bias, -0.02, 0.02)


@pytest.mark.parametrize(
    ("family", "imgsz"),
    [("yolo9", 64), ("rfdetr", 384)],
)
def test_detection_flagship_raw_parity_and_predict(
    tmp_path, monkeypatch, family, imgsz
):
    _require_executorch(monkeypatch)

    from libreyolo import LibreRFDETR, LibreYOLO, LibreYOLO9
    from libreyolo.export.exporter import ExecuTorchExporter
    from libreyolo.utils.results import Results

    torch.manual_seed(0)
    if family == "yolo9":
        model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    else:
        pytest.importorskip("transformers")
        model = LibreRFDETR(
            {}, size="n", nb_classes=2, device="cpu", task="detect"
        )
    model.model.eval()
    original_training = model.model.training
    original_export = getattr(
        getattr(model.model, "head", None), "export", None
    )

    rng = np.random.default_rng(0)
    first = torch.from_numpy(
        rng.standard_normal((1, 3, imgsz, imgsz), dtype=np.float32)
    )
    second = (
        torch.full_like(first, 100.0)
        if family == "yolo9"
        else torch.from_numpy(
            np.random.default_rng(1).standard_normal(
                (1, 3, imgsz, imgsz), dtype=np.float32
            )
        )
    )

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (imgsz, imgsz)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    assert model.model.training is original_training
    if original_export is not None:
        assert model.model.head.export is original_export

    backend = LibreYOLO(artifact)
    actual = backend._run_inference(first.numpy())
    changed = backend._run_inference(second.numpy())
    assert len(actual) == len(expected)

    parity_error = 0.0
    for actual_output, expected_output in zip(actual, expected):
        expected_array = expected_output.detach().cpu().numpy()
        np.testing.assert_allclose(
            actual_output, expected_array, rtol=1e-3, atol=2e-4
        )
        parity_error = max(
            parity_error,
            float(np.max(np.abs(actual_output - expected_array))),
        )

    sensitivity = max(
        float(np.max(np.abs(first_output - second_output)))
        for first_output, second_output in zip(actual, changed)
    )
    assert sensitivity > max(parity_error * 100, 1e-4)

    image = np.random.default_rng(2).integers(
        0, 256, (imgsz, imgsz, 3), dtype=np.uint8
    )
    result = backend.predict(image)
    assert isinstance(result, Results)
    assert result.boxes is not None


@pytest.mark.extended_backend
@pytest.mark.parametrize(
    ("family", "class_name", "size"),
    [
        ("teed", "LibreTEED", "t"),
        ("dexined", "LibreDexiNed", "b"),
    ],
)
def test_edge_map_runtime_parity(
    tmp_path, monkeypatch, family, class_name, size
):
    """Prove edge-map conversion and parity without restricted checkpoints."""
    _require_executorch(monkeypatch)

    import libreyolo
    from libreyolo import LibreYOLO

    torch.manual_seed(7)
    model_class = getattr(libreyolo, class_name)
    model = model_class(None, size=size, device="cpu")
    first = np.random.default_rng(7).integers(
        0, 256, (40, 64, 3), dtype=np.uint8
    )
    second = np.random.default_rng(8).integers(
        0, 256, (40, 64, 3), dtype=np.uint8
    )
    expected = model.predict(first, imgsz=64).edges.data.numpy()
    expected_changed = model.predict(second, imgsz=64).edges.data.numpy()

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=64,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime.predict(first).edges.data.numpy()
    changed = runtime.predict(second).edges.data.numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-4)
    np.testing.assert_allclose(
        changed,
        expected_changed,
        rtol=1e-4,
        atol=2e-4,
    )
    parity_error = max(
        float(np.max(np.abs(actual - expected))),
        float(np.max(np.abs(changed - expected_changed))),
    )
    sensitivity = float(np.max(np.abs(expected - expected_changed)))
    assert sensitivity > max(parity_error * 100.0, 1e-4)
    assert runtime.model_family == family
    assert runtime.task == "edge"
    assert runtime.imgsz == 64


@pytest.mark.extended_backend
@pytest.mark.network
def test_dinov2_semantic_runtime_parity(tmp_path, monkeypatch):
    """Cover the real DINOv2 backbone and dense semantic output contract."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import ExecuTorchExporter
    from libreyolo.models.dinov2.model import LibreDINOv2

    torch.manual_seed(43)
    model = LibreDINOv2(
        None, size="n", task="semantic", nb_classes=3, device="cpu"
    )
    first = torch.from_numpy(
        np.random.default_rng(43).standard_normal(
            (1, 3, 518, 518), dtype=np.float32
        )
    )
    second = torch.zeros_like(first)

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (518, 518)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / "dinov2_semantic.pte"),
        imgsz=518,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())
    changed = runtime._run_inference(second.numpy())

    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(
            actual_output,
            expected_output.detach().cpu().numpy(),
            rtol=1e-3,
            atol=2e-4,
        )
    assert max(
        float(np.max(np.abs(a - b))) for a, b in zip(actual, changed)
    ) > 1e-4

    image = np.random.default_rng(44).integers(
        0, 256, (518, 518, 3), dtype=np.uint8
    )
    expected_result = _scalar_result(model.predict(image, imgsz=518))
    actual_result = _scalar_result(runtime.predict(image))
    expected_mask = expected_result.semantic_mask.data.detach().cpu().numpy()
    actual_mask = actual_result.semantic_mask.data.detach().cpu().numpy()
    assert actual_mask.shape == expected_mask.shape
    assert float(np.mean(actual_mask == expected_mask)) > 0.95


@pytest.mark.extended_backend
@pytest.mark.network
def test_dinov2_classification_runtime_parity(tmp_path, monkeypatch):
    """Cover the real DINOv2 backbone and classification logits contract."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import ExecuTorchExporter
    from libreyolo.models.dinov2.model import LibreDINOv2

    torch.manual_seed(51)
    model = LibreDINOv2(
        None, size="n", task="classify", nb_classes=5, device="cpu"
    )
    first = torch.from_numpy(
        np.random.default_rng(51).standard_normal(
            (1, 3, 224, 224), dtype=np.float32
        )
    )
    second = torch.zeros_like(first)

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (224, 224)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / "dinov2_classify.pte"),
        imgsz=224,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())
    changed = runtime._run_inference(second.numpy())

    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(
            actual_output,
            expected_output.detach().cpu().numpy(),
            rtol=1e-3,
            atol=2e-4,
        )
    assert max(
        float(np.max(np.abs(a - b))) for a, b in zip(actual, changed)
    ) > 1e-4
    assert runtime.predict(np.zeros((224, 224, 3), dtype=np.uint8)).probs is not None


@pytest.mark.extended_backend
@pytest.mark.network
def test_dinov2_embedding_runtime_parity(tmp_path, monkeypatch):
    """Cover the real DINOv2 backbone and normalized embedding contract."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO
    from libreyolo.export.exporter import ExecuTorchExporter
    from libreyolo.models.dinov2.model import LibreDINOv2

    model = LibreDINOv2(None, size="n", task="embed", device="cpu")
    first = torch.from_numpy(
        np.random.default_rng(52).standard_normal(
            (1, 3, 224, 224), dtype=np.float32
        )
    )
    second = torch.zeros_like(first)

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (224, 224)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / "dinov2_embed.pte"),
        imgsz=224,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())[0]
    changed = runtime._run_inference(second.numpy())[0]

    np.testing.assert_allclose(
        actual,
        expected.detach().cpu().numpy(),
        rtol=1e-3,
        atol=2e-4,
    )
    assert float(np.max(np.abs(actual - changed))) > 1e-4
    result = runtime.predict(np.zeros((224, 224, 3), dtype=np.uint8))
    assert result.embeddings is not None
    np.testing.assert_allclose(
        result.embeddings.data.norm(dim=1).numpy(),
        np.ones(1, dtype=np.float32),
        atol=1e-5,
    )


@pytest.mark.extended_backend
@pytest.mark.parametrize("family", ["rtmdet", "yolonas", "yolo9_p2"])
def test_additional_detection_raw_parity(tmp_path, monkeypatch, family):
    """Cover detector families lacking redistributable trained parity data."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreRTMDet, LibreYOLO, LibreYOLO9P2
    from libreyolo.export.exporter import ExecuTorchExporter

    torch.manual_seed(11)
    if family == "rtmdet":
        model = LibreRTMDet(None, size="t", nb_classes=2, device="cpu")
    elif family == "yolonas":
        model = _build_synthetic_yolonas_detect(64)
    else:
        model = LibreYOLO9P2(
            None, size="t", nb_classes=2, device="cpu"
        )
        _strengthen_yolo9_p2_fixture(model)

    first = torch.from_numpy(
        np.random.default_rng(11).standard_normal(
            (1, 3, 64, 64), dtype=np.float32
        )
    )
    second = (
        torch.full_like(first, 100.0)
        if family in {"rtmdet", "yolo9_p2"}
        else torch.from_numpy(
            np.random.default_rng(12).standard_normal(
                (1, 3, 64, 64), dtype=np.float32
            )
        )
    )
    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (64, 64)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
        expected_changed = prepared(second)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
        expected_changed = (expected_changed,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=64,
        batch=1,
        dynamic=False,
    )
    if family == "yolo9_p2":
        assert model.model.head.export is False
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())
    changed = runtime._run_inference(second.numpy())

    assert len(expected) == len(actual)
    for expected_output, expected_changed_output, actual_output, changed_output in zip(
        expected,
        expected_changed,
        actual,
        changed,
    ):
        expected_array = expected_output.detach().cpu().numpy()
        expected_changed_array = expected_changed_output.detach().cpu().numpy()
        np.testing.assert_allclose(
            actual_output, expected_array, rtol=1e-3, atol=2e-4
        )
        np.testing.assert_allclose(
            changed_output,
            expected_changed_array,
            rtol=1e-3,
            atol=2e-4,
        )
        parity_error = max(
            float(np.max(np.abs(actual_output - expected_array))),
            float(np.max(np.abs(changed_output - expected_changed_array))),
        )
        sensitivity = float(
            np.max(np.abs(expected_array - expected_changed_array))
        )
        sensitivity_floor = 1e-8 if family == "rtmdet" else 1e-4
        assert sensitivity > max(parity_error * 100.0, sensitivity_floor)

    image = np.random.default_rng(13).integers(
        0, 256, (64, 64, 3), dtype=np.uint8
    )
    confidence = 0.1 if family == "yolo9_p2" else 0.0
    actual_result = _scalar_result(
        runtime.predict(image, conf=confidence, max_det=20)
    )
    assert actual_result.boxes is not None
    if family != "rtmdet":
        expected_result = _scalar_result(
            model.predict(image, imgsz=64, conf=confidence, max_det=20)
        )
        _match_detection_rows(expected_result, actual_result)
    assert runtime.model_family == family
    assert runtime.task == "detect"
    assert runtime.imgsz == 64


@pytest.mark.extended_backend
@pytest.mark.parametrize(
    ("case", "imgsz"),
    [
        ("convnext_classify", 64),
        ("dinov2_classify", 224),
        ("depth_anything3_depth", 56),
        ("nafnet_restore", 64),
        pytest.param(
            "depth_anything_depth",
            56,
            marks=pytest.mark.network,
        ),
        ("ec_pose", 64),
        ("ec_segment", 128),
        ("fomo_point", 64),
        ("l2cs_gaze", 448),
        ("realesrgan_restore", 32),
        pytest.param(
            "rfdetr_segment",
            312,
            marks=pytest.mark.external_data,
        ),
        pytest.param(
            "rfdetr_pose",
            576,
            marks=pytest.mark.external_data,
        ),
        ("rfdetr_obb", 384),
        ("zipdepth_depth", 64),
        ("segformer_semantic", 64),
        ("yolonas_pose", 64),
    ],
)
def test_additional_task_raw_parity(tmp_path, monkeypatch, case, imgsz):
    """Cover fixed-shape task graphs without redistributable trained parity data."""
    _require_executorch(monkeypatch)

    from libreyolo import (
        LibreConvNeXt,
        LibreDepthAnything3,
        LibreDINOv2,
        LibreEC,
        LibreFOMO,
        LibreL2CS,
        LibreNAFNet,
        LibreRealESRGAN,
        LibreRFDETR,
        LibreSegformer,
        LibreYOLO,
        LibreYOLONAS,
        LibreZipDepth,
    )
    from libreyolo.export.exporter import ExecuTorchExporter

    constructors = {
        "convnext_classify": lambda: LibreConvNeXt(
            None, size="t", nb_classes=3, device="cpu"
        ),
        "dinov2_classify": lambda: LibreDINOv2(
            None, size="n", task="classify", nb_classes=3, device="cpu"
        ),
        "depth_anything3_depth": lambda: LibreDepthAnything3(
            None, size="l", device="cpu"
        ),
        "nafnet_restore": lambda: LibreNAFNet(None, size="s", device="cpu"),
        "depth_anything_depth": lambda: LibreYOLO(
            "weights/LibreDepthAnythingV2s-depth.pt",
            device="cpu",
        ),
        "ec_pose": lambda: LibreEC(None, size="s", task="pose", device="cpu"),
        "ec_segment": lambda: LibreEC(
            None, size="s", task="segment", nb_classes=2, device="cpu"
        ),
        "fomo_point": lambda: LibreFOMO(
            None, size="s", nb_classes=2, device="cpu"
        ),
        "l2cs_gaze": lambda: LibreL2CS(None, size="r18", device="cpu"),
        "realesrgan_restore": lambda: LibreRealESRGAN(
            None, size="x4t", device="cpu"
        ),
        "rfdetr_segment": lambda: LibreYOLO(
            "weights/LibreRFDETRn-seg.pt",
            device="cpu",
        ),
        "rfdetr_pose": lambda: LibreYOLO(
            "weights/LibreRFDETRx-pose.pt",
            device="cpu",
        ),
        "rfdetr_obb": lambda: LibreRFDETR(
            {},
            size="n",
            task="obb",
            nb_classes=2,
            device="cpu",
        ),
        "zipdepth_depth": lambda: LibreZipDepth(
            None, size="b", device="cpu"
        ),
        "segformer_semantic": lambda: LibreSegformer(
            None, size="b0", nb_classes=3, device="cpu"
        ),
        "yolonas_pose": lambda: LibreYOLONAS(
            None, size="n", task="pose", device="cpu"
        ),
    }
    torch.manual_seed(21)
    model = constructors[case]()
    if case == "yolonas_pose":
        _strengthen_yolonas_pose_fixture(model)
    elif case == "rfdetr_obb":
        _strengthen_rfdetr_obb_fixture(model)
    model.model.eval()
    first = torch.rand(
        1,
        3,
        imgsz,
        imgsz,
        generator=torch.Generator().manual_seed(21),
    )
    second = torch.rand(
        1,
        3,
        imgsz,
        imgsz,
        generator=torch.Generator().manual_seed(22),
    )

    exporter = ExecuTorchExporter(model)
    with exporter._model_context(
        torch.device("cpu"), False, False, 1, (imgsz, imgsz)
    ) as (prepared, _), torch.no_grad():
        expected = prepared(first)
        expected_changed = prepared(second)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
        expected_changed = (expected_changed,)

    artifact = model.export(
        "executorch",
        output_path=str(tmp_path / f"{case}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)
    actual = runtime._run_inference(first.numpy())
    changed = runtime._run_inference(second.numpy())

    assert len(expected) == len(actual)
    expected_arrays = [output.detach().cpu().numpy() for output in expected]
    expected_changed_arrays = [
        output.detach().cpu().numpy() for output in expected_changed
    ]
    if case in {
        "ec_segment",
        "rfdetr_obb",
        "rfdetr_pose",
        "rfdetr_segment",
    }:
        actual = _align_unordered_queries(expected_arrays, actual)
        changed = _align_unordered_queries(expected_changed_arrays, changed)
    raw_rtol = 5e-3 if case.startswith("rfdetr_") else 1e-3
    raw_atol = 2e-2 if case.startswith("rfdetr_") else 2e-4
    for (
        expected_array,
        expected_changed_array,
        actual_output,
        changed_output,
    ) in zip(
        expected_arrays,
        expected_changed_arrays,
        actual,
        changed,
    ):
        np.testing.assert_allclose(
            actual_output, expected_array, rtol=raw_rtol, atol=raw_atol
        )
        np.testing.assert_allclose(
            changed_output,
            expected_changed_array,
            rtol=raw_rtol,
            atol=raw_atol,
        )
        parity_error = max(
            float(np.max(np.abs(actual_output - expected_array))),
            float(np.max(np.abs(changed_output - expected_changed_array))),
        )
        sensitivity = float(
            np.max(np.abs(expected_array - expected_changed_array))
        )
        assert sensitivity > max(parity_error * 100.0, 1e-4)

    image = np.random.default_rng(22).integers(
        0, 256, (imgsz, imgsz, 3), dtype=np.uint8
    )
    _assert_public_task_parity(case, model, runtime, image, imgsz)
    assert runtime.model_family == model.FAMILY
    assert runtime.task == model.task
    assert runtime.imgsz == imgsz


@pytest.mark.external_data
@pytest.mark.parametrize(
    ("family", "weights_env", "imgsz"),
    [
        ("yolo9", "LIBREYOLO_EXECUTORCH_YOLO9_WEIGHTS", 640),
        ("rfdetr", "LIBREYOLO_EXECUTORCH_RFDETR_WEIGHTS", 384),
        ("yolox", "LIBREYOLO_EXECUTORCH_YOLOX_WEIGHTS", 416),
        ("picodet", "LIBREYOLO_EXECUTORCH_PICODET_WEIGHTS", 320),
        ("yolo9_e2e", "LIBREYOLO_EXECUTORCH_YOLO9_E2E_WEIGHTS", 640),
        ("ec", "LIBREYOLO_EXECUTORCH_EC_WEIGHTS", 640),
        ("rtdetr", "LIBREYOLO_EXECUTORCH_RTDETR_WEIGHTS", 640),
        ("rtdetrv2", "LIBREYOLO_EXECUTORCH_RTDETRV2_WEIGHTS", 640),
        ("rtdetrv4", "LIBREYOLO_EXECUTORCH_RTDETRV4_WEIGHTS", 640),
        ("yolo1", "LIBREYOLO_EXECUTORCH_YOLO1_WEIGHTS", 448),
        ("yolo2", "LIBREYOLO_EXECUTORCH_YOLO2_WEIGHTS", 416),
        ("yolo3", "LIBREYOLO_EXECUTORCH_YOLO3_WEIGHTS", 416),
        ("yolo4", "LIBREYOLO_EXECUTORCH_YOLO4_WEIGHTS", 416),
        ("yolo7", "LIBREYOLO_EXECUTORCH_YOLO7_WEIGHTS", 640),
    ],
)
def test_trained_detection_parity(
    tmp_path, monkeypatch, family, weights_env, imgsz
):
    """Match trained native and ExecuTorch post-NMS detections on real images."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to a newline-separated list of at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        expected = _detections(
            native.predict(str(image), conf=0.25, iou=0.6)
        )
        actual = _detections(
            runtime.predict(str(image), conf=0.25, iou=0.6)
        )
        assert len(expected) > 0
        assert len(actual) == len(expected)

        remaining = set(range(len(actual)))
        for expected_row in expected:
            same_class = [
                index
                for index in remaining
                if int(actual[index, 5]) == int(expected_row[5])
            ]
            assert same_class
            match = max(
                same_class,
                key=lambda index: _box_iou(expected_row, actual[index]),
            )
            remaining.remove(match)
            assert _box_iou(expected_row, actual[match]) >= 0.95
            assert abs(float(expected_row[4] - actual[match, 4])) <= 0.01


@pytest.mark.external_data
@pytest.mark.parametrize(
    ("family", "weights_env"),
    [
        ("mobilenetv4", "LIBREYOLO_EXECUTORCH_MOBILENETV4_WEIGHTS"),
        ("efficientnetv2", "LIBREYOLO_EXECUTORCH_EFFICIENTNETV2_WEIGHTS"),
        ("resnet", "LIBREYOLO_EXECUTORCH_RESNET_WEIGHTS"),
        ("convnext", "LIBREYOLO_EXECUTORCH_CONVNEXT_WEIGHTS"),
    ],
)
def test_trained_classification_parity(
    tmp_path, monkeypatch, family, weights_env
):
    """Match trained native and ExecuTorch logits and top-1 predictions."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to a newline-separated list of at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=224,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        native_result = native.predict(str(image))
        runtime_result = runtime.predict(str(image))
        expected = native_result.probs.data.detach().cpu().numpy()
        actual = runtime_result.probs.data.detach().cpu().numpy()
        cosine = float(
            np.dot(expected, actual)
            / (np.linalg.norm(expected) * np.linalg.norm(actual))
        )
        assert cosine >= 0.999
        assert int(np.argmax(actual)) == int(np.argmax(expected))


@pytest.mark.external_data
@pytest.mark.parametrize(
    ("family", "weights_env", "imgsz"),
    [
        ("pidnet", "LIBREYOLO_EXECUTORCH_PIDNET_WEIGHTS", 1024),
        (
            "lingbotvision",
            "LIBREYOLO_EXECUTORCH_LINGBOTVISION_WEIGHTS",
            512,
        ),
    ],
)
def test_trained_semantic_parity(
    tmp_path, monkeypatch, family, weights_env, imgsz
):
    """Match trained semantic maps after public postprocessing."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to at least two images"
        )

    weights = Path(weights_value)
    images = [Path(value) for value in image_values if value.strip()]
    if not weights.is_file() or any(not image.is_file() for image in images):
        pytest.skip("staged trained-checkpoint parity inputs are unavailable")

    native = LibreYOLO(str(weights), device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image in images:
        expected = (
            native.predict(str(image)).semantic_mask.data.detach().cpu().numpy()
        )
        actual = (
            runtime.predict(str(image)).semantic_mask.data.detach().cpu().numpy()
        )
        assert expected.shape == actual.shape
        assert float(np.mean(expected == actual)) >= 0.95


@pytest.mark.external_data
@pytest.mark.parametrize(
    ("family", "weights_env", "imgsz"),
    [
        (
            "depth_anything",
            "LIBREYOLO_EXECUTORCH_DEPTH_ANYTHING_WEIGHTS",
            518,
        ),
        ("zipdepth", "LIBREYOLO_EXECUTORCH_ZIPDEPTH_WEIGHTS", 384),
    ],
)
def test_trained_depth_parity(
    tmp_path, monkeypatch, family, weights_env, imgsz
):
    """Match a trained depth map on its fixed export canvas."""
    _require_executorch(monkeypatch)

    from PIL import Image

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to at least two images"
        )

    native = LibreYOLO(weights_value, device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image_path in image_values:
        image = np.asarray(
            Image.open(image_path).convert("RGB").resize((imgsz, imgsz))
        )
        expected = native.predict(image, imgsz=imgsz).depth_map.data.numpy()
        actual = runtime.predict(image).depth_map.data.numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.external_data
def test_trained_moge2_normal_parity(tmp_path, monkeypatch):
    """Match trained MoGe-2 surface normals on its fixed export canvas."""
    _require_executorch(monkeypatch)

    from PIL import Image

    from libreyolo import LibreYOLO

    weights_value = os.environ.get("LIBREYOLO_EXECUTORCH_MOGE2_WEIGHTS")
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            "set LIBREYOLO_EXECUTORCH_MOGE2_WEIGHTS and "
            "LIBREYOLO_EXECUTORCH_IMAGES to at least two images"
        )

    native = LibreYOLO(weights_value, device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / "moge2.pte"),
        imgsz=518,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image_path in image_values:
        image = np.asarray(
            Image.open(image_path).convert("RGB").resize((518, 518))
        )
        expected = native.predict(image, imgsz=518).normal_map.data.numpy()
        actual = runtime.predict(image).normal_map.data.numpy()
        cosine = np.clip(np.sum(actual * expected, axis=-1), -1.0, 1.0)
        angular_error = np.degrees(np.arccos(cosine))
        assert float(np.mean(angular_error)) < 0.01
        assert float(np.max(angular_error)) < 0.1


@pytest.mark.external_data
def test_trained_realesrgan_parity(tmp_path, monkeypatch):
    """Match trained x4 restoration and enforce the fixed-canvas contract."""
    _require_executorch(monkeypatch)

    from PIL import Image

    from libreyolo import LibreYOLO

    weights_value = os.environ.get("LIBREYOLO_EXECUTORCH_REALESRGAN_WEIGHTS")
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            "set LIBREYOLO_EXECUTORCH_REALESRGAN_WEIGHTS and "
            "LIBREYOLO_EXECUTORCH_IMAGES to at least two images"
        )

    native = LibreYOLO(weights_value, device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / "realesrgan.pte"),
        imgsz=64,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image_path in image_values:
        image = np.asarray(
            Image.open(image_path).convert("RGB").resize((64, 64))
        )
        expected = native.predict(image, imgsz=64).restored.data.numpy()
        actual = runtime.predict(image).restored.data.numpy()
        assert actual.shape == expected.shape == (256, 256, 3)
        assert int(
            np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))
        ) <= 1


@pytest.mark.external_data
@pytest.mark.parametrize(
    ("family", "task", "weights_env", "imgsz"),
    [
        ("ec", "segment", "LIBREYOLO_EXECUTORCH_EC_SEG_WEIGHTS", 640),
        ("ec", "pose", "LIBREYOLO_EXECUTORCH_EC_POSE_WEIGHTS", 640),
        (
            "rfdetr",
            "segment",
            "LIBREYOLO_EXECUTORCH_RFDETR_SEG_WEIGHTS",
            312,
        ),
        ("rfdetr", "pose", "LIBREYOLO_EXECUTORCH_RFDETR_POSE_WEIGHTS", 576),
    ],
)
def test_trained_instance_task_parity(
    tmp_path, monkeypatch, family, task, weights_env, imgsz
):
    """Match trained boxes plus pose keypoints or instance masks."""
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO

    weights_value = os.environ.get(weights_env)
    image_values = os.environ.get("LIBREYOLO_EXECUTORCH_IMAGES", "").splitlines()
    if not weights_value or len(image_values) < 2:
        pytest.skip(
            f"set {weights_env} and LIBREYOLO_EXECUTORCH_IMAGES "
            "to at least two images"
        )

    native = LibreYOLO(weights_value, device="cpu")
    artifact = native.export(
        "executorch",
        output_path=str(tmp_path / f"{family}_{task}.pte"),
        imgsz=imgsz,
        batch=1,
        dynamic=False,
    )
    runtime = LibreYOLO(artifact)

    for image_path in image_values:
        from PIL import Image

        expected = native.predict(
            image_path, imgsz=imgsz, conf=0.25, iou=0.6, max_det=100
        )
        actual = runtime.predict(
            image_path, conf=0.25, iou=0.6, max_det=100
        )
        width, height = Image.open(image_path).size
        expected_boxes = expected.boxes.data.numpy().copy()
        expected_boxes[:, [0, 2]] = np.clip(
            expected_boxes[:, [0, 2]], 0, width
        )
        expected_boxes[:, [1, 3]] = np.clip(
            expected_boxes[:, [1, 3]], 0, height
        )
        np.testing.assert_allclose(
            actual.boxes.data.numpy(),
            expected_boxes,
            rtol=1e-3,
            atol=2e-3,
        )
        if task == "segment":
            assert float(
                (actual.masks.data == expected.masks.data).float().mean()
            ) >= 0.999
        else:
            np.testing.assert_allclose(
                actual.keypoints.data.numpy(),
                expected.keypoints.data.numpy(),
                rtol=1e-3,
                atol=2e-3,
            )


def test_failed_export_restores_yolo9_state(tmp_path, monkeypatch):
    _require_executorch(monkeypatch)

    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    model.model.train()
    original_training = model.model.training
    original_export = model.model.head.export

    def fail_export(*args, **kwargs):
        raise RuntimeError("simulated lowering failure")

    monkeypatch.setattr(
        "libreyolo.export.executorch.export_executorch", fail_export
    )
    output = tmp_path / "failed.pte"
    with pytest.raises(RuntimeError, match="simulated"):
        model.export(
            "executorch",
            output_path=str(output),
            imgsz=64,
            batch=1,
            dynamic=False,
        )

    assert model.model.training is original_training
    assert model.model.head.export is original_export
    assert not output.exists()
    assert not Path(f"{output}.json").exists()
