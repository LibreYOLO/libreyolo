"""TensorRT FP32 parity for the Round 10 conventional detector families.

Each case exports a complete LibreYOLO architecture with deterministic
synthetic weights.  The suite verifies raw runtime parity on two inputs,
rejects disconnected or effectively constant graphs, reloads through the
public factory, and compares public ``predict()`` detections.  This validates
conversion and runtime behavior, not detector accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from .conftest import requires_tensorrt

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.tensorrt,
    pytest.mark.trt,
]


@dataclass(frozen=True)
class TensorRTRound10Case:
    class_name: str
    size: str
    imgsz: int
    nb_classes: int = 5


ROUND10_VALIDATED_CASES = (
    TensorRTRound10Case("LibreYOLO2", "t", 416),
    TensorRTRound10Case("LibreYOLO3", "t", 416),
    TensorRTRound10Case("LibreYOLO4", "t", 416),
)

ROUND10_AVAILABLE_CASES = (
    TensorRTRound10Case("LibreYOLO1", "t", 448, 20),
    TensorRTRound10Case("LibreYOLO7", "b", 128),
    TensorRTRound10Case("LibreYOLO9E2E", "t", 128),
    TensorRTRound10Case("LibreYOLO9P2", "t", 128),
    TensorRTRound10Case("LibreYOLOX", "n", 128),
    TensorRTRound10Case("LibrePICODET", "s", 160),
    TensorRTRound10Case("LibreRTMDet", "t", 128),
)


def _build_model(case: TensorRTRound10Case):
    import libreyolo

    model_cls = getattr(libreyolo, case.class_name)
    return model_cls(
        model_path=None,
        size=case.size,
        nb_classes=case.nb_classes,
        device="cuda",
    )


def _tensor_outputs(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict):
        values = output.values()
    elif isinstance(output, (tuple, list)):
        values = output
    else:
        raise TypeError(f"Unsupported export output type: {type(output)!r}")
    tensors = []
    for value in values:
        tensors.extend(_tensor_outputs(value))
    return tensors


def _raw_outputs(model, tensor: torch.Tensor, imgsz: int) -> list[np.ndarray]:
    from libreyolo.export.exporter import TensorRTExporter

    with TensorRTExporter(model)._model_context(
        torch.device("cuda"),
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (wrapped, _), torch.inference_mode():
        output = wrapped(tensor)
    return [value.detach().float().cpu().numpy() for value in _tensor_outputs(output)]


def _image(imgsz: int) -> np.ndarray:
    y, x = np.mgrid[:imgsz, :imgsz]
    image = np.empty((imgsz, imgsz, 3), dtype=np.uint8)
    image[..., 0] = (3 * x + y) % 256
    image[..., 1] = (x + 5 * y) % 256
    image[..., 2] = ((x // 7) * 31 + (y // 11) * 17) % 256
    return image


def _pairwise_box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0), axis=2)
    first_area = np.prod(np.maximum(first[:, 2:] - first[:, :2], 0.0), axis=1)
    second_area = np.prod(np.maximum(second[:, 2:] - second[:, :2], 0.0), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def _assert_predict_parity(native_model, backend, image, imgsz: int) -> None:
    expected = native_model.predict(image, imgsz=imgsz, conf=0.0, max_det=25)
    actual = backend.predict(image, conf=0.0, max_det=25)
    assert expected.boxes is not None
    assert actual.boxes is not None
    expected_data = expected.boxes.data.detach().float().cpu().numpy()
    actual_data = actual.boxes.data.detach().float().cpu().numpy()
    assert len(expected_data) == len(actual_data)
    if not len(expected_data):
        pytest.fail("The synthetic detector produced no predictions at conf=0")

    matched_ious = []
    coordinate_errors = []
    score_errors = []
    expected_classes = expected_data[:, 5].astype(np.int64)
    actual_classes = actual_data[:, 5].astype(np.int64)
    assert sorted(expected_classes.tolist()) == sorted(actual_classes.tolist())
    for class_id in np.unique(expected_classes):
        expected_rows = expected_data[expected_classes == class_id]
        actual_rows = actual_data[actual_classes == class_id]
        coordinate_cost = np.max(
            np.abs(expected_rows[:, None, :4] - actual_rows[None, :, :4]),
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(coordinate_cost)
        ious = _pairwise_box_iou(expected_rows[:, :4], actual_rows[:, :4])
        for row_index, column_index in zip(row_indices, column_indices):
            matched_ious.append(float(ious[row_index, column_index]))
            coordinate_errors.append(float(coordinate_cost[row_index, column_index]))
            score_errors.append(
                abs(
                    float(expected_rows[row_index, 4])
                    - float(actual_rows[column_index, 4])
                )
            )

    assert all(
        iou > 0.95 or coordinate_error < 1e-2
        for iou, coordinate_error in zip(matched_ious, coordinate_errors)
    )
    assert max(score_errors) < 0.01


def _strengthen_synthetic_head(model) -> None:
    """Make random detector heads image-sensitive without training data."""
    tokens_by_family = {
        "yolo7": ("head_conv",),
        "yolo9_e2e": ("head.one_to_one_anchor_convs", "head.one_to_one_class_convs"),
        "yolo9_p2": ("head.anchor_convs", "head.class_convs"),
        "yolox": ("head.cls_preds", "head.reg_preds", "head.obj_preds"),
        "picodet": ("head.gfl_cls",),
        "rtmdet": ("head.rtm_cls", "head.rtm_reg"),
    }
    tokens = tokens_by_family.get(model.FAMILY, ())
    if not tokens:
        return
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if not any(token in name for token in tokens):
                continue
            if name.endswith(".weight"):
                parameter.mul_(32.0)
            elif name.endswith(".bias"):
                parameter.zero_()


def _run_tensorrt_case(tmp_path, case: TensorRTRound10Case) -> None:
    from libreyolo import LibreYOLO

    torch.manual_seed(10)
    model = _build_model(case)
    model.model.eval()
    _strengthen_synthetic_head(model)

    first = torch.zeros(1, 3, case.imgsz, case.imgsz, device="cuda")
    second = torch.ones(1, 3, case.imgsz, case.imgsz, device="cuda")
    expected_first = _raw_outputs(model, first, case.imgsz)
    expected_second = _raw_outputs(model, second, case.imgsz)

    engine_path = tmp_path / f"{model.FAMILY}.engine"
    artifact = model.export(
        format="tensorrt",
        output_path=str(engine_path),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cuda")

    actual_first = backend._run_inference(first.cpu().numpy())
    actual_second = backend._run_inference(second.cpu().numpy())
    assert len(actual_first) == len(expected_first)
    assert len(actual_second) == len(expected_second)
    for expected, actual in zip(expected_first, actual_first):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    for expected, actual in zip(expected_second, actual_second):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)

    expected_signal = max(
        float(np.sqrt(np.mean((first_out - second_out).astype(np.float64) ** 2)))
        for first_out, second_out in zip(expected_first, expected_second)
    )
    actual_signal = max(
        float(np.sqrt(np.mean((first_out - second_out).astype(np.float64) ** 2)))
        for first_out, second_out in zip(actual_first, actual_second)
    )
    parity_error = max(
        float(np.sqrt(np.mean((expected - actual).astype(np.float64) ** 2)))
        for expected, actual in zip(expected_first, actual_first)
    )
    assert expected_signal > 1e-6
    assert actual_signal > max(1e-6, 100.0 * parity_error)

    assert backend.model_family == model.FAMILY
    assert backend.task == "detect"
    assert backend.imgsz == case.imgsz
    _assert_predict_parity(model, backend, _image(case.imgsz), case.imgsz)

    del backend, model
    torch.cuda.empty_cache()


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ROUND10_VALIDATED_CASES,
    ids=lambda case: case.class_name,
)
def test_tensorrt_round10_raw_and_predict_parity(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)


@requires_tensorrt
@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Round 10 measured this conversion, but its strict signal/raw/predict "
        "parity gate remains unresolved."
    ),
)
@pytest.mark.parametrize(
    "case",
    ROUND10_AVAILABLE_CASES,
    ids=lambda case: case.class_name,
)
def test_tensorrt_round10_measured_available(tmp_path, case):
    _run_tensorrt_case(tmp_path, case)
