"""LiteRT FP32 measurements for ten previously blocked export cells.

The nine published checkpoints are permissively licensed and loaded through
LibreYOLO's normal factory.  YOLO9-P2 uses the pinned MIT transfer fixture and
YOLO-NAS pose uses deterministic synthetic training.  Each case must export,
reload, preserve two input-sensitive raw probes, and match public ``predict()``.
"""

from __future__ import annotations

import gc
import importlib.util
from dataclasses import dataclass

import pytest
import torch

from .test_tensorrt_round11 import (
    _align_outputs,
    _build_synthetic_yolonas_pose,
    _build_yolo9_p2_transfer,
    _rms,
)
from .test_tensorrt_round12 import _assert_predict_parity

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("onnx2tf") is None
        or importlib.util.find_spec("ai_edge_litert") is None,
        reason="onnx2tf and ai-edge-litert are required",
    ),
]


@dataclass(frozen=True)
class TFLiteRound13Case:
    family: str
    task: str
    imgsz: int
    weights: str | None = None
    fixture: str | None = None


_CASES = (
    TFLiteRound13Case("yolo1", "detect", 448, weights="LibreYOLO1b.pt"),
    TFLiteRound13Case(
        "yolo9_e2e", "detect", 128, weights="LibreYOLO9E2Et.pt"
    ),
    TFLiteRound13Case("yolo9_p2", "detect", 128, fixture="yolo9_transfer"),
    TFLiteRound13Case("rtmdet", "detect", 640, weights="LibreRTMDett.pt"),
    TFLiteRound13Case("picodet", "detect", 160, weights="LibrePICODETs.pt"),
    TFLiteRound13Case("dfine", "detect", 256, weights="LibreDFINEn.pt"),
    TFLiteRound13Case("ec", "detect", 256, weights="LibreECs.pt"),
    TFLiteRound13Case("rtdetr", "detect", 256, weights="LibreRTDETRr18.pt"),
    TFLiteRound13Case("yolonas", "pose", 96, fixture="synthetic_pose"),
    TFLiteRound13Case("dfine", "segment", 256, weights="LibreDFINEn-seg.pt"),
)

_HOLD_REASONS = {
    "yolo1-detect": "LiteRT cannot prepare ONNX_EINSUM.",
    "yolo9_p2-detect": "Public top-k class membership changes.",
    "rtmdet-detect": "Native-canvas public boxes fall to 0.911 IoU.",
    "picodet-detect": "LiteRT rejects a 19,200-to-9,600 RESHAPE.",
    "dfine-detect": "onnx2tf crashes while lowering GatherElements.",
    "ec-detect": "LiteRT cannot prepare ONNX_LAYERNORMALIZATION.",
    "rtdetr-detect": "LiteRT rejects incompatible CONCATENATION dimensions.",
    "yolonas-pose": "LiteRT rejects an invalid CONCATENATION input type.",
    "dfine-segment": "onnx2tf crashes while lowering GatherElements.",
}

# Cases without a hold reason pass on the current toolchain and run as
# ordinary tests (YOLO9-E2E since 2026-11-01).
ROUND13_CASES = tuple(
    pytest.param(
        case,
        marks=pytest.mark.xfail(
            strict=True,
            reason=_HOLD_REASONS[f"{case.family}-{case.task}"],
        )
        if f"{case.family}-{case.task}" in _HOLD_REASONS
        else (),
    )
    for case in _CASES
)


def _build_model(case: TFLiteRound13Case):
    if case.weights:
        from libreyolo import LibreYOLO

        return LibreYOLO(case.weights, device="cuda")
    if case.fixture == "yolo9_transfer":
        return _build_yolo9_p2_transfer()
    if case.fixture == "synthetic_pose":
        return _build_synthetic_yolonas_pose(case.imgsz)
    raise AssertionError(f"Round 13 case has no fixture: {case}")


def _native_outputs(model, first, second, imgsz):
    from libreyolo.export.exporter import TFLiteExporter

    def tensor_outputs(output):
        if isinstance(output, torch.Tensor):
            return [output]
        if isinstance(output, dict):
            values = output.values()
        elif isinstance(output, (tuple, list)):
            values = output
        else:
            return []
        tensors = []
        for value in values:
            tensors.extend(tensor_outputs(value))
        return tensors

    device = next(model.model.parameters()).device
    with TFLiteExporter(model)._model_context(
        device,
        False,
        False,
        1,
        (imgsz, imgsz),
    ) as (wrapped, _), torch.inference_mode():
        first_output = wrapped(first)
        second_output = wrapped(second)
    return (
        [
            value.detach().float().cpu().numpy()
            for value in tensor_outputs(first_output)
        ],
        [
            value.detach().float().cpu().numpy()
            for value in tensor_outputs(second_output)
        ],
    )


def _run_tflite_case(tmp_path, monkeypatch, case: TFLiteRound13Case) -> None:
    from libreyolo import LibreYOLO
    from libreyolo.export.support import SUPPORT, SupportEntry

    torch.manual_seed(13)
    model = _build_model(case)
    model.model.eval()
    assert model.FAMILY == case.family
    assert model.task == case.task

    device = next(model.model.parameters()).device
    first = torch.zeros(1, 3, case.imgsz, case.imgsz, device=device)
    second = torch.rand(1, 3, case.imgsz, case.imgsz, device=device)
    expected_first, expected_second = _native_outputs(
        model,
        first,
        second,
        case.imgsz,
    )

    key = (case.family, case.task, "tflite")
    monkeypatch.setitem(
        SUPPORT,
        key,
        SupportEntry(
            "available",
            "Round 13 measured probe bypasses the recorded conversion hold.",
        ),
    )
    artifact = model.export(
        format="tflite",
        output_path=str(tmp_path / f"{case.family}-{case.task}.tflite"),
        imgsz=case.imgsz,
        dynamic=False,
        half=False,
        simplify=False,
    )
    backend = LibreYOLO(artifact, device="cpu")

    actual_first = _align_outputs(
        expected_first,
        backend._run_inference(first.detach().cpu().numpy()),
    )
    actual_second = _align_outputs(
        expected_second,
        backend._run_inference(second.detach().cpu().numpy()),
    )

    assert backend.model_family == case.family
    assert backend.task == case.task
    assert backend.imgsz == case.imgsz

    for expected_a, expected_b, actual_a, actual_b in zip(
        expected_first,
        expected_second,
        actual_first,
        actual_second,
    ):
        parity_error = _rms(expected_a - actual_a)
        reference_scale = max(_rms(expected_a), 1e-6)
        expected_signal = _rms(expected_a - expected_b)
        actual_signal = _rms(actual_a - actual_b)
        relative_error = parity_error / reference_scale
        relative_signal = expected_signal / reference_scale
        signal_to_error = actual_signal / max(parity_error, 1e-12)
        assert relative_error < 1e-3, f"relative raw error {relative_error:.6g}"
        assert relative_signal > 1e-6, (
            f"relative input signal {relative_signal:.6g}"
        )
        assert signal_to_error > 20.0, (
            f"runtime signal/error ratio {signal_to_error:.6g}"
        )

    _assert_predict_parity(case, model, backend)

    del backend, model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "case",
    ROUND13_CASES,
    ids=lambda case: f"{case.family}-{case.task}",
)
def test_tflite_round13_measured_blocks(tmp_path, monkeypatch, case):
    _run_tflite_case(tmp_path, monkeypatch, case)
