"""Unit tests for the inference profiler (no GPU, no weights download).

The harness is exercised with a tiny stub model that implements the three stage
methods (``_preprocess``/``_forward``/``_postprocess``); the analysis reuse is
exercised against a synthetic inference Chrome-trace. Both run in CI on CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
import typer
from typer.testing import CliRunner

from libreyolo.profiling import InferenceProfiler
from libreyolo.training.profiler import analyze_trace

pytestmark = pytest.mark.unit
runner = CliRunner()


class _StubModel:
    """Minimal BaseModel-shaped stub: real torch ops, controllable stage cost."""

    def __init__(self, *, device="cpu", nms_sleep=0.0, pre_sleep=0.0):
        self.device = torch.device(device)
        self.model = torch.nn.Identity()  # provides .eval()
        self._nms_sleep = nms_sleep
        self._pre_sleep = pre_sleep

    def _get_input_size(self):
        return 32

    def _preprocess(self, image, color_format="auto", input_size=32):
        if self._pre_sleep:
            time.sleep(self._pre_sleep)
        return torch.zeros(1, 3, input_size, input_size), None, (input_size, input_size), 1.0

    def _forward(self, x):
        return torch.zeros(x.shape[0], 6, 8)

    def _postprocess(self, output, conf, iou, original_size, max_det=300, ratio=1.0, **kwargs):
        if self._nms_sleep:
            time.sleep(self._nms_sleep)
        return {"boxes": torch.zeros(0, 4)}


def test_latency_stages_and_schema(tmp_path):
    prof = InferenceProfiler(
        _StubModel(), warmup=1, runs=6, batch=1, imgsz=32,
        trace=False, save_dir=tmp_path, meta={"model": "stub"},
    )
    a = prof.run(["img"], conf=0.25, iou=0.45, max_det=100)

    assert a["mode"] == "inference"
    assert a["schema"] == "libreyolo.profile.analysis/v1"
    assert a["latency"]["n"] >= 1
    assert a["latency"]["p50_ms"] is not None
    assert set(a["stages_ms"]) == {"preprocess", "forward", "postprocess"}
    assert a["img_per_s"] > 0
    # inference metrics are `get`-able and self-consistent
    m = a["metrics"]
    assert m["throughput_img_s"] == a["img_per_s"]
    assert m["nms_ms"] == a["stages_ms"]["postprocess"]
    assert "latency_p50_ms" in m
    # self-contained artifact written
    assert (tmp_path / "profile.json").exists()
    assert json.loads((tmp_path / "profile.json").read_text())["mode"] == "inference"



class _RectStubModel(_StubModel):
    """A rectangular fine-tune (#899) reports its size as (height, width)."""

    def _get_input_size(self):
        return (16, 32)

    def _preprocess(self, image, color_format="auto", input_size=(16, 32)):
        height, width = input_size
        return torch.zeros(1, 3, height, width), None, (width, height), 1.0


def test_rectangular_native_size_is_profiled_without_imgsz(tmp_path):
    prof = InferenceProfiler(
        _RectStubModel(), warmup=0, runs=2, trace=False, save_dir=tmp_path
    )
    a = prof.run(["img"])

    assert a["latency"]["n"] >= 1
    assert tuple(prof.summary["meta"]["imgsz"]) == (16, 32)

def test_batch_throughput_scales(tmp_path):
    prof = InferenceProfiler(_StubModel(), warmup=1, runs=6, batch=4, imgsz=32,
                             trace=False, save_dir=tmp_path)
    a = prof.run(["img"])
    # img/s = batch / mean_latency; batch is recorded in meta
    assert a["config"]["batch"] == 4
    assert a["img_per_s"] > 0


def test_cuda_autocast_uses_configured_dtype(monkeypatch):
    calls = []

    class _Context:
        def __enter__(self):
            return None

        def __exit__(self, *_):
            return False

    def fake_autocast(device_type, *, dtype):
        calls.append((device_type, dtype))
        return _Context()

    prof = InferenceProfiler(
        _StubModel(),
        warmup=0,
        runs=2,
        imgsz=32,
        half=True,
        amp_dtype="bf16",
        trace=False,
    )
    prof._is_cuda = True
    monkeypatch.setattr(
        "libreyolo.profiling.inference.torch.autocast",
        fake_autocast,
    )

    prof._forward(torch.zeros(1, 3, 32, 32))

    assert calls == [("cuda", torch.bfloat16)]
    assert prof.amp_dtype == "bfloat16"


def test_verdict_nms_bound(tmp_path):
    prof = InferenceProfiler(_StubModel(nms_sleep=0.004), warmup=1, runs=6,
                             imgsz=32, trace=False, save_dir=tmp_path)
    a = prof.run(["img"])
    assert a["bound"] == "nms / postprocess"
    assert "max_det" in a["bound_why"]


def test_verdict_preprocess_bound(tmp_path):
    prof = InferenceProfiler(_StubModel(pre_sleep=0.004), warmup=1, runs=6,
                             imgsz=32, trace=False, save_dir=tmp_path)
    a = prof.run(["img"])
    assert a["bound"] == "preprocess"


def test_cpu_trace_reuses_analyze_trace(tmp_path):
    """With trace=True on CPU the profile flows through analyze_trace."""
    prof = InferenceProfiler(_StubModel(), warmup=2, runs=6, imgsz=32,
                             trace=True, save_dir=tmp_path)
    a = prof.run(["img"])
    assert a["mode"] == "inference"
    assert a["schema"] == "libreyolo.profile.analysis/v1"
    # analyze_trace populated the phase tables; inference stages are present
    phases = [p["phase"] for p in a["phases"]]
    assert any(p in phases for p in ("preprocess", "forward", "postprocess"))
    assert (tmp_path / "profile_trace.json").exists()


# --- shared analyzer handles inference traces (the reuse contract) ----------

def _infer_trace_events():
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{i}",
         "ts": i * 100, "dur": 100, "pid": 1, "tid": 1}
        for i in range(6)
    ]
    for base in (200, 300, 400):
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/preprocess",
                   "ts": base + 5, "dur": 20, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/forward",
                   "ts": base + 30, "dur": 40, "pid": 1, "tid": 1})
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step/postprocess",
                   "ts": base + 75, "dur": 20, "pid": 1, "tid": 1})
        for name, offset, dur in [
            ("elementwise_resize_kernel", 8, 6),     # preprocess
            ("ampere_sgemm_128x128_nt", 35, 25),     # forward
            ("nms_gather_kernel", 78, 8),            # postprocess
        ]:
            ev.append({"ph": "X", "cat": "kernel", "name": name,
                       "ts": base + offset, "dur": dur, "pid": 0, "tid": 7})
    return ev


def test_analyze_trace_inference_phase_order(tmp_path):
    trace = tmp_path / "profile_trace.json"
    trace.write_text(json.dumps({"traceEvents": _infer_trace_events()}))
    a = analyze_trace(trace)
    order = [p["phase"] for p in a["phases"]]
    assert "preprocess" in order and "forward" in order and "postprocess" in order
    assert order.index("preprocess") < order.index("forward") < order.index("postprocess")
    ph = {p["phase"]: p for p in a["phases"]}
    assert ph["postprocess"]["kernels_per_step"] == 1
    assert ph["forward"]["kernels_per_step"] == 1


# --- CLI ---------------------------------------------------------------------

def _app() -> typer.Typer:
    from libreyolo.cli.commands import profile

    app = typer.Typer(add_completion=False)
    app.add_typer(profile.profile_app, name="profile")
    return app


def test_cli_infer_missing_source(tmp_path):
    # source resolution happens before any model load, so this needs no weights
    r = runner.invoke(_app(), ["profile", "infer", str(tmp_path / "nope.jpg")])
    assert r.exit_code == 2
