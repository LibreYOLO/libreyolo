"""Unit tests for libreyolo.training.autobatch.

Tests are CPU-only and avoid triggering real CUDA initialisation.
The linear-fit core (_fit_batch_size) is tested directly; higher-level
functions are tested by patching autobatch() itself rather than mocking
CUDA internals.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

from libreyolo.training.autobatch import (
    _BATCH_SAFE_MAX,
    _fit_batch_size,
    _floor_pow2_strict,
    autobatch,
    resolve_auto_batch,
)


# =============================================================================
# _floor_pow2_strict
# =============================================================================


def test_floor_pow2_strict_basic():
    """Values above 2 return the largest power of 2 strictly less than x."""
    assert _floor_pow2_strict(33.0) == 32
    assert _floor_pow2_strict(65.0) == 64
    assert _floor_pow2_strict(17.5) == 16
    assert _floor_pow2_strict(5.0) == 4


def test_floor_pow2_strict_exact_power_goes_lower():
    """An exact power of 2 returns the next lower power (strictly less than)."""
    assert _floor_pow2_strict(32.0) == 16
    assert _floor_pow2_strict(64.0) == 32
    assert _floor_pow2_strict(4.0) == 2


def test_floor_pow2_strict_small_values():
    """Values ≤ 2 return 1."""
    assert _floor_pow2_strict(2.0) == 1
    assert _floor_pow2_strict(1.5) == 1
    assert _floor_pow2_strict(0.5) == 1


def test_floor_pow2_strict_result_divides_nbs():
    """For typical nbs values the result always divides nbs (power-of-2 property)."""
    for nbs in (32, 64, 128):
        for raw in (raw_val + 0.1 for raw_val in range(3, nbs * 2)):
            result = _floor_pow2_strict(raw)
            if result <= nbs:
                assert nbs % result == 0, f"raw={raw:.1f} → {result} does not divide nbs={nbs}"


# =============================================================================
# _fit_batch_size — pure math, no I/O
# =============================================================================


def test_fit_exact_linear():
    """Perfect linear data → extrapolation is exact."""
    # mem = 0.10 * batch + 0.50  GiB
    slope, intercept = 0.10, 0.50
    probes = [1, 2, 4, 8, 16]
    mems = [slope * b + intercept for b in probes]
    target = 4.80  # free_gib * fraction = 8.0 * 0.60

    result = _fit_batch_size(probes, mems, target)
    expected = round((target - intercept) / slope)  # 43
    assert result == expected


def test_fit_clamps_to_safe_max():
    """Extrapolated value above _BATCH_SAFE_MAX is clamped."""
    # Tiny slope → huge extrapolated batch
    probes = [1, 2, 4, 8, 16]
    mems = [0.0001 * b for b in probes]
    result = _fit_batch_size(probes, mems, target_gib=100.0)
    assert result == _BATCH_SAFE_MAX


def test_fit_minimum_one():
    """Extrapolated batch below 1 is clamped to 1."""
    # Target already exceeded at batch=1 → negative extrapolation → clamped to 1
    probes = [1, 2, 4]
    mems = [5.0, 6.0, 8.0]  # already above target
    result = _fit_batch_size(probes, mems, target_gib=1.0)
    assert result == 1


def test_fit_non_positive_slope_returns_none():
    """Flat or negative slope (degenerate data) returns None."""
    probes = [1, 2, 4]
    mems = [2.0, 2.0, 2.0]  # flat → slope = 0
    assert _fit_batch_size(probes, mems, target_gib=4.0) is None

    mems_dec = [3.0, 2.0, 1.0]  # decreasing → slope < 0
    assert _fit_batch_size(probes, mems_dec, target_gib=4.0) is None


# =============================================================================
# autobatch — CPU fallback
# =============================================================================


def test_autobatch_cpu_returns_default():
    """Non-CUDA device must return the default without probing."""
    model = nn.Linear(4, 2)
    result = autobatch(model, imgsz=32, amp=False, default=8)
    assert result == 8


def test_autobatch_cpu_ignores_fraction():
    """Fraction is irrelevant on CPU — default is always returned."""
    model = nn.Linear(4, 2)
    assert autobatch(model, imgsz=32, amp=False, fraction=0.99, default=24) == 24


# =============================================================================
# autobatch — CUDA paths via patching the probe internals
# =============================================================================


def _make_cuda_patches(probe_mem_fn, total_gib=8.0, model=None):
    """Return context managers that fake a CUDA environment inside autobatch.

    *probe_mem_fn(batch_size)* returns peak allocated memory in bytes for a
    given batch size probe call.  torch.zeros is patched to stay on CPU so no
    real CUDA initialisation occurs.  *model.forward* is mocked to a no-op so
    shape-mismatches in the probe model are avoided.
    """
    import contextlib

    class _FakeDevice:
        type = "cuda"
        index = 0

        def __str__(self):
            return "cuda:0"

    class _FakeParam:
        """Stand-in for a model parameter; carries a .device attribute."""
        device = _FakeDevice()

    class _FakeProps:
        name = "FakeGPU"
        total_memory = int(total_gib * 1024**3)

    peak = [0]
    _real_zeros = torch.zeros  # capture before patching to avoid recursion

    def _fake_zeros(*args, dtype=None, device=None):  # noqa: ARG001
        b = args[0]
        peak[0] = probe_mem_fn(b)
        return _real_zeros(*args, dtype=dtype or torch.float32)

    @contextlib.contextmanager
    def _ctx():
        base = [
            patch("libreyolo.training.autobatch.next", return_value=_FakeParam()),
            patch("torch.cuda.get_device_properties", return_value=_FakeProps()),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", side_effect=lambda *_: peak[0]),
            patch("torch.cuda.empty_cache"),
            patch("torch.zeros", side_effect=_fake_zeros),
        ]
        fwd = [patch.object(model, "forward", return_value=None)] if model is not None else []
        with contextlib.ExitStack() as stack:
            for p in base + fwd:
                stack.enter_context(p)
            yield

    return _ctx()


def test_autobatch_linear_extrapolates_correctly():
    """With linear memory growth the result is _floor_pow2_strict of the extrapolation."""
    slope_gib = 0.10
    intercept_gib = 0.50
    total_gib = 8.0
    fraction = 0.70

    def mem_fn(b):
        return int((slope_gib * b + intercept_gib) * 1024**3)

    model = nn.Linear(4, 2)
    with _make_cuda_patches(mem_fn, total_gib=total_gib, model=model):
        result = autobatch(model, imgsz=32, amp=False, fraction=fraction, default=16)

    raw = (total_gib * fraction - intercept_gib) / slope_gib
    expected = _floor_pow2_strict(raw)
    assert result == expected


def test_autobatch_clamps_to_safe_max():
    """Tiny memory-per-sample → extrapolation → clamped to _BATCH_SAFE_MAX."""
    def mem_fn(b):
        return int(b * 0.0001 * 1024**3)

    model = nn.Linear(4, 2)
    with _make_cuda_patches(mem_fn, total_gib=128.0, model=model):
        result = autobatch(model, imgsz=32, amp=False, default=16)

    assert result <= _BATCH_SAFE_MAX


def test_autobatch_returns_default_on_oom_at_first_probe():
    """OOM on batch=1 → < 2 probe points → return default."""
    call_count = [0]

    def mem_fn(*_):
        call_count[0] += 1
        raise RuntimeError("CUDA out of memory")

    model = nn.Linear(4, 2)
    with _make_cuda_patches(mem_fn, model=model):
        result = autobatch(model, imgsz=32, amp=False, default=7)

    assert result == 7


def test_autobatch_returns_default_on_oom_at_second_probe():
    """OOM on batch=2 leaves 1 point → fallback to default."""
    call_count = [0]

    def mem_fn(*_):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("CUDA out of memory")
        return int(0.5 * 1024**3)

    model = nn.Linear(4, 2)
    with _make_cuda_patches(mem_fn, model=model):
        result = autobatch(model, imgsz=32, amp=False, default=5)

    assert result == 5


# =============================================================================
# resolve_auto_batch — single-process path
# =============================================================================


def test_resolve_cpu_returns_default():
    model = nn.Linear(4, 2)
    result = resolve_auto_batch(model, imgsz=32, amp=False, world_size=1, default=12)
    assert result == 12


def test_resolve_rounds_down_to_world_size_multiple():
    """Result is always divisible by world_size."""
    with patch("libreyolo.training.autobatch.autobatch", return_value=13):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=4, default=16
        )
    # 13 // 4 * 4 = 12
    assert result % 4 == 0
    assert result == 12


def test_resolve_minimum_is_world_size():
    """Result is always ≥ world_size (each rank gets ≥1 sample)."""
    with patch("libreyolo.training.autobatch.autobatch", return_value=1):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=8, default=16
        )
    assert result >= 8


def test_resolve_exact_multiple_unchanged():
    """An autobatch result already divisible by world_size is returned as-is."""
    with patch("libreyolo.training.autobatch.autobatch", return_value=32):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=8, default=16
        )
    assert result == 32


# =============================================================================
# Trainer wiring: batch=-1 triggers resolve_auto_batch
# =============================================================================


try:
    from libreyolo.training.trainer import BaseTrainer as _BaseTrainer
    _HAS_TRAINER = True
except Exception:
    _BaseTrainer = object  # type: ignore[assignment,misc]
    _HAS_TRAINER = False

_requires_trainer = pytest.mark.skipif(not _HAS_TRAINER, reason="libreyolo trainer unavailable")


class _ConstScheduler:
    def update_lr(self, _): return 0.01


def _make_minimal_trainer(batch):
    class _MinimalTrainer(_BaseTrainer):
        def get_model_family(self): return "test"
        def get_model_tag(self): return "test"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters): return _ConstScheduler()
        def get_loss_components(self, outputs): return {}
        def on_forward(self, imgs, targets, polygons=None):
            return {"total_loss": self.model(imgs).mean()}
        def _setup_data(self):
            self.train_loader = [None] * 4  # fake length; content never iterated
        def _setup_optimizer(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.01)

    return _MinimalTrainer(
        model=nn.Linear(4, 2),
        num_classes=2,
        epochs=1,
        batch=batch,
        device="cpu",
        amp=False,
        ema=False,
    )


@_requires_trainer
def test_trainer_batch_minus1_calls_resolve_auto_batch():
    """setup() must call resolve_auto_batch and update config.batch when batch=-1."""
    trainer = _make_minimal_trainer(batch=-1)
    calls = []

    def _fake_resolve(m, imgsz, amp, world_size, **_):
        calls.append(world_size)
        return 24

    # Patch in the autobatch module (local import inside setup())
    with patch("libreyolo.training.autobatch.resolve_auto_batch", side_effect=_fake_resolve):
        trainer.setup()

    assert len(calls) == 1, "resolve_auto_batch must be called exactly once"
    assert trainer.config.batch == 24


# =============================================================================
# resolve_auto_batch — nbs-aware per-GPU scaling
# =============================================================================


def test_resolve_nbs_1gpu_uses_accumulation():
    """1 GPU: global = per_gpu (≤ nbs), accumulation makes up the rest."""
    # per_gpu=16, nbs=32, world_size=1 → global=min(16,32)=16
    with patch("libreyolo.training.autobatch.autobatch", return_value=16):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=1, nbs=32,
        )
    assert result == 16  # accumulate=round(32/16)=2 → effective=32=nbs


def test_resolve_nbs_2gpu_scales_to_nbs():
    """2 GPUs: global = per_gpu * world_size capped at nbs — no accumulation needed."""
    # per_gpu=16, nbs=32, world_size=2 → global=min(32,32)=32
    with patch("libreyolo.training.autobatch.autobatch", return_value=16):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=2, nbs=32,
        )
    assert result == 32  # per-rank=16, accumulate=1 → effective=32=nbs


def test_resolve_nbs_4gpu_caps_at_nbs():
    """4 GPUs with headroom: global is capped at nbs, not per_gpu * world_size."""
    # per_gpu=16, nbs=32, world_size=4 → min(64,32)=32 → round to mult of 4 → 32
    with patch("libreyolo.training.autobatch.autobatch", return_value=16):
        result = resolve_auto_batch(
            nn.Linear(4, 2), imgsz=32, amp=False, world_size=4, nbs=32,
        )
    assert result == 32  # per-rank=8, accumulate=1 → effective=32=nbs


def test_resolve_nbs_result_divisible_by_world_size():
    """Global batch is always divisible by world_size."""
    with patch("libreyolo.training.autobatch.autobatch", return_value=16):
        for ws in (1, 2, 4, 8):
            result = resolve_auto_batch(
                nn.Linear(4, 2), imgsz=32, amp=False, world_size=ws, nbs=64,
            )
            assert result % ws == 0, f"world_size={ws}: {result} not divisible"


def test_resolve_nbs_world_size_exceeds_nbs_warns(caplog):
    """world_size > nbs forces global > nbs — a warning must be emitted."""
    import logging
    with patch("libreyolo.training.autobatch.autobatch", return_value=4):
        with caplog.at_level(logging.WARNING, logger="libreyolo.training.autobatch"):
            result = resolve_auto_batch(
                nn.Linear(4, 2), imgsz=32, amp=False, world_size=8, nbs=4,
            )
    assert result >= 8
    assert any("world_size" in r.message and "nbs" in r.message for r in caplog.records)


@_requires_trainer
def test_trainer_explicit_batch_skips_resolve():
    """Explicit batch > 0 must never trigger autobatch."""
    trainer = _make_minimal_trainer(batch=8)
    calls = []

    def _fake_resolve(*a, **kw):
        calls.append(1)
        return 99

    with patch("libreyolo.training.autobatch.resolve_auto_batch", side_effect=_fake_resolve):
        trainer.setup()

    assert calls == [], "resolve_auto_batch must not be called for explicit batch"
    assert trainer.config.batch == 8
