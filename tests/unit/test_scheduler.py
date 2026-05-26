"""Unit tests for libreyolo.training.scheduler.

Focuses on edge cases: zero warmup, LR monotonicity, boundary values.
"""

from __future__ import annotations

import math

import pytest

from libreyolo.training.scheduler import WarmupCosineScheduler

pytestmark = pytest.mark.unit


# =============================================================================
# WarmupCosineScheduler — warmup_epochs=0 regression (ZeroDivisionError fix)
# =============================================================================


def test_warmup_cosine_zero_warmup_does_not_raise_at_iter_zero():
    """warmup_epochs=0: update_lr(0) must not divide by zero."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0
    )
    lr = sched.update_lr(0)
    assert isinstance(lr, float)
    assert 0.0 <= lr <= 0.01


def test_warmup_cosine_zero_warmup_full_lr_at_iter_zero():
    """With no warmup and no plateau, iter=0 is the peak of the cosine → lr."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0, plateau_epochs=0
    )
    assert sched.update_lr(0) == pytest.approx(0.01, abs=1e-9)


def test_warmup_cosine_zero_warmup_monotone_decay():
    """With no warmup the LR must be non-increasing for every iteration."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=10, total_epochs=10, warmup_epochs=0, plateau_epochs=0
    )
    lrs = [sched.update_lr(i) for i in range(100)]
    for prev, curr in zip(lrs, lrs[1:]):
        assert prev >= curr - 1e-10, f"LR increased: {prev} → {curr}"


# =============================================================================
# WarmupCosineScheduler — normal warmup sanity checks
# =============================================================================


def test_warmup_cosine_starts_at_warmup_lr_start():
    """iter=0 with warmup active must equal warmup_lr_start (quadratic: 0^2 = 0)."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=5, warmup_lr_start=0.0
    )
    assert sched.update_lr(0) == pytest.approx(0.0, abs=1e-9)


def test_warmup_cosine_peaks_at_end_of_warmup():
    """At the last warmup iter the LR should equal lr (quadratic reaches 1.0)."""
    iters_per_epoch = 100
    warmup_epochs = 5
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=iters_per_epoch, total_epochs=100, warmup_epochs=warmup_epochs
    )
    warmup_iters = iters_per_epoch * warmup_epochs
    assert sched.update_lr(warmup_iters) == pytest.approx(0.01, abs=1e-9)


def test_warmup_cosine_plateau_is_min_lr():
    """Iterations past the plateau boundary must return min_lr."""
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=100, total_epochs=100, warmup_epochs=0,
        plateau_epochs=10, min_lr_ratio=0.05
    )
    min_lr = 0.01 * 0.05
    total_iters = 100 * 100
    plateau_iters = 100 * 10
    # Last iteration is firmly in plateau
    assert sched.update_lr(total_iters - plateau_iters) == pytest.approx(min_lr, abs=1e-9)
    assert sched.update_lr(total_iters) == pytest.approx(min_lr, abs=1e-9)


def test_warmup_cosine_cosine_midpoint():
    """At the midpoint of cosine annealing the LR should be midway between lr and min_lr."""
    iters_per_epoch = 100
    total_epochs = 100
    sched = WarmupCosineScheduler(
        lr=0.01, iters_per_epoch=iters_per_epoch, total_epochs=total_epochs,
        warmup_epochs=0, plateau_epochs=0, min_lr_ratio=0.0
    )
    mid = (iters_per_epoch * total_epochs) // 2
    # cos(pi * 0.5) = 0 → lr = min_lr + 0.5*(lr - min_lr)*(1+0) = lr/2
    assert sched.update_lr(mid) == pytest.approx(0.005, abs=1e-3)
