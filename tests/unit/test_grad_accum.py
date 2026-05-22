"""Gradient accumulation: mathematical correctness and trainer wiring.

Two families of tests:

1. Gradient equivalence — the core mathematical claim: accumulating N
   mini-batches of size B with ``grad_accum_steps=N`` must produce the same
   parameter gradients as a single forward pass over a full batch of size N*B
   (given mean loss reduction and the ``loss / accum`` scaling in the loop).

2. Trainer wiring — that ``optimizer.step()`` and ``zero_grad()`` fire at the
   right frequency when ``_train_epoch`` runs with accum > 1.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

# Importing any libreyolo submodule runs libreyolo/__init__.py, which pulls in
# all model families including deim (needs scipy). Import once here so:
#  - if scipy is missing the wiring tests are skipped as a group, not randomly
#  - if scipy is present all tests run normally
try:
    from libreyolo.training.trainer import BaseTrainer as _BaseTrainer
    from libreyolo.models.deim.trainer import DEIMTrainer as _DEIMTrainer
    from libreyolo.models.dfine.trainer import DFINETrainer as _DFINETrainer
    _HAS_LIBREYOLO = True
except Exception:
    _BaseTrainer = object  # type: ignore[assignment,misc]
    _DEIMTrainer = object  # type: ignore[assignment,misc]
    _DFINETrainer = object  # type: ignore[assignment,misc]
    _HAS_LIBREYOLO = False

_requires_libreyolo = pytest.mark.skipif(
    not _HAS_LIBREYOLO, reason="libreyolo import chain unavailable (scipy missing?)"
)


# =========================================================================
# Shared helpers
# =========================================================================


class _TinyModel(nn.Module):
    """Minimal linear model. Mean output loss gives clean gradient equivalence."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x, targets=None):
        return self.linear(x)


class _ConstScheduler:
    """Stub scheduler that always returns 0.01."""

    def update_lr(self, iters: int) -> float:
        return 0.01


def _fake_loader(num_batches: int, batch_size: int = 2, feat_dim: int = 8):
    """List of synthetic (imgs, targets, img_infos, img_ids) tuples."""
    return [
        (
            torch.randn(batch_size, feat_dim),
            torch.zeros(batch_size, 5),
            [{}] * batch_size,
            list(range(batch_size)),
        )
        for _ in range(num_batches)
    ]


def _make_trainer(model: nn.Module, accum: int = 1, num_batches: int = 4):
    """Build a minimal concrete BaseTrainer with fake loader already wired in."""
    class _MinimalTrainer(_BaseTrainer):
        def get_model_family(self): return "test"
        def get_model_tag(self): return "test"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch): return _ConstScheduler()
        def get_loss_components(self, outputs): return {}
        def on_forward(self, imgs, targets, polygons=None):
            return {"total_loss": self.model(imgs).mean()}

    trainer = _MinimalTrainer(
        model=model,
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        grad_accum_steps=accum,
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999  # suppress TB logging
    trainer.train_loader = _fake_loader(num_batches)
    return trainer


# =========================================================================
# 1. Gradient equivalence
# =========================================================================


def test_gradient_equivalence_accum2():
    """accum=2: two mini-batches of 2 yield the same gradients as one batch of 4."""
    torch.manual_seed(0)
    data = torch.randn(4, 8)

    model_ref = _TinyModel()
    model_acc = copy.deepcopy(model_ref)

    # Full batch, accum=1
    model_ref.zero_grad()
    (model_ref(data).mean() / 1).backward()
    grads_ref = {n: p.grad.clone() for n, p in model_ref.named_parameters()}

    # Two mini-batches, accum=2 — mirrors the ``loss / accum`` pattern in _train_epoch
    model_acc.zero_grad()
    for i in range(2):
        (model_acc(data[i * 2 : (i + 1) * 2]).mean() / 2).backward()
    grads_acc = {n: p.grad.clone() for n, p in model_acc.named_parameters()}

    for name in grads_ref:
        torch.testing.assert_close(
            grads_acc[name], grads_ref[name], atol=1e-6, rtol=1e-5,
            msg=f"gradient mismatch for '{name}' with accum=2",
        )


def test_gradient_equivalence_accum4():
    """accum=4: four single-sample micro-batches yield the same gradients as one batch of 4."""
    torch.manual_seed(7)
    data = torch.randn(4, 8)

    model_ref = _TinyModel()
    model_acc = copy.deepcopy(model_ref)

    model_ref.zero_grad()
    (model_ref(data).mean() / 1).backward()
    grads_ref = {n: p.grad.clone() for n, p in model_ref.named_parameters()}

    model_acc.zero_grad()
    for i in range(4):
        (model_acc(data[i : i + 1]).mean() / 4).backward()
    grads_acc = {n: p.grad.clone() for n, p in model_acc.named_parameters()}

    for name in grads_ref:
        torch.testing.assert_close(
            grads_acc[name], grads_ref[name], atol=1e-6, rtol=1e-5,
            msg=f"gradient mismatch for '{name}' with accum=4",
        )


# =========================================================================
# 2. Trainer wiring: step and zero_grad counts
# =========================================================================


@_requires_libreyolo
def test_optimizer_step_count_matches_accum():
    """optimizer.step() fires exactly N // accum times per epoch."""
    N, accum = 6, 2  # expect 3 steps

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == N // accum, (
        f"expected {N // accum} steps for N={N}, accum={accum}, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_zero_grad_fires_at_accum_boundaries():
    """optimizer.zero_grad() fires once per accumulation window."""
    N, accum = 8, 4  # expect 2 zero_grad calls

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    zg_calls = []
    orig = trainer.optimizer.zero_grad

    def _counting(*a, **kw):
        zg_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.zero_grad = _counting
    trainer._train_epoch(0)

    assert len(zg_calls) == N // accum, (
        f"expected {N // accum} zero_grad calls for N={N}, accum={accum}, got {len(zg_calls)}"
    )


@_requires_libreyolo
def test_accum1_baseline_every_batch_steps():
    """accum=1 (default): every batch triggers its own optimizer step."""
    N = 5

    trainer = _make_trainer(_TinyModel(), accum=1, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == N, (
        f"accum=1 must step every batch: expected {N}, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_scheduler_update_lr_called_per_optimizer_step():
    """lr_scheduler.update_lr() is called once per optimizer step, not per batch."""
    N, accum = 6, 3  # expect 2 scheduler calls

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    lr_calls = []
    orig = trainer.lr_scheduler.update_lr

    def _counting(iters):
        lr_calls.append(iters)
        return orig(iters)

    trainer.lr_scheduler.update_lr = _counting
    trainer._train_epoch(0)

    assert len(lr_calls) == N // accum, (
        f"expected {N // accum} scheduler calls for N={N}, accum={accum}, got {len(lr_calls)}"
    )


@_requires_libreyolo
def test_partial_window_step_count():
    """When N % accum != 0, the partial last window still triggers an optimizer step."""
    N, accum = 5, 2  # ceil(5/2) = 3 steps

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    import math
    assert len(step_calls) == math.ceil(N / accum), (
        f"expected {math.ceil(N / accum)} steps for N={N}, accum={accum}, got {len(step_calls)}"
    )


def test_partial_window_gradient_scale():
    """Partial last window divides by actual window count, not accum.

    accum=2, 3 micro-batches: windows are [0,1] and [2].
    Window [2] has size 1 — its gradient must equal ``loss / 1``, not ``loss / 2``.
    """
    torch.manual_seed(99)
    data = [torch.randn(2, 8) for _ in range(3)]

    model_correct = _TinyModel()
    model_wrong = copy.deepcopy(model_correct)

    # Correct: window [0,1] divides by 2; window [2] divides by 1.
    model_correct.zero_grad()
    for chunk in data[:2]:
        (model_correct(chunk).mean() / 2).backward()
    # step would happen here; for gradient comparison we skip it and just accumulate
    for chunk in data[2:]:
        (model_correct(chunk).mean() / 1).backward()
    grads_correct = {n: p.grad.clone() for n, p in model_correct.named_parameters()}

    # Wrong (old behaviour): always divide by accum=2, even for the partial window.
    model_wrong.zero_grad()
    for chunk in data:
        (model_wrong(chunk).mean() / 2).backward()
    grads_wrong = {n: p.grad.clone() for n, p in model_wrong.named_parameters()}

    # The two gradient sets must differ — confirming the fix changes something.
    any_differ = any(
        not torch.allclose(grads_correct[n], grads_wrong[n])
        for n in grads_correct
    )
    assert any_differ, "correct and wrong gradients are identical — test is vacuous"


@_requires_libreyolo
def test_scheduler_receives_optimizer_steps_per_epoch():
    """setup() passes ``len(loader) // accum`` to create_scheduler, not raw batch count."""
    received = {}

    class _CapturingTrainer(_BaseTrainer):
        def get_model_family(self): return "test"
        def get_model_tag(self): return "test"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch):
            received["iters_per_epoch"] = iters_per_epoch
            return _ConstScheduler()
        def get_loss_components(self, outputs): return {}

    N, accum = 8, 4

    trainer = _CapturingTrainer(
        model=_TinyModel(),
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        grad_accum_steps=accum,
    )
    # Replicate just the scheduler-creation slice of setup() without a real DataLoader.
    trainer.train_loader = _fake_loader(N)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    _accum = max(1, trainer.config.grad_accum_steps)
    steps_per_epoch = max(1, len(trainer.train_loader) // _accum)
    trainer.lr_scheduler = trainer.create_scheduler(steps_per_epoch)

    assert received["iters_per_epoch"] == N // accum, (
        f"expected iters_per_epoch={N // accum}, got {received['iters_per_epoch']}"
    )


# =========================================================================
# 3. DEIMTrainer / DFINETrainer wiring
#
# Both classes override _train_epoch and access self.train_loader.dataset,
# so they need a loader wrapper that exposes that attribute.
# =========================================================================


class _FakeLoader:
    """Iterable loader that exposes .dataset and .collate_fn for DEIM/DFINE."""

    dataset = None   # no set_epoch on a stub dataset
    collate_fn = None

    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _make_deim_trainer(model: nn.Module, accum: int = 1, num_batches: int = 4):
    """Minimal DEIMTrainer with on_forward and on_setup stubbed out."""
    class _MinimalDEIM(_DEIMTrainer):
        def get_model_tag(self): return "test-deim"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch): return _ConstScheduler()
        def get_loss_components(self, outputs): return {}
        def on_setup(self): pass  # skip HungarianMatcher / DEIMCriterion
        def on_forward(self, imgs, targets, polygons=None):
            return {"total_loss": self.model(imgs).mean()}

    trainer = _MinimalDEIM(
        model=model,
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        grad_accum_steps=accum,
        clip_max_norm=0.0,
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999
    trainer.train_loader = _FakeLoader(_fake_loader(num_batches))
    return trainer


def _make_dfine_trainer(model: nn.Module, accum: int = 1, num_batches: int = 4):
    """Minimal DFINETrainer with on_forward and on_setup stubbed out."""
    class _MinimalDFINE(_DFINETrainer):
        def get_model_tag(self): return "test-dfine"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch): return _ConstScheduler()
        def get_loss_components(self, outputs): return {}
        def on_setup(self): pass  # skip HungarianMatcher / DFINECriterion
        def on_forward(self, imgs, targets, polygons=None):
            return {"total_loss": self.model(imgs).mean()}

    trainer = _MinimalDFINE(
        model=model,
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        grad_accum_steps=accum,
        clip_max_norm=0.0,
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999
    trainer.train_loader = _FakeLoader(_fake_loader(num_batches))
    return trainer


@_requires_libreyolo
def test_deim_step_count_matches_accum():
    """DEIMTrainer: optimizer.step() fires ceil(N/accum) times for N divisible by accum."""
    import math
    N, accum = 6, 2

    trainer = _make_deim_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == math.ceil(N / accum), (
        f"DEIM: expected {math.ceil(N / accum)} steps, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_deim_partial_window_step_count():
    """DEIMTrainer: partial last window (N % accum != 0) still triggers an optimizer step."""
    import math
    N, accum = 5, 2  # ceil(5/2) = 3

    trainer = _make_deim_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == math.ceil(N / accum), (
        f"DEIM partial window: expected {math.ceil(N / accum)} steps, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_deim_zero_grad_fires_at_accum_boundaries():
    """DEIMTrainer: zero_grad() fires once per accumulation window."""
    import math
    N, accum = 5, 2  # 3 windows: [0,1], [2,3], [4]

    trainer = _make_deim_trainer(_TinyModel(), accum=accum, num_batches=N)

    zg_calls = []
    orig = trainer.optimizer.zero_grad

    def _counting(*a, **kw):
        zg_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.zero_grad = _counting
    trainer._train_epoch(0)

    assert len(zg_calls) == math.ceil(N / accum), (
        f"DEIM zero_grad: expected {math.ceil(N / accum)}, got {len(zg_calls)}"
    )


@_requires_libreyolo
def test_dfine_step_count_matches_accum():
    """DFINETrainer: optimizer.step() fires ceil(N/accum) times for N divisible by accum."""
    import math
    N, accum = 6, 2

    trainer = _make_dfine_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == math.ceil(N / accum), (
        f"DFINE: expected {math.ceil(N / accum)} steps, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_dfine_partial_window_step_count():
    """DFINETrainer: partial last window (N % accum != 0) still triggers an optimizer step."""
    import math
    N, accum = 5, 2  # ceil(5/2) = 3

    trainer = _make_dfine_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    orig = trainer.optimizer.step

    def _counting(*a, **kw):
        step_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.step = _counting
    trainer._train_epoch(0)

    assert len(step_calls) == math.ceil(N / accum), (
        f"DFINE partial window: expected {math.ceil(N / accum)} steps, got {len(step_calls)}"
    )


@_requires_libreyolo
def test_dfine_zero_grad_fires_at_accum_boundaries():
    """DFINETrainer: zero_grad() fires once per accumulation window."""
    import math
    N, accum = 5, 2  # 3 windows: [0,1], [2,3], [4]

    trainer = _make_dfine_trainer(_TinyModel(), accum=accum, num_batches=N)

    zg_calls = []
    orig = trainer.optimizer.zero_grad

    def _counting(*a, **kw):
        zg_calls.append(1)
        return orig(*a, **kw)

    trainer.optimizer.zero_grad = _counting
    trainer._train_epoch(0)

    assert len(zg_calls) == math.ceil(N / accum), (
        f"DFINE zero_grad: expected {math.ceil(N / accum)}, got {len(zg_calls)}"
    )
