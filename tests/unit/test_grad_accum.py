"""Gradient accumulation: mathematical correctness and trainer wiring.

Two families of tests:

1. Gradient equivalence — the core mathematical claim: accumulating N
   mini-batches of size B (``nbs`` set so ``round(nbs / batch) == N``) must
   produce the same parameter gradients as a single forward pass over a full
   batch of size N*B (given mean loss reduction and the ``loss / accum``
   scaling in the loop).

2. Trainer wiring — that ``optimizer.step()`` and ``zero_grad()`` fire at the
   right frequency when ``_train_epoch_accum`` runs with accum > 1.
"""

from __future__ import annotations

import copy
import hashlib
import math

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


def _fake_image_loader(
    num_batches: int,
    batch_size: int = 2,
    imgsz: int = 320,
    max_labels: int = 30,
    num_classes: int = 2,
):
    """Synthetic image batches with one valid xywh target per image."""
    batches = []
    for _ in range(num_batches):
        imgs = torch.randn(batch_size, 3, imgsz, imgsz)
        targets = torch.zeros(batch_size, max_labels, 5)
        for i in range(batch_size):
            cls = float(i % num_classes)
            cx = imgsz * (0.35 + 0.1 * (i % 2))
            cy = imgsz * (0.40 + 0.1 * (i % 2))
            targets[i, 0] = torch.tensor(
                [cls, cx, cy, imgsz * 0.25, imgsz * 0.20]
            )
        batches.append((imgs, targets, [{}] * batch_size, list(range(batch_size))))
    return batches


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
        nbs=2 * accum,  # helpers use batch=2, so round(nbs / batch) == accum
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999  # suppress TB logging
    trainer.train_loader = _fake_loader(num_batches)
    return trainer


def _hash_state_dict(model: nn.Module) -> str:
    """Stable byte hash for tensors in a state_dict."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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
    """create_scheduler receives ceil(len(loader) / accum) — the optimizer-step count.

    N is deliberately not divisible by accum (ceil=3, floor would be 2) so the
    partial-window rounding is observable, and the test exercises the real
    production helper ``_scheduler_steps_per_epoch`` rather than re-deriving it.
    """
    import math

    received = {}

    class _CapturingTrainer(_BaseTrainer):
        def get_model_family(self): return "test"
        def get_model_tag(self): return "test"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch):
            received["iters_per_epoch"] = iters_per_epoch
            return _ConstScheduler()
        def get_loss_components(self, outputs): return {}

    N, accum = 10, 4  # ceil(10/4) = 3 optimizer steps; floor would be 2

    trainer = _CapturingTrainer(
        model=_TinyModel(),
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        nbs=2 * accum,  # helpers use batch=2, so round(nbs / batch) == accum
    )
    trainer.train_loader = _fake_loader(N)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    # Exercise the real production helper used by setup(), not a re-derivation.
    trainer.lr_scheduler = trainer.create_scheduler(
        trainer._scheduler_steps_per_epoch()
    )

    assert received["iters_per_epoch"] == math.ceil(N / accum) == 3, (
        f"expected ceil({N}/{accum})=3 optimizer steps, "
        f"got {received['iters_per_epoch']}"
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
        nbs=2 * accum,  # helpers use batch=2, so round(nbs / batch) == accum
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
        nbs=2 * accum,  # helpers use batch=2, so round(nbs / batch) == accum
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


# =========================================================================
# 4. AMP and EMA wiring under accumulation
#
# The wiring tests above run with amp=False / ema=False. These cover the
# remaining branches of _train_epoch_accum: the AMP (GradScaler) path and the
# EMA update. Stand-ins keep both exercisable on CPU without CUDA.
# =========================================================================


class _FakeScaler:
    """Minimal ``GradScaler`` stand-in that records the AMP call sequence."""

    def __init__(self):
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def unscale_(self, optimizer):
        self.unscale_calls += 1

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


class _FakeEMA:
    """EMA stand-in that counts ``update()`` calls."""

    def __init__(self):
        self.updates = 0

    def update(self, model):
        self.updates += 1


@_requires_libreyolo
def test_amp_accum_steps_once_per_window():
    """AMP path: scaler.step/update fire once per window; scale fires every batch."""
    import math
    N, accum = 7, 3  # ceil(7/3) = 3 windows

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)
    scaler = _FakeScaler()
    trainer.scaler = scaler

    trainer._train_epoch(0)

    assert scaler.scale_calls == N, (
        f"scaler.scale should fire every batch: {scaler.scale_calls} != {N}"
    )
    assert scaler.step_calls == math.ceil(N / accum), (
        f"scaler.step should fire once per window: "
        f"{scaler.step_calls} != {math.ceil(N / accum)}"
    )
    assert scaler.update_calls == math.ceil(N / accum), (
        f"scaler.update should fire once per window: "
        f"{scaler.update_calls} != {math.ceil(N / accum)}"
    )


@_requires_libreyolo
def test_amp_accum_clips_once_per_window():
    """AMP path with clipping on: unscale_ fires once per window, not per batch.

    Guards the conflict resolution that gated BaseTrainer's gradient clipping
    on the optimizer step. Clipping a half-accumulated gradient on every
    micro-batch would be wrong — and would call unscale_ N times instead of
    ceil(N / accum).
    """
    import math
    N, accum = 7, 3

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)
    trainer.scaler = _FakeScaler()
    trainer.config.clip_max_norm = 1.0  # enable BaseTrainer gradient clipping

    trainer._train_epoch(0)

    assert trainer.scaler.unscale_calls == math.ceil(N / accum), (
        f"unscale_ (clipping) should fire once per window: "
        f"{trainer.scaler.unscale_calls} != {math.ceil(N / accum)}"
    )


@_requires_libreyolo
def test_ema_updates_once_per_optimizer_step():
    """EMA updates once per optimizer step under accumulation, not per micro-batch."""
    import math
    N, accum = 7, 3

    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)
    ema = _FakeEMA()
    trainer.ema_model = ema

    trainer._train_epoch(0)

    assert ema.updates == math.ceil(N / accum), (
        f"EMA should update once per optimizer step: "
        f"{ema.updates} != {math.ceil(N / accum)}"
    )


# =========================================================================
# 5. Stress paths added after PR #236 review
# =========================================================================


@_requires_libreyolo
@pytest.mark.parametrize(
    ("nbs", "expected_accum"),
    [
        (None, 1),
        (1, 1),  # nbs < batch
        (2, 1),  # nbs == batch
    ],
)
def test_nbs_edge_values_keep_standard_path(nbs, expected_accum):
    """nbs unset, below batch, or equal to batch must not dispatch to accum."""
    trainer = _make_trainer(_TinyModel(), accum=1, num_batches=3)
    trainer.config.nbs = nbs

    def _should_not_run(epoch):
        raise AssertionError("_train_epoch_accum should not run when accum is 1")

    trainer._train_epoch_accum = _should_not_run
    step_calls = []
    orig = trainer.optimizer.step

    def _counting_step(*args, **kwargs):
        step_calls.append(1)
        return orig(*args, **kwargs)

    trainer.optimizer.step = _counting_step
    trainer._train_epoch(0)

    assert trainer._accum_steps == expected_accum
    assert len(step_calls) == len(trainer.train_loader)


@_requires_libreyolo
def test_nbs_larger_than_epoch_still_steps_once():
    """When accum exceeds loader length, the final partial window still steps."""
    N, accum = 3, 10
    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    step_calls = []
    scheduler_calls = []
    zero_grad_calls = []
    orig_step = trainer.optimizer.step
    orig_zero_grad = trainer.optimizer.zero_grad
    orig_update_lr = trainer.lr_scheduler.update_lr

    def _counting_step(*args, **kwargs):
        step_calls.append(1)
        return orig_step(*args, **kwargs)

    def _counting_zero_grad(*args, **kwargs):
        zero_grad_calls.append(1)
        return orig_zero_grad(*args, **kwargs)

    def _counting_update_lr(iteration):
        scheduler_calls.append(iteration)
        return orig_update_lr(iteration)

    trainer.optimizer.step = _counting_step
    trainer.optimizer.zero_grad = _counting_zero_grad
    trainer.lr_scheduler.update_lr = _counting_update_lr

    trainer._train_epoch(0)

    assert len(step_calls) == 1
    assert len(zero_grad_calls) == 1
    assert scheduler_calls == [1]


@_requires_libreyolo
def test_nondivisible_actual_train_loop_scales_partial_window_by_actual_count():
    """The real accum loop divides a short final window by its own length."""

    class _ScalarModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, coeff):
            return self.weight * coeff.squeeze()

    class _ScalarTrainer(_BaseTrainer):
        def get_model_family(self): return "test"
        def get_model_tag(self): return "test"
        def create_transforms(self): return None, None
        def create_scheduler(self, iters_per_epoch): return _ConstScheduler()
        def get_loss_components(self, outputs): return {}
        def on_forward(self, imgs, targets, polygons=None):
            return {"total_loss": self.model(imgs)}

    model = _ScalarModel()
    trainer = _ScalarTrainer(
        model=model,
        num_classes=1,
        epochs=1,
        batch=2,
        device="cpu",
        amp=False,
        ema=False,
        nbs=4,  # accum=2
    )
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999
    trainer.train_loader = [
        (torch.tensor([2.0]), torch.zeros(1, 5), [{}], [0]),
        (torch.tensor([4.0]), torch.zeros(1, 5), [{}], [1]),
        (torch.tensor([6.0]), torch.zeros(1, 5), [{}], [2]),
    ]

    grads_at_step = []
    orig_step = trainer.optimizer.step

    def _recording_step(*args, **kwargs):
        grads_at_step.append(model.weight.grad.detach().item())
        return orig_step(*args, **kwargs)

    trainer.optimizer.step = _recording_step
    trainer._train_epoch(0)

    assert grads_at_step == pytest.approx([3.0, 6.0])


class _CountingRealScaler:
    """Delegates to torch.amp.GradScaler while recording call order."""

    def __init__(self):
        self.inner = torch.amp.GradScaler("cpu")
        self.events = []

    def scale(self, loss):
        self.events.append("scale")
        return self.inner.scale(loss)

    def unscale_(self, optimizer):
        self.events.append("unscale")
        return self.inner.unscale_(optimizer)

    def step(self, optimizer):
        self.events.append("step")
        return self.inner.step(optimizer)

    def update(self):
        self.events.append("update")
        return self.inner.update()


@_requires_libreyolo
def test_real_grad_scaler_steps_once_per_accum_window():
    """A real torch GradScaler advances unscale/step/update once per window."""
    N, accum = 5, 2
    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)
    trainer.scaler = _CountingRealScaler()
    trainer.config.clip_max_norm = 1.0

    trainer._train_epoch(0)

    assert trainer.scaler.events == [
        "scale", "scale", "unscale", "step", "update",
        "scale", "scale", "unscale", "step", "update",
        "scale", "unscale", "step", "update",
    ]


@_requires_libreyolo
def test_base_accum_gradient_clipping_sees_accumulated_norm(monkeypatch):
    """Clipping runs once after both micro-batches have contributed gradients."""
    torch.manual_seed(123)
    data = [
        (
            torch.randn(2, 8),
            torch.zeros(2, 5),
            [{}] * 2,
            [0, 1],
        ),
        (
            torch.randn(2, 8),
            torch.zeros(2, 5),
            [{}] * 2,
            [2, 3],
        ),
    ]
    model = _TinyModel()
    expected_model = copy.deepcopy(model)

    expected_model.zero_grad()
    for imgs, *_ in data:
        (expected_model(imgs).mean() / 2).backward()
    expected_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                p.grad.detach().norm(2)
                for p in expected_model.parameters()
                if p.grad is not None
            ]
        ),
        ord=2,
    )

    trainer = _make_trainer(model, accum=2, num_batches=2)
    trainer.train_loader = data
    trainer.config.clip_max_norm = 0.05

    orig_clip = torch.nn.utils.clip_grad_norm_
    seen_norms = []

    def _recording_clip(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        seen_norms.append(
            torch.linalg.vector_norm(
                torch.stack(
                    [
                        p.grad.detach().norm(2)
                        for p in params
                        if p.grad is not None
                    ]
                ),
                ord=2,
            )
        )
        return orig_clip(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _recording_clip)
    trainer._train_epoch(0)

    assert len(seen_norms) == 1
    torch.testing.assert_close(seen_norms[0], expected_norm, rtol=1e-6, atol=1e-6)


@_requires_libreyolo
def test_scheduler_advances_ceil_windows_for_nondivisible_epoch():
    """Scheduler update count follows ceil(N / accum), including partial window."""
    N, accum = 10, 3
    trainer = _make_trainer(_TinyModel(), accum=accum, num_batches=N)

    calls = []
    orig = trainer.lr_scheduler.update_lr

    def _counting_update_lr(iteration):
        calls.append(iteration)
        return orig(iteration)

    trainer.lr_scheduler.update_lr = _counting_update_lr
    trainer._train_epoch(0)

    assert calls == [1, 2, 3, 4]
    assert len(calls) == math.ceil(N / accum)


@_requires_libreyolo
def test_real_ema_updates_and_stays_finite_per_optimizer_step():
    """ModelEMA uses optimizer-step cadence, not micro-batch cadence."""
    from libreyolo.training.ema import ModelEMA

    N, accum = 5, 2
    model = _TinyModel()
    trainer = _make_trainer(model, accum=accum, num_batches=N)
    ema = ModelEMA(model, decay=0.9)
    before = _hash_state_dict(ema.ema)
    trainer.ema_model = ema

    trainer._train_epoch(0)

    assert ema.updates == math.ceil(N / accum)
    assert _hash_state_dict(ema.ema) != before
    assert all(torch.isfinite(p).all() for p in ema.ema.parameters())


@_requires_libreyolo
def test_resume_checkpoint_preserves_nbs_and_optimizer_step_iter(tmp_path):
    """Saved config carries nbs; resumed accum current_iter remains step-based."""
    trainer = _make_trainer(_TinyModel(), accum=2, num_batches=3)
    trainer.save_dir = tmp_path / "first"
    trainer.save_dir.mkdir()
    trainer._save_checkpoint(epoch=0, loss=1.0, is_best=True)
    checkpoint_path = trainer.save_dir / "weights" / "last.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["nbs"] == 4

    resumed = _make_trainer(_TinyModel(), accum=2, num_batches=3)
    resumed.save_dir = tmp_path / "second"
    resumed.resume(str(checkpoint_path))
    assert resumed.start_epoch == 1
    assert resumed.config.nbs == 4

    resumed._train_epoch(resumed.start_epoch)
    assert resumed.current_iter == 3


@_requires_libreyolo
def test_default_path_hash_matches_accum_disabled_reference():
    """nbs=None is byte-identical to explicit accum=1 on the standard path."""
    torch.manual_seed(2026)
    batches = _fake_loader(4)
    model_default = _TinyModel()
    model_reference = copy.deepcopy(model_default)

    trainer_default = _make_trainer(model_default, accum=1, num_batches=4)
    trainer_reference = _make_trainer(model_reference, accum=1, num_batches=4)
    trainer_default.train_loader = batches
    trainer_reference.train_loader = copy.deepcopy(batches)
    trainer_default.config.nbs = None
    trainer_reference.config.nbs = trainer_reference.config.batch

    trainer_default._train_epoch(0)
    trainer_reference._train_epoch(0)

    assert _hash_state_dict(model_default) == _hash_state_dict(model_reference)


def _run_real_accum_epoch(trainer, num_batches: int):
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    trainer.lr_scheduler = _ConstScheduler()
    trainer.scaler = None
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.config.log_interval = 9999
    trainer.train_loader = _FakeLoader(
        _fake_image_loader(
            num_batches,
            batch_size=trainer.config.batch,
            imgsz=trainer.config.imgsz,
            max_labels=120,
            num_classes=trainer.config.num_classes,
        )
    )

    step_calls = []
    orig_step = trainer.optimizer.step

    def _counting_step(*args, **kwargs):
        step_calls.append(1)
        return orig_step(*args, **kwargs)

    trainer.optimizer.step = _counting_step
    avg_loss, *_ = trainer._train_epoch(0)
    return avg_loss, step_calls


@_requires_libreyolo
@pytest.mark.parametrize("family", ["yolo9", "dfine", "deim", "rfdetr"])
def test_real_models_accum_epoch_finite_loss_and_step_count(family):
    """Real model/trainers run one accum epoch on CPU with finite loss."""
    torch.manual_seed(1234)
    N, accum = 3, 2

    if family == "yolo9":
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        wrapper = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
        trainer = YOLO9Trainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="t",
            num_classes=2,
            data=None,
            epochs=1,
            batch=2,
            imgsz=320,
            device="cpu",
            amp=False,
            ema=False,
            eval_interval=-1,
            nbs=4,
        )
    elif family == "dfine":
        from libreyolo import LibreDFINE
        from libreyolo.models.dfine.trainer import DFINETrainer

        wrapper = LibreDFINE(None, size="n", nb_classes=2, device="cpu")
        trainer = DFINETrainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="n",
            num_classes=2,
            data=None,
            epochs=1,
            batch=2,
            imgsz=320,
            device="cpu",
            amp=False,
            ema=False,
            eval_interval=-1,
            nbs=4,
            clip_max_norm=0.0,
        )
        trainer.on_setup()
    elif family == "deim":
        from libreyolo import LibreDEIM
        from libreyolo.models.deim.trainer import DEIMTrainer

        wrapper = LibreDEIM(None, size="n", nb_classes=2, device="cpu")
        trainer = DEIMTrainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="n",
            num_classes=2,
            data=None,
            epochs=1,
            batch=2,
            imgsz=320,
            device="cpu",
            amp=False,
            ema=False,
            eval_interval=-1,
            nbs=4,
            clip_max_norm=0.0,
        )
        trainer.on_setup()
    else:
        from libreyolo.models.rfdetr.model import LibreRFDETR
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        wrapper = LibreRFDETR(model_path={}, size="n", nb_classes=2, device="cpu")
        trainer = RFDETRTrainer(
            model=wrapper.model,
            wrapper_model=wrapper,
            size="n",
            num_classes=2,
            data=None,
            epochs=1,
            batch=2,
            imgsz=320,
            device="cpu",
            amp=False,
            ema=False,
            eval_interval=-1,
            nbs=4,
            clip_max_norm=0.0,
        )
        trainer.on_setup()

    avg_loss, step_calls = _run_real_accum_epoch(trainer, N)

    assert math.isfinite(avg_loss)
    assert avg_loss > 0
    assert len(step_calls) == math.ceil(N / accum)
