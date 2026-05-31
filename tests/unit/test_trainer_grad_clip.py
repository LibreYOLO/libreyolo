"""Unit tests for BaseTrainer gradient clipping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


class TinyTrainer(BaseTrainer):
    def get_model_family(self) -> str:
        return "tiny"

    def get_model_tag(self) -> str:
        return "tiny"

    def create_transforms(self):
        raise NotImplementedError

    def create_scheduler(self, iters_per_epoch: int):
        raise NotImplementedError

    def get_loss_components(self, outputs):
        return {}


class OneBatchLoader:
    def __init__(self):
        self.dataset = SimpleNamespace()

    def __iter__(self):
        imgs = torch.zeros(1, 1)
        targets = torch.zeros(1, 1)
        yield imgs, targets, (None,), (0,)

    def __len__(self):
        return 1


class MultiBatchLoader:
    def __init__(self, num_batches):
        self.dataset = SimpleNamespace()
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            imgs = torch.zeros(1, 1)
            targets = torch.zeros(1, 1)
            yield imgs, targets, (None,), (0,)

    def __len__(self):
        return self.num_batches


class DummyLoss:
    def __init__(self, param, events):
        self.param = param
        self.events = events

    def backward(self):
        self.events.append("backward")
        self.param.grad = torch.ones_like(self.param)

    def item(self):
        return 1.0


class ScaledLoss:
    def __init__(self, loss, events):
        self.loss = loss
        self.events = events

    def backward(self):
        self.events.append("scaled_backward")
        self.loss.backward()


class FakeScaler:
    def __init__(self, events, fail_on_unscale=False):
        self.events = events
        self.fail_on_unscale = fail_on_unscale

    def scale(self, loss):
        self.events.append("scale")
        return ScaledLoss(loss, self.events)

    def unscale_(self, optimizer):
        if self.fail_on_unscale:
            raise AssertionError("unscale_ should not be called")
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("scaler_step")
        optimizer.step()

    def update(self):
        self.events.append("scaler_update")


def _build_trainer(*, clip_max_norm=None, scaler=None, events=None):
    events = events if events is not None else []
    trainer = TinyTrainer.__new__(TinyTrainer)
    trainer.model = nn.Linear(1, 1, bias=False)
    param = next(trainer.model.parameters())
    trainer.train_loader = OneBatchLoader()
    trainer.config = SimpleNamespace(
        epochs=1,
        batch=1,
        log_interval=999,
        eval_interval=-1,
    )
    if clip_max_norm != "missing":
        trainer.config.clip_max_norm = clip_max_norm
    trainer.device = torch.device("cpu")
    trainer.optimizer = torch.optim.SGD([param], lr=0.1)
    trainer.scaler = scaler
    trainer.ema_model = None
    trainer.lr_scheduler = SimpleNamespace(update_lr=lambda _: 0.1)
    trainer.wrapper_model = SimpleNamespace(task="detect")

    def on_forward(imgs, targets, polygons=None):
        events.append("forward")
        return {"total_loss": DummyLoss(param, events)}

    trainer.on_forward = on_forward
    return trainer, param, events


def _wrap_optimizer_steps(trainer, events):
    original_zero_grad = trainer.optimizer.zero_grad
    original_step = trainer.optimizer.step

    def zero_grad(*args, **kwargs):
        events.append("zero_grad")
        return original_zero_grad(*args, **kwargs)

    def step(*args, **kwargs):
        events.append("step")
        return original_step(*args, **kwargs)

    trainer.optimizer.zero_grad = zero_grad
    trainer.optimizer.step = step


@pytest.mark.parametrize("clip_max_norm", ["missing", None, 0.0])
def test_gradient_clipping_disabled_is_noop(monkeypatch, clip_max_norm):
    events = []
    scaler = FakeScaler(events, fail_on_unscale=True)
    trainer, _, events = _build_trainer(
        clip_max_norm=clip_max_norm,
        scaler=scaler,
        events=events,
    )

    def fail_clip(*args, **kwargs):
        raise AssertionError("clip_grad_norm_ should not be called")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail_clip)

    avg_loss, val_metrics, loss_items, lr = TinyTrainer._train_epoch(trainer, 0)

    assert avg_loss == pytest.approx(1.0)
    assert val_metrics is None
    assert loss_items == {}
    assert lr == {"group0": pytest.approx(0.1)}
    assert "unscale" not in events


def test_non_amp_gradient_clipping_runs_between_backward_and_step(monkeypatch):
    events = []
    trainer, param, events = _build_trainer(clip_max_norm=0.1, events=events)
    _wrap_optimizer_steps(trainer, events)

    def fake_clip(params, max_norm):
        clipped = list(params)
        events.append("clip")
        assert clipped == [param]
        assert max_norm == pytest.approx(0.1)
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    TinyTrainer._train_epoch(trainer, 0)

    assert events[:5] == ["forward", "zero_grad", "backward", "clip", "step"]


def test_amp_gradient_clipping_unscales_before_clip(monkeypatch):
    events = []
    scaler = FakeScaler(events)
    trainer, param, events = _build_trainer(
        clip_max_norm=0.1,
        scaler=scaler,
        events=events,
    )
    _wrap_optimizer_steps(trainer, events)

    def fake_clip(params, max_norm):
        clipped = list(params)
        events.append("clip")
        assert clipped == [param]
        assert max_norm == pytest.approx(0.1)
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)

    TinyTrainer._train_epoch(trainer, 0)

    assert events[:8] == [
        "forward",
        "zero_grad",
        "scale",
        "scaled_backward",
        "backward",
        "unscale",
        "clip",
        "scaler_step",
    ]
    assert events[8:10] == ["step", "scaler_update"]


def test_nbs_accumulation_delays_optimizer_step():
    events = []
    trainer, param, events = _build_trainer(clip_max_norm=0.0, events=events)
    trainer.train_loader = MultiBatchLoader(num_batches=3)
    trainer.config.nbs = trainer.config.batch * 2
    _wrap_optimizer_steps(trainer, events)

    def on_forward(imgs, targets, polygons=None):
        events.append("forward")
        return {"total_loss": (param * 0).sum() + 1.0}

    trainer.on_forward = on_forward

    TinyTrainer._train_epoch(trainer, 0)

    assert events == [
        "zero_grad",
        "forward",
        "forward",
        "step",
        "zero_grad",
        "forward",
        "step",
    ]


def test_scheduler_warmup_lr_is_applied_before_first_step():
    trainer, param, _ = _build_trainer(clip_max_norm=0.0)
    trainer.optimizer = torch.optim.SGD(
        [
            {"params": [param], "lr": 0.1, "lr_mult": 0.5},
        ],
        lr=0.1,
    )
    trainer._scale_lr = lambda base_lr, group: base_lr * group.get("lr_mult", 1.0)

    class WarmupScheduler:
        warmup_iters = 4

        def update_lr(self, iters):
            return 0.01 if iters == 0 else 0.1

    trainer.lr_scheduler = WarmupScheduler()

    trainer._initialize_scheduler_lr()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


@pytest.mark.parametrize("clip_max_norm", [-0.1, float("nan"), float("inf"), "bad"])
def test_invalid_clip_max_norm_raises(clip_max_norm):
    trainer, _, _ = _build_trainer(clip_max_norm=clip_max_norm)

    with pytest.raises(ValueError, match="clip_max_norm"):
        trainer._get_clip_max_norm()


def test_initialize_scheduler_lr_fastforwards_on_resume():
    """On resume, _initialize_scheduler_lr fast-forwards past warmup instead of resetting to iter 0."""
    trainer, param, _ = _build_trainer(clip_max_norm=0.0)
    trainer.optimizer = torch.optim.SGD(
        [{"params": [param], "lr": 0.1, "lr_mult": 1.0}],
        lr=0.1,
    )
    trainer._scale_lr = lambda base_lr, group: base_lr * group.get("lr_mult", 1.0)

    class WarmupScheduler:
        warmup_iters = 4

        def update_lr(self, iters):
            # Warmup-start (iter 0) is near-zero; post-warmup (iter > 4) is 0.1.
            return 0.0001 if iters == 0 else 0.1

    trainer.lr_scheduler = WarmupScheduler()
    # Simulate a resume from epoch 5 (warmup is long past).
    # OneBatchLoader has len=1, so steps_per_epoch=1, init_iter=5.
    trainer.start_epoch = 5

    trainer._initialize_scheduler_lr()

    # Must use post-warmup LR, not the near-zero warmup-start.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_resume_before_setup_defers_optimizer_state(tmp_path):
    """Optimizer state is deferred when resume() is called before setup() exists."""
    # Build a trainer and do one step to populate SGD momentum buffers.
    trainer, param, _ = _build_trainer(clip_max_norm=0.0)
    param.grad = torch.ones_like(param)
    trainer.optimizer.step()
    saved_lr = trainer.optimizer.param_groups[0]["lr"]
    saved_opt_state = trainer.optimizer.state_dict()

    # Persist a minimal checkpoint (validate_checkpoint_metadata warns but doesn't raise).
    ckpt_path = tmp_path / "last.pt"
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "epoch": 2,
            "optimizer": saved_opt_state,
        },
        ckpt_path,
    )

    # New trainer with no optimizer yet (simulates pre-setup() state).
    trainer2, param2, _ = _build_trainer(clip_max_norm=0.0)
    trainer2.optimizer = None

    trainer2.resume(str(ckpt_path))

    # State must be deferred, not silently dropped.
    assert getattr(trainer2, "_resume_optimizer_state", None) is not None
    assert trainer2.start_epoch == 3

    # Simulate what setup() does: create optimizer then apply deferred state.
    trainer2.optimizer = torch.optim.SGD([param2], lr=0.9)  # wrong LR on purpose
    trainer2.optimizer.load_state_dict(trainer2._resume_optimizer_state)
    trainer2._resume_optimizer_state = None

    # The restored optimizer must carry the saved LR, not the initialisation LR.
    assert trainer2.optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)
