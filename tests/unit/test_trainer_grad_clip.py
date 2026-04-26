"""Tests for gradient clipping in BaseTrainer.

Asserts that ``BaseTrainer._backward_and_step`` honours ``config.grad_clip``
on both the AMP and non-AMP code paths.  Future refactors that silently drop
the ``clip_grad_norm_`` call will fail these tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Type
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from libreyolo.training.config import TrainConfig
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


@dataclass(kw_only=True)
class _BareConfig(TrainConfig):
    """Minimal config used by the bare trainer below."""

    optimizer: str = "sgd"
    lr0: float = 0.1
    weight_decay: float = 0.0


class _BareTrainer(BaseTrainer):
    """Subclass that lets tests construct a trainer without dataset setup.

    BaseTrainer.__init__ pulls model and config off the same
    arguments the production trainer uses, so we just bypass it.
    """

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return _BareConfig

    def __init__(self, model: nn.Module, config: TrainConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.lr0)
        self.scaler = None

    def get_model_family(self) -> str:
        return "test"

    def get_model_tag(self) -> str:
        return "test"

    def create_transforms(self):
        return None, None

    def create_scheduler(self, iters_per_epoch):
        return None

    def get_loss_components(self, outputs):
        return {}

    def on_forward(self, imgs, targets):
        return {}


def _make_loss_with_large_grads(
    model: nn.Module, target_norm: float = 50.0
) -> torch.Tensor:
    """Compute a loss that produces a gradient norm well above ``target_norm``.

    Without clipping the parameter-grad L2 norm should clearly exceed any
    reasonable ``grad_clip`` threshold like 0.1.
    """
    x = torch.randn(8, 4)
    y = torch.full((8,), target_norm)
    pred = model(x).squeeze(-1)
    return ((pred - y) ** 2).mean()


def _global_grad_norm(model: nn.Module) -> float:
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return torch.norm(torch.stack(norms), 2).item()


class TestBackwardAndStepGradClip:
    """``_backward_and_step`` should clip the global grad norm to ``grad_clip``."""

    def test_grad_clip_bounds_norm_no_amp(self):
        """Non-AMP path: post-step grad norm must be ≤ grad_clip + small slack."""
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        config = _BareConfig(data="dummy", grad_clip=0.1, amp=False)
        trainer = _BareTrainer(model, config)

        trainer.optimizer.zero_grad()
        loss = _make_loss_with_large_grads(model)

        # Sanity check: pre-clip gradient norm is well above the clip threshold.
        loss.backward(retain_graph=True)
        pre_clip_norm = _global_grad_norm(model)
        assert pre_clip_norm > config.grad_clip * 5, (
            f"Test setup failed: pre-clip grad norm {pre_clip_norm} is not "
            f"meaningfully larger than grad_clip {config.grad_clip}"
        )
        # Reset grads so _backward_and_step starts from a clean slate.
        trainer.optimizer.zero_grad()

        trainer._backward_and_step(loss)

        post_clip_norm = _global_grad_norm(model)
        assert post_clip_norm <= config.grad_clip * 1.05, (
            f"Gradient norm {post_clip_norm} exceeds clip threshold "
            f"{config.grad_clip}; clip_grad_norm_ may have been removed."
        )

    def test_grad_clip_zero_disables_clipping(self):
        """With grad_clip=0, ``clip_grad_norm_`` must not be called."""
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        config = _BareConfig(data="dummy", grad_clip=0.0, amp=False)
        trainer = _BareTrainer(model, config)

        trainer.optimizer.zero_grad()
        loss = _make_loss_with_large_grads(model)

        with pytest.MonkeyPatch().context() as mp:
            spy = MagicMock()
            mp.setattr("torch.nn.utils.clip_grad_norm_", spy)
            trainer._backward_and_step(loss)

        spy.assert_not_called()

    def test_grad_clip_amp_path_calls_clip(self):
        """AMP path: must unscale before clip and pass max_norm=grad_clip."""
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        config = _BareConfig(data="dummy", grad_clip=0.1, amp=True)
        trainer = _BareTrainer(model, config)
        # Inject a mock scaler — we only care about call ordering, not numerics.
        scaler = MagicMock()
        scaler.scale.return_value = MagicMock()
        trainer.scaler = scaler

        loss_mock = MagicMock(spec=torch.Tensor)

        with pytest.MonkeyPatch().context() as mp:
            clip_spy = MagicMock()
            mp.setattr("torch.nn.utils.clip_grad_norm_", clip_spy)
            trainer._backward_and_step(loss_mock)

        scaler.scale.assert_called_once_with(loss_mock)
        scaler.unscale_.assert_called_once_with(trainer.optimizer)
        clip_spy.assert_called_once()
        kwargs = clip_spy.call_args.kwargs
        assert kwargs.get("max_norm") == config.grad_clip
        scaler.step.assert_called_once_with(trainer.optimizer)
        scaler.update.assert_called_once()


class TestRTDETRConfigDefaults:
    """RTDETRConfig should ship with the post-audit fine-tune defaults."""

    def test_rtdetr_defaults_grad_clip_and_warmup(self):
        from libreyolo.models.rtdetr.config import RTDETRConfig

        config = RTDETRConfig(data="dummy.yaml")
        assert config.grad_clip == 0.1
        assert config.warmup_epochs == 1

    def test_base_config_grad_clip_disabled_by_default(self):
        config = TrainConfig(data="dummy.yaml")
        assert config.grad_clip == 0.0
