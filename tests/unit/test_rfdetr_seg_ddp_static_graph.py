"""Unit tests for RF-DETR DDP static_graph / find_unused_parameters configuration.

Bug: RFDETRTrainer._ddp_find_unused_parameters() returned True for segmentation,
so DDP was created with find_unused_parameters=True, static_graph=False.
The seg head is called from both encoder and decoder branches in one forward, so
its parameters receive gradients from two call sites. DDP's per-param hook fired
twice per step → "Expected to mark a variable ready only once" crash.

Fix: segmentation uses static_graph=True (locks reducer after iteration 1);
detection uses find_unused_parameters=True (some transformer params are unused
on certain forward passes and need dynamic graph traversal).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

rfdetr_trainer = pytest.importorskip("libreyolo.models.rfdetr.trainer")


def _make_trainer(task: str) -> rfdetr_trainer.RFDETRTrainer:
    trainer = rfdetr_trainer.RFDETRTrainer.__new__(rfdetr_trainer.RFDETRTrainer)

    class _FakeWrapper:
        pass

    wrapper = _FakeWrapper()
    wrapper.task = task
    trainer.wrapper_model = wrapper
    return trainer


def test_seg_trainer_ddp_uses_static_graph_not_find_unused():
    trainer = _make_trainer("segment")
    kwargs = trainer._ddp_kwargs()
    assert kwargs["static_graph"] is True
    assert kwargs["find_unused_parameters"] is False


def test_det_trainer_ddp_uses_find_unused_not_static_graph():
    trainer = _make_trainer("detect")
    kwargs = trainer._ddp_kwargs()
    assert kwargs["find_unused_parameters"] is True
    assert kwargs["static_graph"] is False
