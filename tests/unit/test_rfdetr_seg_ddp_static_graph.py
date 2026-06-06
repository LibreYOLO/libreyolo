"""Unit tests for RF-DETR DDP static_graph / find_unused_parameters configuration.

Bug: RFDETRTrainer._ddp_find_unused_parameters() returned True for segmentation,
so DDP was created with find_unused_parameters=True, static_graph=False.
The seg head is called from both encoder and decoder branches in one forward, so
its parameters receive gradients from two call sites. DDP's per-param hook fired
twice per step → "Expected to mark a variable ready only once" crash.

Fix: segmentation uses static_graph=True (locks reducer after iteration 1);
detection and OBB use find_unused_parameters=True (some transformer params are
unused on certain forward passes and need dynamic graph traversal).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

rfdetr_loss = pytest.importorskip("libreyolo.models.rfdetr.loss")
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


def test_obb_trainer_ddp_uses_find_unused_not_static_graph():
    trainer = _make_trainer("obb")
    kwargs = trainer._ddp_kwargs()
    assert kwargs["find_unused_parameters"] is True
    assert kwargs["static_graph"] is False


def test_mask_loss_no_match_zero_stays_connected_to_mask_head_tensors():
    criterion = rfdetr_loss.SetCriterion(
        num_classes=1,
        matcher=None,
        weight_dict={},
        focal_alpha=0.25,
        losses=["masks"],
    )

    spatial_features = torch.randn(2, 4, 8, 8, requires_grad=True)
    query_features = torch.randn(2, 5, 4, requires_grad=True)
    bias = torch.randn(1, requires_grad=True)
    outputs = {
        "pred_masks": {
            "spatial_features": spatial_features,
            "query_features": query_features,
            "bias": bias,
        }
    }
    targets = [
        {
            "labels": torch.zeros(0, dtype=torch.long),
            "boxes": torch.zeros(0, 4),
            "masks": torch.zeros(0, 8, 8, dtype=torch.bool),
        }
        for _ in range(2)
    ]
    indices = [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        for _ in targets
    ]

    losses = criterion.loss_masks(outputs, targets, indices, num_boxes=1.0)
    loss = losses["loss_mask_ce"] + losses["loss_mask_dice"]
    loss.backward()

    assert loss.ndim == 0
    assert loss.item() == 0.0
    assert spatial_features.grad is not None
    assert query_features.grad is not None
    assert bias.grad is not None
    assert spatial_features.grad.abs().sum().item() == 0.0
    assert query_features.grad.abs().sum().item() == 0.0
    assert bias.grad.abs().sum().item() == 0.0
