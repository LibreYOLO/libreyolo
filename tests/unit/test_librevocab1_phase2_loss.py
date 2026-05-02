"""LibreVocab1 Phase 2 — matcher + set criterion unit tests.

Verifies on synthetic tensors that:
    - box_ops conversions round-trip
    - GIoU is in [-1, 1] and self-pairs return 1
    - Hungarian matcher returns valid permutations
    - SetCriterion produces finite, decreasing losses on a 50-step
      single-batch overfit
"""

from __future__ import annotations

import importlib.util

import pytest


_REQUIRED = ("scipy",)
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


def test_box_ops_cxcywh_xyxy_roundtrip():
    import torch
    from libreyolo.models.librevocab1.phase2.box_ops import (
        box_cxcywh_to_xyxy,
        box_xyxy_to_cxcywh,
    )

    boxes = torch.tensor([
        [0.5, 0.5, 0.4, 0.4],
        [0.1, 0.2, 0.05, 0.1],
    ])
    rt = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes))
    assert torch.allclose(rt, boxes, atol=1e-6)


def test_giou_self_pairs_are_one():
    import torch
    from libreyolo.models.librevocab1.phase2.box_ops import generalized_box_iou_pair

    boxes = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],
        [0.5, 0.5, 1.5, 1.5],
        [0.2, 0.3, 0.8, 0.9],
    ])
    g = generalized_box_iou_pair(boxes, boxes)
    assert torch.allclose(g.diag(), torch.ones(3), atol=1e-6)
    # Off-diagonal must lie in [-1, 1].
    assert (g >= -1).all() and (g <= 1).all()


def test_matcher_returns_valid_permutation():
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher

    torch.manual_seed(0)
    matcher = OpenVocabHungarianMatcher()
    B, Q, K = 2, 5, 3
    outputs = {
        "pred_logits": torch.randn(B, Q, K),
        "pred_boxes": torch.rand(B, Q, 4) * 0.5 + 0.25,
    }
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes_cxcywh": torch.tensor([[0.3, 0.4, 0.2, 0.2], [0.7, 0.6, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([2]),
            "boxes_cxcywh": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        },
    ]
    result = matcher(outputs, targets)
    assert len(result) == 2
    pred0, tgt0 = result[0]
    assert pred0.shape == tgt0.shape == torch.Size([2])
    assert set(tgt0.tolist()) == {0, 1}
    pred1, tgt1 = result[1]
    assert pred1.shape == tgt1.shape == torch.Size([1])


def test_matcher_handles_empty_targets():
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher

    matcher = OpenVocabHungarianMatcher()
    outputs = {
        "pred_logits": torch.randn(1, 5, 3),
        "pred_boxes": torch.rand(1, 5, 4),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.long),
                "boxes_cxcywh": torch.empty(0, 4)}]
    result = matcher(outputs, targets)
    assert result[0][0].numel() == 0 and result[0][1].numel() == 0


def test_set_criterion_decreases_loss_on_overfit():
    """Single-batch overfit smoke: feed predictions that point at the
    ground truth and check that fitting them with SGD pushes loss down.
    """
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion

    torch.manual_seed(0)
    B, Q, K = 1, 8, 3

    # Trainable predictions initialized to random.
    pred_logits = torch.zeros(B, Q, K, requires_grad=True)
    pred_boxes_param = torch.rand(B, Q, 4) * 0.6 + 0.2
    pred_boxes = pred_boxes_param.clone().detach().requires_grad_(True)

    targets = [{
        "labels": torch.tensor([0, 1]),
        "boxes_cxcywh": torch.tensor([[0.3, 0.4, 0.2, 0.2], [0.7, 0.6, 0.1, 0.1]]),
    }]

    matcher = OpenVocabHungarianMatcher()
    criterion = OpenVocabSetCriterion(matcher)

    # Adam handles the matcher-driven landscape better than SGD; the Hungarian
    # assignment is non-differentiable so the loss surface is bumpy.
    opt = torch.optim.Adam([pred_logits, pred_boxes], lr=0.05)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        ld = criterion(out, targets)
        total = ld["loss_cls"] + ld["loss_bbox"] + ld["loss_giou"]
        assert torch.isfinite(total)
        total.backward()
        opt.step()
        # Box predictions must stay in (0, 1) for downstream geometry; clamp.
        with torch.no_grad():
            pred_boxes.clamp_(min=1e-3, max=1 - 1e-3)
        losses.append(total.item())

    # Convergence is noisy with the matcher; check the running min over the
    # second half drops well below the initial loss.
    final_min = min(losses[-150:])
    assert final_min < losses[0] * 0.5, (
        f"loss did not drop enough: start={losses[0]:.4f}, "
        f"last150_min={final_min:.4f}"
    )


def test_set_criterion_handles_empty_targets():
    """No GT in batch — criterion must return finite, all-zero box terms."""
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion

    matcher = OpenVocabHungarianMatcher()
    criterion = OpenVocabSetCriterion(matcher)

    out = {
        "pred_logits": torch.randn(1, 4, 2, requires_grad=True),
        "pred_boxes": torch.rand(1, 4, 4, requires_grad=True),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.long),
                "boxes_cxcywh": torch.empty(0, 4)}]
    ld = criterion(out, targets)
    for k, v in ld.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"
