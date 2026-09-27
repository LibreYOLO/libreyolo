"""Parity tests for the RF-DETR training-speed changes.

Covers: broadcast L1 cost == cdist cost (bitwise), split
compute_cost_matrix/solve == the one-call forward, the single-sync
SetCriterion.forward == a per-level matcher reference, tensor num_boxes ==
float num_boxes, and the fused-AdamW construction fallback.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from libreyolo.models.rfdetr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from libreyolo.models.rfdetr.loss import SetCriterion
from libreyolo.models.rfdetr.matcher import HungarianMatcher
from libreyolo.models.rfdetr.trainer import RFDETRTrainer

pytestmark = pytest.mark.unit

GROUP_DETR = 3
NUM_QUERIES = 30 * GROUP_DETR
NUM_CLASSES = 7


def _matcher():
    return HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25)


def _outputs(bs, generator, num_queries=NUM_QUERIES):
    return {
        "pred_logits": torch.randn(bs, num_queries, NUM_CLASSES, generator=generator) * 3,
        "pred_boxes": torch.rand(bs, num_queries, 4, generator=generator).clamp(1e-3, 1.0),
    }


def _targets(bs, generator, counts=None):
    targets = []
    for i in range(bs):
        n = counts[i] if counts is not None else int(torch.randint(1, 9, (1,), generator=generator))
        targets.append(
            {
                "labels": torch.randint(0, NUM_CLASSES, (n,), generator=generator),
                "boxes": torch.rand(n, 4, generator=generator).clamp(1e-3, 1.0),
            }
        )
    return targets


def _reference_cost(matcher, outputs, targets):
    """The pre-change cost formula, with torch.cdist for the L1 term."""
    bs, num_queries = outputs["pred_logits"].shape[:2]
    flat_logits = outputs["pred_logits"].flatten(0, 1)
    out_prob = flat_logits.sigmoid()
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])
    giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    alpha, gamma = 0.25, 2.0
    neg = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_logits))
    pos = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_logits))
    cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    cost = (
        matcher.cost_bbox * cost_bbox
        + matcher.cost_class * cost_class
        + matcher.cost_giou * (-giou)
    )
    return cost.view(bs, num_queries, -1).float()


def test_broadcast_l1_cost_is_bitwise_identical_to_cdist():
    generator = torch.Generator().manual_seed(7)
    matcher = _matcher()
    for _ in range(5):
        outputs = _outputs(2, generator)
        targets = _targets(2, generator)
        got = matcher.compute_cost_matrix(outputs, targets)
        ref = _reference_cost(matcher, outputs, targets)
        assert torch.equal(got, ref)


def test_forward_equals_split_cost_and_solve():
    generator = torch.Generator().manual_seed(11)
    matcher = _matcher()
    outputs = _outputs(3, generator)
    targets = _targets(3, generator)
    via_forward = matcher(outputs, targets, group_detr=GROUP_DETR)
    via_split = matcher.solve(
        matcher.compute_cost_matrix(outputs, targets).cpu(),
        targets,
        group_detr=GROUP_DETR,
    )
    for (fi, fj), (si, sj) in zip(via_forward, via_split):
        assert torch.equal(fi, si)
        assert torch.equal(fj, sj)


def _criterion():
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
    return SetCriterion(
        NUM_CLASSES,
        matcher=_matcher(),
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        group_detr=GROUP_DETR,
    )


def _criterion_outputs(bs, generator):
    outputs = _outputs(bs, generator)
    outputs["aux_outputs"] = [_outputs(bs, generator) for _ in range(2)]
    outputs["enc_outputs"] = _outputs(bs, generator)
    return outputs


def _reference_criterion_losses(criterion, outputs, targets):
    """Per-level matcher calls, float num_boxes: the pre-change control flow."""
    group_detr = criterion.group_detr if criterion.training else 1
    outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
    indices = criterion.matcher(outputs_without_aux, targets, group_detr=group_detr)
    num_boxes = float(
        max(1, sum(len(t["labels"]) for t in targets) * (1 if criterion.sum_group_losses else group_detr))
    )
    losses = {}
    for loss in criterion.losses:
        losses.update(criterion.get_loss(loss, outputs, targets, indices, num_boxes))
    for i, aux_outputs in enumerate(outputs["aux_outputs"]):
        indices = criterion.matcher(aux_outputs, targets, group_detr=group_detr)
        for loss in criterion.losses:
            kwargs = {"log": False} if loss == "labels" else {}
            l_dict = criterion.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
            losses.update({k + f"_{i}": v for k, v in l_dict.items()})
    enc_outputs = outputs["enc_outputs"]
    indices = criterion.matcher(enc_outputs, targets, group_detr=group_detr)
    for loss in criterion.losses:
        kwargs = {"log": False} if loss == "labels" else {}
        l_dict = criterion.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
        losses.update({k + "_enc": v for k, v in l_dict.items()})
    return losses


def test_single_sync_criterion_matches_per_level_reference():
    generator = torch.Generator().manual_seed(13)
    criterion = _criterion()
    criterion.train()
    outputs = _criterion_outputs(2, generator)
    targets = _targets(2, generator)
    got = criterion(outputs, targets)
    ref = _reference_criterion_losses(criterion, outputs, targets)
    assert set(got.keys()) == set(ref.keys())
    for key in ref:
        torch.testing.assert_close(got[key], ref[key], rtol=0, atol=0, msg=key)


def test_criterion_eval_path_matches_reference_group1():
    generator = torch.Generator().manual_seed(17)
    criterion = _criterion()
    criterion.eval()
    outputs = _criterion_outputs(2, generator)
    targets = _targets(2, generator)
    got = criterion(outputs, targets)
    ref = _reference_criterion_losses(criterion, outputs, targets)
    for key in ref:
        torch.testing.assert_close(got[key], ref[key], rtol=0, atol=0, msg=key)


def test_num_boxes_is_tensor_and_losses_stay_scalar():
    generator = torch.Generator().manual_seed(19)
    criterion = _criterion()
    criterion.train()
    outputs = _criterion_outputs(1, generator)
    targets = _targets(1, generator)
    num_boxes = criterion._box_count_normalizer(outputs, targets, GROUP_DETR)
    assert isinstance(num_boxes, torch.Tensor)
    assert num_boxes.dim() == 0
    losses = criterion(outputs, targets)
    total = sum(losses[k] * criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    assert total.dim() == 0  # backward() must see a scalar


def test_adamw_helper_falls_back_when_fused_unsupported(monkeypatch):
    # The helper now routes through libreyolo.training.optim.build_optimizer,
    # whose device gate skips fused off-CUDA; force the gate open so the
    # construction-time fallback path is exercised on CPU params.
    import libreyolo.training.optim as optim_mod

    param = torch.nn.Parameter(torch.randn(3))
    real_adamw = torch.optim.AdamW
    calls = []

    class Picky(real_adamw):
        def __init__(self, params, **kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("fused"):
                raise RuntimeError("fused not supported here")
            super().__init__(params, **kwargs)

    monkeypatch.setattr(optim_mod, "_all_params_cuda", lambda groups: True)
    monkeypatch.setattr(torch.optim, "AdamW", Picky)
    opt = RFDETRTrainer._adamw([{"params": [param]}], lr=1e-3, betas=(0.9, 0.999))
    assert isinstance(opt, real_adamw)
    assert calls[0].get("fused") is True
    assert "fused" not in calls[1]


def test_adamw_helper_keeps_cpu_construction_stock():
    # Non-CUDA params must never be handed fused=True, even though recent
    # torch would silently accept them (issue #763 portability guarantee).
    param = torch.nn.Parameter(torch.randn(3))
    opt = RFDETRTrainer._adamw([{"params": [param]}], lr=1e-3, betas=(0.9, 0.999))
    assert not any(group.get("fused") for group in opt.param_groups)


def test_adamw_helper_constructs_and_steps():
    param = torch.nn.Parameter(torch.randn(4))
    opt = RFDETRTrainer._adamw([{"params": [param]}], lr=1e-3, betas=(0.9, 0.999))
    (param.sum() ** 2).backward()
    opt.step()
    assert param.grad is not None
