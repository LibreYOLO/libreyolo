"""Parity and sync-count tests for the RT-DETR single-drain criterion (issue #763).

The RT-DETR ``SetCriterion`` used to call its monolithic matcher once per
output level (main + 5 decoder aux + 1 encoder level appended to
``aux_outputs``), each call ending in an isolated ``.cpu()`` pipeline drain,
plus one ``.item()`` drain from the float ``num_boxes`` normalizer. Measured
per training step at the pre-change base (dev @ d3869b27) on the synthetic
step below:

    rtdetr v1:  7 x ``.cpu()`` (matcher) + 1 x ``.item()`` (normalizer)
    rtdetrv2:   8 x ``.cpu()``           + 2 x ``.item()``

Parity scope: bitwise on CPU (asserted below). On CUDA the tensor
``_normalizer`` division selects a different kernel than the old
divide-by-Python-float and differs by up to ~3e-5 in ``total_loss``; this is
the same acknowledged deviation the shipped rfdetr/yolo9 tensor normalizers
carry (PR #765). The matcher pipeline itself is bitwise on CUDA.

The restructure splits the matcher into ``compute_cost_matrix`` (device work)
and ``solve`` (host LSAP) and pipelines the levels at depth 2, so each
transfer has the next level's cost enqueued behind it, and the normalizer
stays a 0-dim tensor. The depth-2 theoretical floor is one ``.cpu()``
transfer per matched level and zero ``.item()``; the tests below pin the
criterion to that floor and prove the losses bitwise-identical against a
reference implementation of the old per-level control flow.
"""

from __future__ import annotations

import copy

import pytest
import torch

from libreyolo.models.rtdetr.loss import (
    HungarianMatcher,
    RTDETRLoss,
    SetCriterion,
)
from libreyolo.models.rtdetrv2.loss import RTDETRv2Loss
from libreyolo.training.distributed import all_reduce_avg_scalar

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is an rtdetr training dep
    linear_sum_assignment = None

pytestmark = pytest.mark.unit

BS = 2
NUM_QUERIES = 30
NUM_CLASSES = 5
NUM_AUX = 6  # rtdetr v1: 5 decoder aux levels + 1 encoder level
NUM_DN_AUX = 6
DN_NUM_GROUP = 4

TARGET_COUNTS = [
    (4, 7),
    (0, 0),
    (0, 5),
    (1, 0),
    (1, 1),
    (40, 25),
]
COUNT_IDS = ["normal", "all-empty", "half-empty", "single-and-empty", "single", "many"]


# ---------------------------------------------------------------------------
# Synthetic inputs mirroring a training step
# ---------------------------------------------------------------------------


def _outputs(generator, num_queries=NUM_QUERIES):
    return {
        "pred_logits": torch.randn(BS, num_queries, NUM_CLASSES, generator=generator) * 3,
        "pred_boxes": torch.rand(BS, num_queries, 4, generator=generator).clamp(1e-3, 1.0),
    }


def _targets(generator, counts):
    return [
        {
            "labels": torch.randint(0, NUM_CLASSES, (n,), generator=generator),
            "boxes": torch.rand(n, 4, generator=generator).clamp(1e-3, 1.0),
        }
        for n in counts
    ]


def _step_outputs(generator, counts, v2=False):
    """Full training-step output dict: main + aux + denoising (+ v2 enc)."""
    out = _outputs(generator)
    out["aux_outputs"] = [_outputs(generator) for _ in range(NUM_AUX)]
    dn_queries = max(max(counts), 1) * 2 * DN_NUM_GROUP
    out["dn_aux_outputs"] = [
        _outputs(generator, dn_queries) for _ in range(NUM_DN_AUX)
    ]
    out["dn_meta"] = {
        "dn_positive_idx": [
            torch.arange(n * DN_NUM_GROUP, dtype=torch.int64) for n in counts
        ],
        "dn_num_group": DN_NUM_GROUP,
        "dn_num_split": [dn_queries, NUM_QUERIES],
    }
    if v2:
        out["enc_aux_outputs"] = [_outputs(generator)]
        out["enc_meta"] = {"class_agnostic": False}
    return out


# ---------------------------------------------------------------------------
# Reference implementations of the old (pre-#763) control flow
# ---------------------------------------------------------------------------


@torch.no_grad()
def _reference_matcher_indices(matcher, outputs, targets):
    """The old monolithic ``HungarianMatcher.forward``: full cost build,
    one ``.cpu()``, then per-image LSAP, all in a single call."""
    bs, num_queries = outputs["pred_logits"].shape[:2]

    if matcher.use_focal_loss:
        out_prob = torch.nn.functional.sigmoid(outputs["pred_logits"].flatten(0, 1))
    else:
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

    out_bbox = outputs["pred_boxes"].flatten(0, 1)

    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])

    if len(tgt_ids) == 0:
        return [
            (
                torch.as_tensor([], dtype=torch.int64),
                torch.as_tensor([], dtype=torch.int64),
            )
            for _ in range(bs)
        ]

    if matcher.use_focal_loss:
        out_prob = out_prob[:, tgt_ids]
        neg_cost_class = (
            (1 - matcher.alpha)
            * (out_prob**matcher.gamma)
            * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = (
            matcher.alpha
            * ((1 - out_prob) ** matcher.gamma)
            * (-(out_prob + 1e-8).log())
        )
        cost_class = pos_cost_class - neg_cost_class
    else:
        cost_class = -out_prob[:, tgt_ids]

    cost_bbox = (out_bbox[:, None, :] - tgt_bbox[None, :, :]).abs().sum(-1)

    from libreyolo.models.rtdetr.box_ops import (
        box_cxcywh_to_xyxy,
        generalized_box_iou,
    )

    cost_giou = -generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
    )

    C = (
        matcher.cost_bbox * cost_bbox
        + matcher.cost_class * cost_class
        + matcher.cost_giou * cost_giou
    )
    C = C.view(bs, num_queries, -1).cpu()

    sizes = [len(v["boxes"]) for v in targets]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return [
        (
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        )
        for i, j in indices
    ]


class _ReferenceSetCriterion(SetCriterion):
    """Old ``SetCriterion.forward``: sequential per-level matcher calls
    (one full drain each) and the float ``num_boxes`` normalizer."""

    def _normalizer(self, count, *, device):
        if self.distributed_normalize:
            return all_reduce_avg_scalar(count, device=device)
        return float(max(float(count), 1.0))

    def forward(self, outputs, targets):
        outputs = self._cast_to_float32(outputs)

        outputs_without_aux = {
            k: v for k, v in outputs.items() if "aux" not in k and k != "dn_meta"
        }

        indices = _reference_matcher_indices(self.matcher, outputs_without_aux, targets)

        num_boxes = self._normalizer(
            sum(len(t["labels"]) for t in targets),
            device=next(iter(outputs.values())).device,
        )

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {
                k: l_dict[k] * self.weight_dict.get(k, 1.0)
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = _reference_matcher_indices(self.matcher, aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {
                        k: l_dict[k] * self.weight_dict.get(k, 1.0)
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes_dn = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes_dn
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict.get(k, 1.0)
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        losses["total_loss"] = total_loss

        return losses


class _ReferenceRTDETRCriterionv2(_ReferenceSetCriterion):
    """Old v2 flow: reference parent forward plus the per-level
    ``enc_aux_outputs`` matcher calls, all through the reference matcher."""

    def forward(self, outputs, targets):
        losses = _ReferenceSetCriterion.forward(self, outputs, targets)

        if "enc_aux_outputs" in outputs:
            enc_losses = self._compute_enc_aux_losses(outputs, targets)
            if enc_losses:
                losses.update(enc_losses)
                losses["total_loss"] = sum(
                    v
                    for k, v in losses.items()
                    if k != "total_loss" and isinstance(v, torch.Tensor)
                )
        return losses

    def _compute_enc_aux_losses(self, outputs, targets):
        assert "enc_meta" in outputs, "enc_aux_outputs requires enc_meta"

        device = outputs["pred_logits"].device
        num_boxes = self._normalizer(
            sum(len(t["labels"]) for t in targets), device=device
        )

        class_agnostic = bool(outputs["enc_meta"].get("class_agnostic", False))
        if class_agnostic:
            orig_num_classes = self.num_classes
            self.num_classes = 1
            enc_targets = copy.deepcopy(targets)
            for t in enc_targets:
                t["labels"] = torch.zeros_like(t["labels"])
        else:
            enc_targets = targets

        losses = {}
        try:
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                aux_outputs = {
                    k: v.float() if isinstance(v, torch.Tensor) else v
                    for k, v in aux_outputs.items()
                }
                indices = _reference_matcher_indices(
                    self.matcher, aux_outputs, enc_targets
                )
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices, num_boxes
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict.get(k, 1.0)
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        finally:
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses


def _reference_criterion(criterion, cls=_ReferenceSetCriterion):
    """Reference twin sharing the (stateless) matcher and config."""
    return cls(
        matcher=criterion.matcher,
        weight_dict=criterion.weight_dict,
        losses=criterion.losses,
        alpha=criterion.alpha,
        gamma=criterion.gamma,
        num_classes=criterion.num_classes,
        distributed_normalize=criterion.distributed_normalize,
    )


# ---------------------------------------------------------------------------
# Assignment parity: split matcher == old monolithic matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_focal_loss", [True, False], ids=["focal", "softmax"])
@pytest.mark.parametrize("counts", TARGET_COUNTS, ids=COUNT_IDS)
@pytest.mark.parametrize("seed", [0, 1, 763])
def test_split_matcher_assignments_match_reference(seed, counts, use_focal_loss):
    matcher = HungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=use_focal_loss,
    )
    generator = torch.Generator().manual_seed(seed)
    outputs = _outputs(generator)
    targets = _targets(generator, counts)

    new = matcher(outputs, targets)
    ref = _reference_matcher_indices(matcher, outputs, targets)

    assert len(new) == len(ref) == BS
    for (pn, tn), (pr, tr), n in zip(new, ref, counts):
        assert torch.equal(pn, pr)
        assert torch.equal(tn, tr)
        assert len(pn) == len(tn) == min(n, NUM_QUERIES)


def test_compute_cost_matrix_returns_none_without_targets():
    matcher = HungarianMatcher(
        {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}, use_focal_loss=True
    )
    generator = torch.Generator().manual_seed(0)
    outputs = _outputs(generator)
    targets = _targets(generator, (0, 0))

    assert matcher.compute_cost_matrix(outputs, targets) is None
    indices = matcher.solve(None, targets)
    assert len(indices) == BS
    for pi, ti in indices:
        assert pi.dtype == ti.dtype == torch.int64
        assert len(pi) == len(ti) == 0


# ---------------------------------------------------------------------------
# Loss parity: pipelined criterion == old sequential control flow, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_vfl", [True, False], ids=["vfl", "focal"])
@pytest.mark.parametrize("counts", TARGET_COUNTS, ids=COUNT_IDS)
@pytest.mark.parametrize("seed", [0, 763])
def test_criterion_losses_bitwise_match_reference(seed, counts, use_vfl):
    criterion = RTDETRLoss(num_classes=NUM_CLASSES, use_vfl=use_vfl)
    reference = _reference_criterion(criterion)
    criterion.train()
    reference.train()

    generator = torch.Generator().manual_seed(seed)
    outputs = _step_outputs(generator, counts)
    targets = _targets(generator, counts)

    new_losses = criterion(outputs, targets)
    ref_losses = reference(outputs, targets)

    assert new_losses.keys() == ref_losses.keys()
    for key in ref_losses:
        assert torch.equal(new_losses[key], torch.as_tensor(ref_losses[key])), key
    assert new_losses["total_loss"].ndim == 0


@pytest.mark.parametrize("class_agnostic", [False, True], ids=["classed", "agnostic"])
@pytest.mark.parametrize("counts", TARGET_COUNTS, ids=COUNT_IDS)
@pytest.mark.parametrize("seed", [0, 763])
def test_rtdetrv2_criterion_losses_bitwise_match_reference(seed, counts, class_agnostic):
    criterion = RTDETRv2Loss(num_classes=NUM_CLASSES)
    reference = _reference_criterion(criterion, cls=_ReferenceRTDETRCriterionv2)
    criterion.train()
    reference.train()

    generator = torch.Generator().manual_seed(seed)
    outputs = _step_outputs(generator, counts, v2=True)
    outputs["enc_meta"]["class_agnostic"] = class_agnostic
    if class_agnostic:
        # The agnostic enc_score_head emits a single foreground logit.
        for enc_out in outputs["enc_aux_outputs"]:
            enc_out["pred_logits"] = enc_out["pred_logits"][..., :1]
    targets = _targets(generator, counts)

    new_losses = criterion(outputs, targets)
    ref_losses = reference(outputs, targets)

    assert new_losses.keys() == ref_losses.keys()
    assert any(k.endswith("_enc_0") for k in new_losses)
    for key in ref_losses:
        assert torch.equal(new_losses[key], torch.as_tensor(ref_losses[key])), key


# ---------------------------------------------------------------------------
# Sync-count regression: at the depth-2 floor
# ---------------------------------------------------------------------------


class _TransferCounter:
    def __init__(self, monkeypatch):
        self.cpu = 0
        self.item = 0
        self.tolist = 0
        orig_cpu = torch.Tensor.cpu
        orig_item = torch.Tensor.item
        orig_tolist = torch.Tensor.tolist

        def counting_cpu(t, *a, **k):
            self.cpu += 1
            return orig_cpu(t, *a, **k)

        def counting_item(t):
            self.item += 1
            return orig_item(t)

        def counting_tolist(t):
            self.tolist += 1
            return orig_tolist(t)

        monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
        monkeypatch.setattr(torch.Tensor, "item", counting_item)
        monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)


def test_v1_criterion_step_transfers_at_depth2_floor(monkeypatch):
    """Depth-2 floor: one ``.cpu()`` per matched level (1 main + NUM_AUX aux),
    zero ``.item()``/``.tolist()``. Before the restructure the same step did
    7 ``.cpu()`` drains with nothing queued behind them plus 1 ``.item()``."""
    criterion = RTDETRLoss(num_classes=NUM_CLASSES)
    criterion.train()
    generator = torch.Generator().manual_seed(763)
    outputs = _step_outputs(generator, (4, 7))
    targets = _targets(generator, (4, 7))

    counter = _TransferCounter(monkeypatch)
    criterion(outputs, targets)

    assert counter.cpu <= 1 + NUM_AUX
    assert counter.item == 0
    assert counter.tolist == 0


def test_rtdetrv2_criterion_step_transfers_at_depth2_floor(monkeypatch):
    """v2 floor: parent levels (1 + NUM_AUX) plus one transfer per
    ``enc_aux_outputs`` level. Before: 8 ``.cpu()`` + 2 ``.item()``."""
    criterion = RTDETRv2Loss(num_classes=NUM_CLASSES)
    criterion.train()
    generator = torch.Generator().manual_seed(763)
    outputs = _step_outputs(generator, (4, 7), v2=True)
    targets = _targets(generator, (4, 7))

    counter = _TransferCounter(monkeypatch)
    criterion(outputs, targets)

    assert counter.cpu <= 1 + NUM_AUX + len(outputs["enc_aux_outputs"])
    assert counter.item == 0
    assert counter.tolist == 0


def test_empty_targets_do_no_transfers(monkeypatch):
    criterion = RTDETRLoss(num_classes=NUM_CLASSES)
    criterion.train()
    generator = torch.Generator().manual_seed(763)
    outputs = _step_outputs(generator, (0, 0))
    targets = _targets(generator, (0, 0))

    counter = _TransferCounter(monkeypatch)
    criterion(outputs, targets)

    assert counter.cpu == 0
    assert counter.item == 0


# ---------------------------------------------------------------------------
# Pipeline shape: depth 2, never more than two cost matrices in flight
# ---------------------------------------------------------------------------


def test_criterion_pipelines_levels_at_depth_two(monkeypatch):
    """The per-level loop must enqueue level i+1's cost before solving level
    i, and never run more than one compute ahead of the solves."""
    criterion = RTDETRLoss(num_classes=NUM_CLASSES)
    criterion.train()
    generator = torch.Generator().manual_seed(763)
    outputs = _step_outputs(generator, (4, 7))
    targets = _targets(generator, (4, 7))

    events = []
    matcher = criterion.matcher
    orig_compute = type(matcher).compute_cost_matrix
    orig_solve = type(matcher).solve

    def compute(self, *a, **k):
        events.append("compute")
        return orig_compute(self, *a, **k)

    def solve(self, *a, **k):
        events.append("solve")
        return orig_solve(self, *a, **k)

    monkeypatch.setattr(type(matcher), "compute_cost_matrix", compute)
    monkeypatch.setattr(type(matcher), "solve", solve)

    criterion(outputs, targets)

    num_levels = 1 + NUM_AUX
    assert events.count("compute") == num_levels
    assert events.count("solve") == num_levels
    in_flight = 0
    for event in events:
        in_flight += 1 if event == "compute" else -1
        assert 0 <= in_flight <= 2
