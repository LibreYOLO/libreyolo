"""YOLO-NAS-R (OBB) training loss.

Ported from SuperGradients (Apache-2.0) at the pinned YOLO-NAS-R PR head
``69141b55c1161d939939a270523a7eca5a645f72``:
``src/super_gradients/training/losses/yolo_nas_r_loss.py``.

Three terms, matching upstream:

- **classification**: varifocal loss against the task-aligned assigner's
  soft scores;
- **regression**: ``1 - probIoU`` on the decoded ``cxcywhr`` boxes, which is
  what supervises the centre offsets and the rotation -- upstream has no
  separate offset or angle regression term, the probabilistic IoU covers
  both;
- **distribution focal loss** on the width/height DFL bins.

Weights follow upstream's DOTA recipe
(``recipes/training_hyperparams/default_yolo_nas_r_train_params.yaml``):
``classification 2.5``, ``iou 2.0``, ``dfl 0.5``, assigner ``topk 12``.

The generic top-k / max-IoU assigner helpers are shared with the detection
loss in :mod:`libreyolo.models.yolonas.loss`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .loss import compute_max_iou_anchor, gather_topk_anchors


def check_points_inside_rboxes(points: Tensor, rboxes: Tensor) -> Tensor:
    """Mask anchors whose centre falls inside each rotated box's inradius.

    :param points: ``(L, 2)`` anchor centres in pixels.
    :param rboxes: ``(B, n, 5)`` ground-truth boxes in ``cxcywhr``.
    :return: ``(B, n, L)`` float mask, 1 where the anchor is inside.
    """
    points = points[None, None, :, :]
    x, y = points[..., 0], points[..., 1]

    cx = rboxes[..., 0, None]
    cy = rboxes[..., 1, None]
    w = rboxes[..., 2, None]
    h = rboxes[..., 3, None]
    smallest_radius_sqr = (torch.minimum(w, h) / 2) ** 2

    distance_sqr = (x - cx).pow(2) + (y - cy).pow(2)
    return (distance_sqr <= smallest_radius_sqr).type_as(rboxes)


def _covariance_matrix(w: Tensor, h: Tensor, angle: Tensor):
    """Gaussian covariance terms ``(a, b, c)`` for a rotated box."""
    a = w.pow(2) / 12
    b = h.pow(2) / 12
    cos = angle.cos()
    sin = angle.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def cxcywhr_iou(
    obb1: Tensor,
    obb2: Tensor,
    include_ciou_term: bool = False,
    eps: float = 1e-5,
) -> Tensor:
    """Probabilistic IoU between two sets of ``cxcywhr`` boxes.

    https://arxiv.org/pdf/2106.06072v1.pdf -- differentiable, unlike the exact
    polygon IoU used at postprocessing time, which is why training uses this
    and NMS uses the exact one.
    """
    x1, y1, w1, h1, a1 = (
        obb1[..., 0],
        obb1[..., 1],
        obb1[..., 2],
        obb1[..., 3],
        obb1[..., 4],
    )
    x2, y2, w2, h2, a2 = (
        obb2[..., 0],
        obb2[..., 1],
        obb2[..., 2],
        obb2[..., 3],
        obb2[..., 4],
    )

    a1, b1, c1 = _covariance_matrix(w1, h1, a1)
    a2, b2, c2 = _covariance_matrix(w2, h2, a2)

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * (
                (a1 * b1 - c1.pow(2)).clamp_min(0) * (a2 * b2 - c2.pow(2)).clamp_min(0)
                + eps
            ).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 9.0)
    # expm1 is the numerically stable form; the exp fallback keeps the
    # expression traceable for ONNX, matching upstream.
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        hd = (1.0 - (-bd).exp().clamp_min(eps)).sqrt()
    else:
        hd = torch.sqrt(-torch.expm1(-bd))

    iou = 1 - hd

    if include_ciou_term:
        v = (4 / math.pi**2) * (
            (w2 / (h2 + 1e-6)).atan() - (w1 / (h1 + 1e-6)).atan()
        ).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha

    return torch.masked_fill(iou, ~torch.isfinite(iou), 0)


def pairwise_cxcywhr_iou(obb1: Tensor, obb2: Tensor, eps: float = 1e-7) -> Tensor:
    """``(..., N, M)`` probabilistic IoU between every pair of boxes."""
    return cxcywhr_iou(
        obb1[..., :, None, :], obb2[..., None, :, :], include_ciou_term=False, eps=eps
    )


@dataclasses.dataclass
class YOLONASOBBAssignment:
    """Per-anchor assignment produced by :class:`YOLONASOBBAssigner`."""

    assigned_labels: Tensor  # (B, L)
    assigned_rboxes: Tensor  # (B, L, 5)
    assigned_scores: Tensor  # (B, L, C)
    assigned_gt_index: Tensor  # (B, L)


class YOLONASOBBAssigner(nn.Module):
    """Task-aligned assigner on rotated boxes."""

    def __init__(
        self, topk: int = 12, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9
    ):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_rboxes: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_rboxes: Tensor,
        bg_index: int,
        pad_gt_mask: Optional[Tensor] = None,
    ) -> YOLONASOBBAssignment:
        """
        :param pred_scores: ``(B, L, C)`` sigmoid class scores.
        :param pred_rboxes: ``(B, L, 5)`` decoded predictions, ``cxcywhr``.
        :param anchor_points: ``(L, 2)`` anchor centres, already multiplied by stride.
        :param gt_labels: ``(B, n, 1)`` int64 class indices.
        :param gt_rboxes: ``(B, n, 5)`` ground truth, ``cxcywhr``.
        :param bg_index: background class index (``num_classes``).
        """
        batch_size, num_anchors, num_classes = pred_scores.shape
        num_max_boxes = gt_rboxes.shape[1]
        device = pred_scores.device

        if num_max_boxes == 0:
            return YOLONASOBBAssignment(
                assigned_labels=torch.full(
                    [batch_size, num_anchors], bg_index, dtype=torch.long, device=device
                ),
                assigned_rboxes=torch.zeros(
                    [batch_size, num_anchors, 5], device=device
                ),
                assigned_scores=torch.zeros(
                    [batch_size, num_anchors, num_classes], device=device
                ),
                assigned_gt_index=torch.zeros(
                    [batch_size, num_anchors], dtype=torch.long, device=device
                ),
            )

        ious = pairwise_cxcywhr_iou(gt_rboxes, pred_rboxes)  # [B, n, L]

        scores_t = torch.permute(pred_scores, [0, 2, 1])  # [B, C, L]
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=device
        ).unsqueeze(-1)
        gt_labels_ind = torch.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1
        )
        bbox_cls_scores = scores_t[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)
        is_in_gts = check_points_inside_rboxes(anchor_points, gt_rboxes)
        # torch.topk raises when k exceeds the number of anchors, which real
        # feature maps never do but small synthetic inputs can.
        topk = min(self.topk, num_anchors)
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts, topk, topk_mask=pad_gt_mask
        )

        mask_positive = is_in_topk * is_in_gts
        if pad_gt_mask is not None:
            mask_positive = mask_positive * pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1]
            )
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        flat_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(
            gt_labels.flatten(), index=flat_gt_index.flatten(), dim=0
        ).reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, bg_index),
        )

        assigned_rboxes = gt_rboxes.reshape([-1, 5])[flat_gt_index.flatten(), :]
        assigned_rboxes = assigned_rboxes.reshape([batch_size, num_anchors, 5])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        keep = [i for i in range(num_classes + 1) if i != bg_index]
        assigned_scores = torch.index_select(
            assigned_scores,
            index=torch.tensor(keep, device=device, dtype=torch.long),
            dim=-1,
        )

        alignment_metrics = alignment_metrics * mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = (
            alignment_metrics
            / (max_metrics_per_instance + self.eps)
            * max_ious_per_instance
        )
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return YOLONASOBBAssignment(
            assigned_labels=assigned_labels,
            assigned_rboxes=assigned_rboxes,
            assigned_scores=assigned_scores,
            assigned_gt_index=assigned_gt_index,
        )


class YOLONASOBBLoss(nn.Module):
    """Varifocal classification + probabilistic-IoU + width/height DFL."""

    def __init__(
        self,
        num_classes: int,
        classification_loss_weight: float = 2.5,
        iou_loss_weight: float = 2.0,
        dfl_loss_weight: float = 0.5,
        assigner_topk: int = 12,
        assigner_alpha: float = 1.0,
        assigner_beta: float = 6.0,
        use_varifocal_loss: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classification_loss_weight = classification_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.assigner = YOLONASOBBAssigner(
            topk=assigner_topk, alpha=assigner_alpha, beta=assigner_beta
        )

    @property
    def component_names(self) -> List[str]:
        return ["loss_cls", "loss_iou", "loss_dfl", "loss"]

    def forward(self, outputs, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param outputs: the head's eager output -- ``((boxes, scores), raw)``
            or just the ``raw`` logits dict.
        :param targets: padded ``(B, max_labels, 6)`` rows of
            ``[class, cx, cy, w, h, angle]`` in input-canvas pixels. Rows with
            non-positive width or height are padding.
        """
        decoded, raw = self._split_outputs(outputs)
        pred_boxes, pred_scores = decoded

        batch_size = pred_scores.shape[0]
        num_classes = pred_scores.shape[2]
        anchor_points = raw["anchor_points"] * raw["strides"]

        gt_boxes, gt_labels = self._unpack_targets(targets, batch_size)

        cls_loss_sum = pred_scores.new_zeros(())
        iou_loss_sum = pred_scores.new_zeros(())
        dfl_loss_sum = pred_scores.new_zeros(())
        assigned_scores_sum = pred_scores.new_zeros(())

        for i in range(batch_size):
            with torch.no_grad():
                assignment = self.assigner(
                    pred_scores=pred_scores[i].unsqueeze(0).detach(),
                    pred_rboxes=pred_boxes[i].unsqueeze(0).detach(),
                    anchor_points=anchor_points,
                    gt_labels=gt_labels[i].unsqueeze(0),
                    gt_rboxes=gt_boxes[i].unsqueeze(0),
                    bg_index=num_classes,
                )

            score_logits = raw["score_logits"][i : i + 1].float()
            if self.use_varifocal_loss:
                one_hot_label = F.one_hot(assignment.assigned_labels, num_classes + 1)[
                    ..., :-1
                ]
                cls_loss = self._varifocal_loss(
                    score_logits, assignment.assigned_scores.float(), one_hot_label
                )
            else:
                cls_loss = self._focal_loss(
                    score_logits, assignment.assigned_scores.float()
                )

            loss_iou, loss_dfl = self._rbox_loss(
                pred_dist=raw["size_dist"][i : i + 1],
                pred_bboxes=pred_boxes[i : i + 1],
                assignment=assignment,
                strides=raw["strides"],
                reg_max=int(raw["reg_max"]),
                bg_class_index=num_classes,
            )

            cls_loss_sum = cls_loss_sum + cls_loss
            iou_loss_sum = iou_loss_sum + loss_iou
            dfl_loss_sum = dfl_loss_sum + loss_dfl
            assigned_scores_sum = assigned_scores_sum + assignment.assigned_scores.sum()

        normalizer = assigned_scores_sum.detach().float().clamp_min(1.0)
        cls_loss = cls_loss_sum * self.classification_loss_weight / normalizer
        iou_loss = iou_loss_sum * self.iou_loss_weight / normalizer
        dfl_loss = dfl_loss_sum * self.dfl_loss_weight / normalizer
        loss = cls_loss + iou_loss + dfl_loss

        log_losses = torch.stack(
            [cls_loss.detach(), iou_loss.detach(), dfl_loss.detach(), loss.detach()]
        )
        return loss, log_losses

    @staticmethod
    def _split_outputs(outputs):
        if isinstance(outputs, tuple) and len(outputs) == 2:
            decoded, raw = outputs
            if isinstance(decoded, tuple) and isinstance(raw, dict):
                return decoded, raw
        raise TypeError(
            "YOLO-NAS OBB loss expects the head's eager output "
            "((boxes, scores), raw_logits_dict), got "
            f"{type(outputs)!r}"
        )

    @staticmethod
    def _unpack_targets(
        targets: Tensor, batch_size: int
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Split padded ``(B, max_labels, 6)`` rows into per-image tensors."""
        if targets.ndim != 3 or targets.shape[-1] != 6:
            raise ValueError(
                "YOLO-NAS OBB training expects targets shaped "
                f"(B, max_labels, 6), got {tuple(targets.shape)}"
            )
        boxes: List[Tensor] = []
        labels: List[Tensor] = []
        for i in range(batch_size):
            rows = targets[i]
            valid = (rows[:, 3] > 0) & (rows[:, 4] > 0)
            rows = rows[valid]
            boxes.append(rows[:, 1:6].float())
            labels.append(rows[:, 0].long().unsqueeze(-1))
        return boxes, labels

    def _rbox_loss(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        assignment: YOLONASOBBAssignment,
        strides: Tensor,
        reg_max: int,
        bg_class_index: int,
    ) -> Tuple[Tensor, Tensor]:
        mask_positive = assignment.assigned_labels != bg_class_index  # [B, L]
        bbox_weight = assignment.assigned_scores.sum(-1) * mask_positive

        iou = cxcywhr_iou(
            pred_bboxes, assignment.assigned_rboxes, include_ciou_term=False
        )
        loss_iou = ((1 - iou) * bbox_weight).sum(dtype=torch.float32)

        # DFL targets are the assigned box's width/height in stride units.
        wh_targets = (assignment.assigned_rboxes[..., 2:4] / strides).clamp(
            0, reg_max - 0.01
        )
        pred_dist = pred_dist.reshape([pred_bboxes.shape[0], -1, 2, reg_max + 1])
        loss_dfl = self._df_loss(pred_dist, wh_targets)
        loss_dfl = (loss_dfl.squeeze(-1) * bbox_weight).sum(dtype=torch.float32)
        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist: Tensor, target_dist: Tensor) -> Tensor:
        target_left = target_dist.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target_dist
        weight_right = 1 - weight_left

        pred_dist = torch.moveaxis(pred_dist, -1, 1)
        loss_left = (
            F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    @staticmethod
    def _varifocal_loss(
        pred_logits: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0
    ) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = weight * F.binary_cross_entropy_with_logits(
            pred_logits, gt_score, reduction="none"
        )
        return loss.sum(dtype=torch.float32)

    @staticmethod
    def _focal_loss(
        pred_logits: Tensor, label: Tensor, alpha: float = 0.25, gamma: float = 2.0
    ) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = torch.abs(pred_score - label).pow(gamma)
        if alpha > 0:
            weight = weight * (alpha * label + (1 - alpha) * (1 - label))
        loss = weight * F.binary_cross_entropy_with_logits(
            pred_logits, label, reduction="none"
        )
        return loss.sum(dtype=torch.float32)


__all__ = [
    "YOLONASOBBAssigner",
    "YOLONASOBBAssignment",
    "YOLONASOBBLoss",
    "check_points_inside_rboxes",
    "cxcywhr_iou",
    "pairwise_cxcywhr_iou",
]
