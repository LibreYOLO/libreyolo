"""DAMO-YOLO training losses (QFL + DFL + GIoU) and AlignOTA label assigner.

Ports of upstream's ``damo/base_models/losses/gfocal_loss.py``,
``damo/base_models/core/bbox_calculator.py::bbox_overlaps``, and
``damo/base_models/core/ota_assigner.py``.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IoU / GIoU
# ---------------------------------------------------------------------------


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Compute IoU/GIoU between two sets of xyxy boxes.

    Mirrors upstream's signature: returns shape (m, n) when not aligned,
    (m,) when aligned. ``mode`` ∈ {"iou", "iof", "giou"}.
    """
    assert mode in ("iou", "iof", "giou"), mode
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if rows * cols == 0:
        return bboxes1.new((rows,)) if is_aligned else bboxes1.new((rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - overlap if mode in ("iou", "giou") else area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ("iou", "giou"):
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_t = union.new_tensor([eps])
    union = torch.max(union, eps_t)
    ious = overlap / union
    if mode in ("iou", "iof"):
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = torch.max(enclose_wh[..., 0] * enclose_wh[..., 1], eps_t)
    return ious - (enclose_area - union) / enclose_area


# ---------------------------------------------------------------------------
# Reduction helpers
# ---------------------------------------------------------------------------


def _weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss
    if reduction == "mean":
        return loss.sum() / avg_factor
    if reduction == "none":
        return loss
    raise ValueError(f'avg_factor cannot be combined with reduction={reduction!r}')


def _weighted(loss_fn):
    @functools.wraps(loss_fn)
    def wrapped(pred, target, weight=None, reduction="mean", avg_factor=None, **kw):
        loss = loss_fn(pred, target, **kw)
        return _weight_reduce_loss(loss, weight, reduction, avg_factor)

    return wrapped


# ---------------------------------------------------------------------------
# GIoU loss
# ---------------------------------------------------------------------------


@_weighted
def _giou_loss(pred, target, eps=1e-7):
    return 1 - bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)


class GIoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kw):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        red = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        return self.loss_weight * _giou_loss(
            pred, target, weight, eps=self.eps, reduction=red, avg_factor=avg_factor, **kw
        )


# ---------------------------------------------------------------------------
# Distribution Focal Loss
# ---------------------------------------------------------------------------


@_weighted
def _distribution_focal_loss(pred, label):
    dis_left = label.long()
    dis_right = dis_left + 1
    w_left = dis_right.float() - label
    w_right = label - dis_left.float()
    return (
        F.cross_entropy(pred, dis_left, reduction="none") * w_left
        + F.cross_entropy(pred, dis_right, reduction="none") * w_right
    )


class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        red = reduction_override if reduction_override else self.reduction
        return self.loss_weight * _distribution_focal_loss(
            pred, target, weight, reduction=red, avg_factor=avg_factor
        )


# ---------------------------------------------------------------------------
# Quality Focal Loss
# ---------------------------------------------------------------------------


@_weighted
def _quality_focal_loss(pred, target, beta: float = 2.0, use_sigmoid: bool = True):
    label, score = target  # both shape (N,)
    func = F.binary_cross_entropy_with_logits if use_sigmoid else F.binary_cross_entropy
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction="none") * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    sf_pos = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = (
        func(pred[pos, pos_label], score[pos], reduction="none") * sf_pos.abs().pow(beta)
    )
    return loss.sum(dim=1, keepdim=False)


class QualityFocalLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = True,
        beta: float = 2.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        red = reduction_override if reduction_override else self.reduction
        return self.loss_weight * _quality_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            reduction=red,
            avg_factor=avg_factor,
        )


# ---------------------------------------------------------------------------
# AlignOTA label assigner
# ---------------------------------------------------------------------------


@dataclass
class AssignResult:
    num_gts: int
    gt_inds: torch.Tensor      # (num_priors,) — 1-based gt index, 0 = unassigned
    max_overlaps: torch.Tensor # (num_priors,) — IoU of assigned gt
    labels: Optional[torch.Tensor] = None  # (num_priors,)


class AlignOTAAssigner:
    """SimOTA / AlignOTA label assignment with cls-quality weighting."""

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
    ) -> None:
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps: float = 1e-7):
        try:
            return self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps)
        except RuntimeError:
            origin = pred_scores.device
            warnings.warn("OTA OOM — falling back to CPU for this batch")
            torch.cuda.empty_cache()
            r = self._assign(
                pred_scores.cpu(), priors.cpu(), decoded_bboxes.cpu(),
                gt_bboxes.cpu().float(), gt_labels.cpu(), eps,
            )
            r.gt_inds = r.gt_inds.to(origin)
            r.max_overlaps = r.max_overlaps.to(origin)
            if r.labels is not None:
                r.labels = r.labels.to(origin)
            return r

    def _assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps):
        INF = 100_000_000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        valid_mask, is_in_boxes_and_center = self._get_in_gt_and_in_center_info(priors, gt_bboxes)
        valid_decoded = decoded_bboxes[valid_mask]
        valid_pred = pred_scores[valid_mask]
        num_valid = valid_decoded.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = (
                None if gt_labels is None
                else decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            )
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)

        gt_onehot = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_rep = valid_pred.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_rep
        cls_cost = (
            F.binary_cross_entropy(valid_pred_rep, soft_label, reduction="none")
            * scale_factor.abs().pow(2.0)
        ).sum(dim=-1)

        cost = cls_cost * self.cls_weight + iou_cost * self.iou_weight + (~is_in_boxes_and_center) * INF
        matched_pred_ious, matched_gt_inds = self._dynamic_k_matching(cost, pairwise_ious, num_gt, valid_mask)

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

    def _get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)
        rx = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        ry = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        sx = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        sy = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        l_ = rx - gt_bboxes[:, 0]
        t_ = ry - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - rx
        b_ = gt_bboxes[:, 3] - ry
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_l = gt_cx - self.center_radius * sx
        ct_t = gt_cy - self.center_radius * sy
        ct_r = gt_cx + self.center_radius * sx
        ct_b = gt_cy + self.center_radius * sy
        is_in_cts = torch.stack(
            [rx - ct_l, ry - ct_t, ct_r - rx, ct_b - ry], dim=1
        ).min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        )
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def _dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching = torch.zeros_like(cost)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching[:, gt_idx][pos_idx] = 1.0

        # priors matched to >1 gt: keep only the cheapest
        prior_match = matching.sum(1) > 1
        if prior_match.sum() > 0:
            _, argmin = torch.min(cost[prior_match, :], dim=1)
            matching[prior_match, :] *= 0.0
            matching[prior_match, argmin] = 1.0

        fg_mask = matching.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask
        matched_gt_inds = matching[fg_mask, :].argmax(1)
        matched_pred_ious = (matching * pairwise_ious).sum(1)[fg_mask]
        return matched_pred_ious, matched_gt_inds
