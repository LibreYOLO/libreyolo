"""DAMO-YOLO training losses (QFL + DFL + GIoU) and bbox-IoU helper.

Ports of upstream's ``damo/base_models/losses/gfocal_loss.py`` and
``damo/base_models/core/bbox_calculator.py::bbox_overlaps``.
"""

from __future__ import annotations

import functools

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
