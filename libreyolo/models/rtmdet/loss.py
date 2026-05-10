"""
RTMDet training losses and label assignment.

Cleanroom port of the components defined in mmdetection / mmyolo (Apache-2.0):
- ``QualityFocalLoss``: classification loss with IoU-soft targets
- ``GIoULoss``: bounding-box regression loss
- ``BatchDynamicSoftLabelAssigner``: dynamic-k label assignment with soft cls cost
- ``MlvlPointGenerator``: cell-corner priors with stride for each FPN level

All operations are pure PyTorch; no mmcv / mmengine runtime dependency.

The implementation follows mmyolo's ``loss_by_feat`` (mmyolo/models/dense_heads/
rtmdet_head.py:274-368) but adapts to LibreYOLO's head output convention,
which already multiplies the regression branch by stride and (per-size)
applies ``exp_on_reg``. Therefore the loss does NOT re-multiply by stride
before ``distance2bbox``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_INF = 100_000_000
_EPS = 1.0e-7


# =============================================================================
# Priors
# =============================================================================


class MlvlPointGenerator:
    """Per-level grid of cell-corner priors (mmdet's MlvlPointGenerator with offset=0).

    Returns ``(N_total, 3)`` tensors of ``[x, y, stride]`` where ``N_total`` is
    the sum of ``H_i * W_i`` across all FPN levels.
    """

    def __init__(self, strides: Sequence[int] = (8, 16, 32)):
        self.strides = list(strides)

    def grid_priors(
        self, featmap_sizes: List[Tuple[int, int]], device, dtype=torch.float32
    ) -> torch.Tensor:
        """Build priors for the given (H, W) per level.

        Output: ``(N_total, 3)`` with columns ``[x, y, stride]``.
        """
        all_priors = []
        for (h, w), stride in zip(featmap_sizes, self.strides):
            sx = torch.arange(w, device=device, dtype=dtype) * stride
            sy = torch.arange(h, device=device, dtype=dtype) * stride
            yy, xx = torch.meshgrid(sy, sx, indexing="ij")
            stride_col = torch.full(
                (h * w,), float(stride), device=device, dtype=dtype
            )
            level = torch.stack(
                [xx.reshape(-1), yy.reshape(-1), stride_col], dim=-1
            )
            all_priors.append(level)
        return torch.cat(all_priors, dim=0)


# =============================================================================
# Geometry helpers
# =============================================================================


def distance2bbox(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Decode ``(N, 4)`` ltrb distances against ``(N, 2)`` points to xyxy."""
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def batched_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for batched ``(B, N, 4)`` vs ``(B, M, 4)`` xyxy boxes.

    Returns ``(B, N, M)``.
    """
    b1 = boxes1.unsqueeze(2)  # (B, N, 1, 4)
    b2 = boxes2.unsqueeze(1)  # (B, 1, M, 4)

    inter_lt = torch.maximum(b1[..., :2], b2[..., :2])
    inter_rb = torch.minimum(b1[..., 2:], b2[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=_EPS)


def bbox_giou_aligned(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU for paired (N, 4) xyxy boxes. Returns ``(N,)`` GIoU."""
    pred_lt, pred_rb = pred[..., :2], pred[..., 2:]
    targ_lt, targ_rb = target[..., :2], target[..., 2:]

    inter_lt = torch.maximum(pred_lt, targ_lt)
    inter_rb = torch.minimum(pred_rb, targ_rb)
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_p = (pred_rb[..., 0] - pred_lt[..., 0]).clamp(min=0) * (
        pred_rb[..., 1] - pred_lt[..., 1]
    ).clamp(min=0)
    area_t = (targ_rb[..., 0] - targ_lt[..., 0]).clamp(min=0) * (
        targ_rb[..., 1] - targ_lt[..., 1]
    ).clamp(min=0)
    union = area_p + area_t - inter

    enc_lt = torch.minimum(pred_lt, targ_lt)
    enc_rb = torch.maximum(pred_rb, targ_rb)
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[..., 0] * enc_wh[..., 1]

    iou = inter / union.clamp(min=_EPS)
    giou = iou - (enc_area - union) / enc_area.clamp(min=_EPS)
    return giou


class GIoULoss(nn.Module):
    """``1 - GIoU`` loss with optional sample weights and avg_factor reduction."""

    def __init__(self, loss_weight: float = 2.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0
        loss = 1.0 - bbox_giou_aligned(pred, target)
        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / max(avg_factor, 1.0)
        return self.loss_weight * loss


# =============================================================================
# Quality Focal Loss
# =============================================================================


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss (Li et al., 2020).

    Target is a ``(label, iou_score)`` pair: positives are supervised with the
    IoU score against their assigned GT (so the cls logit learns to predict
    IoU as a quality estimate), negatives are supervised with 0.
    """

    def __init__(self, use_sigmoid: bool = True, beta: float = 2.0, loss_weight: float = 1.0):
        super().__init__()
        assert use_sigmoid, "Only sigmoid-based QFL is implemented (RTMDet uses sigmoid)."
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: Tuple[torch.Tensor, torch.Tensor],
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """``pred``: logits (N, num_classes); ``target``: (labels, scores)."""
        labels, scores = target
        pred_sigmoid = pred.sigmoid()

        # All-zero target (negatives)
        zerolabel = torch.zeros_like(pred)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction="none"
        ) * pred_sigmoid.pow(self.beta)

        bg_class_ind = pred.size(1)
        pos = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        if pos.numel() > 0:
            pos_labels = labels[pos].long()
            pos_scores = scores[pos].to(pred.dtype)
            pos_pred = pred[pos, pos_labels]
            scale_factor = (pos_scores - pred_sigmoid[pos, pos_labels]).abs()
            loss[pos, pos_labels] = F.binary_cross_entropy_with_logits(
                pos_pred, pos_scores, reduction="none"
            ) * scale_factor.pow(self.beta)

        loss = loss.sum(dim=1)
        if weight is not None:
            loss = loss * weight
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / max(avg_factor, 1.0)
        return self.loss_weight * loss


# =============================================================================
# Dynamic-k label assignment (batched)
# =============================================================================


def _find_inside_points(
    boxes: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Boolean ``(N_points, B, N_gt)`` whether each point is inside each GT box.

    ``boxes``: (B, N_gt, 4) xyxy. ``points``: (N_points, 2) xy.
    """
    lt = points[:, None, None] - boxes[..., :2]
    rb = boxes[..., 2:] - points[:, None, None]
    deltas = torch.cat([lt, rb], dim=-1)
    return deltas.min(dim=-1).values > 0


class BatchDynamicSoftLabelAssigner(nn.Module):
    """Dynamic-k assignment with soft cls + IoU + center-prior cost.

    Cleanroom port of mmyolo's BatchDynamicSoftLabelAssigner
    (mmyolo/models/task_modules/assigners/batch_dsl_assigner.py).
    """

    def __init__(
        self,
        num_classes: int,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: torch.Tensor,  # (B, N_priors, 4) xyxy
        pred_scores: torch.Tensor,  # (B, N_priors, num_classes) logits
        priors: torch.Tensor,        # (N_priors, 3) [x, y, stride]
        gt_labels: torch.Tensor,     # (B, N_gt, 1)
        gt_bboxes: torch.Tensor,     # (B, N_gt, 4) xyxy
        pad_bbox_flag: torch.Tensor, # (B, N_gt, 1) 0/1 mask of valid GTs
    ) -> dict:
        batch_size, num_priors, _ = pred_bboxes.shape
        num_gt = gt_bboxes.size(1)

        if num_gt == 0 or num_priors == 0:
            return {
                "assigned_labels": gt_labels.new_full(
                    pred_scores[..., 0].shape, self.num_classes, dtype=torch.long
                ),
                "assigned_bboxes": gt_bboxes.new_zeros(pred_bboxes.shape),
                "assign_metrics": gt_bboxes.new_zeros(pred_scores[..., 0].shape),
            }

        prior_xy = priors[:, :2]
        prior_stride = priors[:, 2]

        is_in_gts = _find_inside_points(gt_bboxes, prior_xy)  # (N_priors, B, N_gt)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        is_in_gts = is_in_gts.permute(1, 0, 2)  # (B, N_priors, N_gt)
        valid_mask = is_in_gts.sum(dim=-1) > 0  # (B, N_priors)

        # Soft center prior: distance from prior to gt center, normalized by stride
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) * 0.5
        distance = (
            (prior_xy[None].unsqueeze(2) - gt_center[:, None, :, :])
            .pow(2)
            .sum(-1)
            .sqrt()
            / prior_stride[None, :, None]
        )
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10.0, distance - self.soft_center_radius)

        # IoU cost
        pairwise_ious = batched_box_iou(pred_bboxes, gt_bboxes)  # (B, N_priors, N_gt)
        iou_cost = -torch.log(pairwise_ious + _EPS) * self.iou_weight

        # Cls cost: gather predicted score for each GT's class
        # pred_scores: (B, N_priors, C). gt_labels: (B, N_gt, 1).
        gt_cls = gt_labels.long().squeeze(-1)  # (B, N_gt)
        # For each (b, n_gt), select pred_scores[b, :, gt_cls[b, n_gt]]
        # → result shape (B, N_priors, N_gt)
        b_idx = torch.arange(batch_size, device=pred_scores.device).view(-1, 1).expand(-1, num_gt)
        pairwise_pred_scores = pred_scores.permute(0, 2, 1)[b_idx, gt_cls].permute(0, 2, 1)

        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious, reduction="none"
        ) * scale_factor.abs().pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior

        # Mask invalid (outside-any-GT) priors with INF so they never get picked
        max_pad_value = torch.full_like(cost_matrix, _INF)
        cost_matrix = torch.where(
            valid_mask[..., None].expand(-1, -1, num_gt), cost_matrix, max_pad_value
        )

        matched_pred_ious, matched_gt_inds, fg_mask = self._dynamic_k_matching(
            cost_matrix, pairwise_ious, pad_bbox_flag
        )

        batch_index = (fg_mask > 0).nonzero(as_tuple=True)[0]
        assigned_labels = gt_labels.new_full(
            pred_scores[..., 0].shape, self.num_classes, dtype=torch.long
        )
        assigned_labels[fg_mask] = gt_labels[batch_index, matched_gt_inds].squeeze(-1).long()

        assigned_bboxes = gt_bboxes.new_zeros(pred_bboxes.shape)
        assigned_bboxes[fg_mask] = gt_bboxes[batch_index, matched_gt_inds]

        assign_metrics = gt_bboxes.new_zeros(pred_scores[..., 0].shape)
        assign_metrics[fg_mask] = matched_pred_ious

        return {
            "assigned_labels": assigned_labels,
            "assigned_bboxes": assigned_bboxes,
            "assign_metrics": assign_metrics,
        }

    def _dynamic_k_matching(
        self,
        cost_matrix: torch.Tensor,   # (B, N_priors, N_gt)
        pairwise_ious: torch.Tensor, # (B, N_priors, N_gt)
        pad_bbox_flag: torch.Tensor, # (B, N_gt, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)

        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        num_gts = pad_bbox_flag.sum((1, 2)).int()
        _, sorted_indices = torch.sort(cost_matrix, dim=1)

        for b in range(pad_bbox_flag.shape[0]):
            for gt_idx in range(int(num_gts[b].item())):
                k = int(dynamic_ks[b, gt_idx].item())
                topk_ids = sorted_indices[b, :k, gt_idx]
                matching_matrix[b, topk_ids, gt_idx] = 1

        # Resolve double-assigned priors by min-cost
        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_argmin = torch.argmin(cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] = 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        fg_mask = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(2)[fg_mask]
        matched_gt_inds = matching_matrix[fg_mask, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask


# =============================================================================
# Top-level RTMDet loss
# =============================================================================


class RTMDetLoss(nn.Module):
    """Combines QFL classification, GIoU box loss, and the dynamic-k assigner.

    Inputs (forward):
        cls_scores: tuple of (B, num_classes, H_l, W_l) per FPN level
        bbox_preds: tuple of (B, 4, H_l, W_l) per FPN level — already in pixel
                    distances (LibreYOLO head pre-multiplies by stride and
                    optionally applies exp_on_reg).
        gt_boxes_list:  per-image list of (n_i, 4) xyxy GT boxes
        gt_labels_list: per-image list of (n_i,) GT class indices
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32),
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        qfl_beta: float = 2.0,
        assigner_topk: int = 13,
        soft_center_radius: float = 3.0,
        iou_weight: float = 3.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = list(strides)
        self.loss_cls = QualityFocalLoss(beta=qfl_beta, loss_weight=loss_cls_weight)
        self.loss_bbox = GIoULoss(loss_weight=loss_bbox_weight)
        self.assigner = BatchDynamicSoftLabelAssigner(
            num_classes=num_classes,
            soft_center_radius=soft_center_radius,
            topk=assigner_topk,
            iou_weight=iou_weight,
        )
        self.prior_generator = MlvlPointGenerator(strides=strides)

    def forward(
        self,
        cls_scores: Sequence[torch.Tensor],
        bbox_preds: Sequence[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> dict:
        device = cls_scores[0].device
        dtype = cls_scores[0].dtype
        batch_size = cls_scores[0].size(0)

        featmap_sizes = [tuple(c.shape[-2:]) for c in cls_scores]
        priors = self.prior_generator.grid_priors(featmap_sizes, device=device, dtype=dtype)

        # Flatten: cat over levels -> (B, N_priors, C / 4)
        flat_cls = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) for c in cls_scores],
            dim=1,
        )
        flat_dist = torch.cat(
            [r.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for r in bbox_preds],
            dim=1,
        )
        # Decode distances to xyxy boxes
        prior_xy = priors[:, :2]
        decoded_boxes = torch.stack(
            [
                prior_xy[:, 0] - flat_dist[..., 0],
                prior_xy[:, 1] - flat_dist[..., 1],
                prior_xy[:, 0] + flat_dist[..., 2],
                prior_xy[:, 1] + flat_dist[..., 3],
            ],
            dim=-1,
        )

        # Pack GTs to a fixed-length tensor for the batched assigner.
        max_gt = max((b.shape[0] for b in gt_boxes_list), default=0)
        gt_bboxes = torch.zeros(batch_size, max(max_gt, 1), 4, device=device, dtype=dtype)
        gt_labels = torch.zeros(batch_size, max(max_gt, 1), 1, device=device, dtype=dtype)
        pad_flag = torch.zeros(batch_size, max(max_gt, 1), 1, device=device, dtype=dtype)
        for i, (gb, gl) in enumerate(zip(gt_boxes_list, gt_labels_list)):
            n = gb.shape[0]
            if n == 0:
                continue
            gt_bboxes[i, :n] = gb.to(device=device, dtype=dtype)
            gt_labels[i, :n, 0] = gl.to(device=device, dtype=dtype)
            pad_flag[i, :n, 0] = 1.0

        assigned = self.assigner(
            decoded_boxes.detach(), flat_cls.detach(), priors,
            gt_labels, gt_bboxes, pad_flag,
        )

        labels = assigned["assigned_labels"].reshape(-1)
        bbox_targets = assigned["assigned_bboxes"].reshape(-1, 4)
        assign_metrics = assigned["assign_metrics"].reshape(-1)
        cls_preds = flat_cls.reshape(-1, self.num_classes)
        decoded_flat = decoded_boxes.reshape(-1, 4)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        avg_factor = max(float(assign_metrics.sum().item()), 1.0)

        loss_cls = self.loss_cls(
            cls_preds, (labels, assign_metrics), avg_factor=avg_factor
        )

        if pos_inds.numel() > 0:
            loss_bbox = self.loss_bbox(
                decoded_flat[pos_inds],
                bbox_targets[pos_inds],
                weight=assign_metrics[pos_inds],
                avg_factor=avg_factor,
            )
        else:
            loss_bbox = decoded_flat.sum() * 0

        total = loss_cls + loss_bbox
        return {
            # ``BaseTrainer`` reads ``total_loss`` for the backward pass.
            "total_loss": total,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
        }
