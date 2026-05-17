"""YOLO-NAS loss functions for native LibreYOLO training.

Ported and adapted from SuperGradients PP-YOLOE / YOLO-NAS training code.
The implementation here stays self-contained and uses LibreYOLO tensor
conventions plus a small target adapter for the shared dataloader output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...utils.general import cxcywh_to_xyxy
from .nn import batch_distance2bbox


def flatten_yolonas_targets(targets: Tensor) -> Tensor:
    """Convert padded `(B, max_labels, 5)` targets to flat `(N, 6)`.

    Input rows are expected in `[class, cx, cy, w, h]` pixel coordinates.
    Empty rows are identified by non-positive width/height.
    """
    if targets.ndim != 3 or targets.shape[-1] != 5:
        raise ValueError(
            "YOLO-NAS training expects targets shaped (B, max_labels, 5), "
            f"got {tuple(targets.shape)}"
        )

    valid = (targets[..., 3] > 0) & (targets[..., 4] > 0)
    if not valid.any():
        return targets.new_zeros((0, 6))

    batch_idx = torch.arange(
        targets.shape[0], dtype=targets.dtype, device=targets.device
    ).view(-1, 1, 1)
    batch_idx = batch_idx.expand(-1, targets.shape[1], 1)
    flat = torch.cat([batch_idx, targets], dim=-1)
    return flat[valid]


def batch_iou_similarity(box1: Tensor, box2: Tensor, eps: float = 1e-9) -> Tensor:
    box1 = box1.unsqueeze(2)
    box2 = box2.unsqueeze(1)
    px1y1, px2y2 = box1[..., 0:2], box1[..., 2:4]
    gx1y1, gx2y2 = box2[..., 0:2], box2[..., 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp_min(0).prod(-1)
    area1 = (px2y2 - px1y1).clamp_min(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp_min(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def iou_similarity(box1: Tensor, box2: Tensor, eps: float = 1e-9) -> Tensor:
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)
    px1y1, px2y2 = box1[..., 0:2], box1[..., 2:4]
    gx1y1, gx2y2 = box2[..., 0:2], box2[..., 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp_min(0).prod(-1)
    area1 = (px2y2 - px1y1).clamp_min(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp_min(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def compute_max_iou_anchor(ious: Tensor) -> Tensor:
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute(0, 2, 1)
    return is_max_iou.type_as(ious)


def compute_max_iou_gt(ious: Tensor) -> Tensor:
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(dim=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.type_as(ious)


def bbox_center(boxes: Tensor) -> Tensor:
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def check_points_inside_bboxes(
    points: Tensor,
    bboxes: Tensor,
    center_radius_tensor: Optional[Tensor] = None,
    eps: float = 1e-9,
):
    points = points.unsqueeze(0).unsqueeze(0)
    x, y = points.chunk(2, dim=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, dim=-1)

    left = x - xmin
    top = y - ymin
    right = xmax - x
    bottom = ymax - y
    delta_ltrb = torch.cat([left, top, right, bottom], dim=-1)
    is_in_bboxes = delta_ltrb.min(dim=-1).values > eps

    if center_radius_tensor is not None:
        center_radius_tensor = center_radius_tensor.unsqueeze(0).unsqueeze(0)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        left = x - (cx - center_radius_tensor)
        top = y - (cy - center_radius_tensor)
        right = (cx + center_radius_tensor) - x
        bottom = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.cat([left, top, right, bottom], dim=-1)
        is_in_center = delta_ltrb_c.min(dim=-1).values > eps
        return (
            torch.logical_and(is_in_bboxes, is_in_center),
            torch.logical_or(is_in_bboxes, is_in_center),
        )

    return is_in_bboxes.type_as(bboxes)


def gather_topk_anchors(
    metrics: Tensor,
    topk: int,
    largest: bool = True,
    topk_mask: Optional[Tensor] = None,
    eps: float = 1e-9,
) -> Tensor:
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).type_as(
            metrics
        )
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(metrics)
    return is_in_topk * topk_mask


class ATSSAssigner(nn.Module):
    def __init__(
        self,
        topk: int = 9,
        num_classes: int = 80,
        force_gt_matching: bool = False,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(
        self,
        gt2anchor_distances: Tensor,
        num_anchors_list: list[int],
        pad_gt_mask: Optional[Tensor],
    ):
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, dim=-1
        )
        num_anchors_index = [0]
        for n in num_anchors_list[:-1]:
            num_anchors_index.append(num_anchors_index[-1] + n)

        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(
            gt2anchor_distances_list, num_anchors_index
        ):
            num_anchors = distances.shape[-1]
            _, topk_idxs = torch.topk(distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = (
                F.one_hot(topk_idxs, num_anchors)
                .sum(dim=-2)
                .type_as(gt2anchor_distances)
            )
            if pad_gt_mask is not None:
                is_in_topk = is_in_topk * pad_gt_mask
            is_in_topk_list.append(is_in_topk)
        return torch.cat(is_in_topk_list, dim=-1), torch.cat(topk_idxs_list, dim=-1)

    @torch.no_grad()
    def forward(
        self,
        anchor_bboxes: Tensor,
        num_anchors_list: list[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_gt_mask: Optional[Tensor],
        bg_index: int,
        gt_scores: Optional[Tensor] = None,
        pred_bboxes: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors],
                bg_index,
                dtype=torch.long,
                device=anchor_bboxes.device,
            )
            assigned_bboxes = torch.zeros(
                [batch_size, num_anchors, 4], device=anchor_bboxes.device
            )
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes], device=anchor_bboxes.device
            )
            return assigned_labels, assigned_bboxes, assigned_scores

        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = torch.norm(
            gt_centers - anchor_centers.unsqueeze(0), p=2, dim=-1
        ).reshape([batch_size, -1, num_anchors])

        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask
        )

        iou_candidates = ious * is_in_topk
        iou_threshold = torch.gather(
            iou_candidates.flatten(end_dim=-2),
            dim=1,
            index=topk_idxs.flatten(end_dim=-2),
        )
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + iou_threshold.std(
            dim=-1, keepdim=True
        )
        is_in_topk = torch.where(
            iou_candidates > iou_threshold, is_in_topk, torch.zeros_like(is_in_topk)
        )

        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)
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

        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious)
            if pad_gt_mask is not None:
                is_max_iou = is_max_iou * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile(
                [1, num_max_boxes, 1]
            )
            mask_positive = torch.where(mask_max_iou, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)

        assigned_gt_index = mask_positive.argmax(dim=-2)
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device
        ).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes

        assigned_labels = torch.gather(
            gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0
        )
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, bg_index),
        )

        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1).float()
        indices = [i for i in range(self.num_classes + 1) if i != bg_index]
        assigned_scores = torch.index_select(
            assigned_scores,
            index=torch.tensor(indices, device=assigned_scores.device),
            dim=-1,
        )

        if pred_bboxes is not None:
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(dim=-2).values.unsqueeze(-1)
            assigned_scores = assigned_scores * ious
        elif gt_scores is not None:
            gather_scores = torch.gather(
                gt_scores.flatten(), assigned_gt_index.flatten(), dim=0
            )
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(
                mask_positive_sum > 0, gather_scores, torch.zeros_like(gather_scores)
            )
            assigned_scores = assigned_scores * gather_scores.unsqueeze(-1)

        return assigned_labels, assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    def __init__(
        self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9
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
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        num_anchors_list: list[int],
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_gt_mask: Optional[Tensor],
        bg_index: int,
        gt_scores: Optional[Tensor] = None,
    ):
        del num_anchors_list, gt_scores

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors],
                bg_index,
                dtype=torch.long,
                device=gt_labels.device,
            )
            assigned_bboxes = torch.zeros(
                [batch_size, num_anchors, 4], device=gt_labels.device
            )
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes], device=gt_labels.device
            )
            return assigned_labels, assigned_bboxes, assigned_scores

        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device
        ).unsqueeze(-1)
        gt_labels_ind = torch.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1
        )
        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask
        )

        mask_positive = is_in_topk * is_in_gts
        if pad_gt_mask is not None:
            mask_positive *= pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1]
            )
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)

        assigned_gt_index = mask_positive.argmax(dim=-2)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(
            gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0
        )
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, bg_index),
        )

        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        indices = [i for i in range(num_classes + 1) if i != bg_index]
        assigned_scores = torch.index_select(
            assigned_scores,
            index=torch.tensor(
                indices, device=assigned_scores.device, dtype=torch.long
            ),
            dim=-1,
        )

        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = (
            alignment_metrics
            / (max_metrics_per_instance + self.eps)
            * max_ious_per_instance
        )
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores


class GIoULoss:
    def __init__(
        self, loss_weight: float = 1.0, eps: float = 1e-10, reduction: str = "none"
    ):
        self.loss_weight = loss_weight
        self.eps = eps
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def bbox_overlap(
        self, box1: Tensor, box2: Tensor, eps: float = 1e-10
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clamp_min(0)
        h_inter = (ykis2 - ykis1).clamp_min(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union
        return iou, overlap, union

    def __call__(self, pbox: Tensor, gbox: Tensor, iou_weight=1.0, loc_reweight=None):
        x1, y1, x2, y2 = pbox.chunk(4, dim=-1)
        x1g, y1g, x2g, y2g = gbox.chunk(4, dim=-1)

        iou, _, union = self.bbox_overlap(
            [x1, y1, x2, y2], [x1g, y1g, x2g, y2g], self.eps
        )
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou

        if self.reduction == "none":
            loss = giou
        elif self.reduction == "sum":
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


class PPYoloELoss(nn.Module):
    """Native YOLO-NAS / PP-YOLOE loss for LibreYOLO."""

    def __init__(
        self,
        num_classes: int,
        use_varifocal_loss: bool = True,
        use_static_assigner: bool = False,
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.use_varifocal_loss = use_varifocal_loss
        self.classification_loss_weight = classification_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.num_classes = num_classes

        self.iou_loss = GIoULoss()
        self.static_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
        self.assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
        self.use_static_assigner = use_static_assigner

    def get_proj_conv_for_reg_max(self, reg_max: int, device: torch.device) -> Tensor:
        return torch.linspace(0, reg_max, reg_max + 1, device=device).reshape(
            [1, reg_max + 1, 1, 1]
        )

    @torch.no_grad()
    def _get_targets_for_batched_assigner(
        self, targets: Tensor, batch_size: int
    ) -> dict[str, Tensor]:
        image_index = targets[:, 0]
        gt_class = targets[:, 1:2].long()
        gt_bbox = cxcywh_to_xyxy(targets[:, 2:6])

        per_image_class = []
        per_image_bbox = []
        per_image_pad_mask = []

        max_boxes = 0
        for i in range(batch_size):
            mask = image_index == i
            image_labels = gt_class[mask]
            image_bboxes = gt_bbox[mask, :]
            valid_bboxes = image_bboxes.sum(dim=1, keepdims=True) > 0

            per_image_class.append(image_labels)
            per_image_bbox.append(image_bboxes)
            per_image_pad_mask.append(valid_bboxes)

            max_boxes = max(max_boxes, int(mask.sum().item()))

        for i in range(batch_size):
            elements_to_pad = max_boxes - len(per_image_class[i])
            pad = (0, 0, 0, elements_to_pad)
            per_image_class[i] = F.pad(
                per_image_class[i], pad, mode="constant", value=0
            )
            per_image_bbox[i] = F.pad(per_image_bbox[i], pad, mode="constant", value=0)
            per_image_pad_mask[i] = F.pad(
                per_image_pad_mask[i], pad, mode="constant", value=0
            )

        return {
            "gt_class": torch.stack(per_image_class, dim=0),
            "gt_bbox": torch.stack(per_image_bbox, dim=0),
            "pad_gt_mask": torch.stack(per_image_pad_mask, dim=0),
        }

    def _forward_batched(
        self,
        predictions: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        targets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            pred_scores,
            pred_distri,
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions

        targets = self._get_targets_for_batched_assigner(
            targets, batch_size=pred_scores.size(0)
        )
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, reg_max, _ = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        pad_gt_mask = targets["pad_gt_mask"]

        if self.use_static_assigner:
            assigned_labels, assigned_bboxes, assigned_scores = self.static_assigner(
                anchor_bboxes=anchors,
                num_anchors_list=num_anchors_list,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                pad_gt_mask=pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor,
            )
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores=pred_scores.detach().sigmoid(),
                pred_bboxes=pred_bboxes.detach() * stride_tensor,
                anchor_points=anchor_points,
                num_anchors_list=num_anchors_list,
                gt_labels=gt_labels,
                gt_bboxes=gt_bboxes,
                pad_gt_mask=pad_gt_mask,
                bg_index=self.num_classes,
            )
            alpha_l = -1

        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            cls_loss_sum = self._varifocal_loss(
                pred_scores, assigned_scores, one_hot_label
            )
        else:
            cls_loss_sum = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        iou_loss_sum, dfl_loss_sum = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points_s,
            assigned_labels,
            assigned_bboxes / stride_tensor,
            assigned_scores,
            reg_max,
        )
        return cls_loss_sum, iou_loss_sum, dfl_loss_sum, assigned_scores_sum

    def forward(self, outputs, targets: Tensor):
        if targets.ndim == 3:
            targets = flatten_yolonas_targets(targets)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            _, predictions = outputs
        else:
            predictions = outputs

        cls_loss_sum, iou_loss_sum, dfl_loss_sum, assigned_scores_sum = (
            self._forward_batched(predictions, targets)
        )

        assigned_scores_sum = torch.clamp(assigned_scores_sum, min=1.0)
        cls_loss = self.classification_loss_weight * cls_loss_sum / assigned_scores_sum
        iou_loss = self.iou_loss_weight * iou_loss_sum / assigned_scores_sum
        dfl_loss = self.dfl_loss_weight * dfl_loss_sum / assigned_scores_sum
        loss = cls_loss + iou_loss + dfl_loss
        log_losses = torch.stack(
            [cls_loss.detach(), iou_loss.detach(), dfl_loss.detach(), loss.detach()]
        )
        return loss, log_losses

    def _df_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        pred_dist = torch.moveaxis(pred_dist, -1, 1)
        loss_left = (
            F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    def _bbox_loss(
        self,
        pred_dist: Tensor,
        pred_bboxes: Tensor,
        anchor_points: Tensor,
        assigned_labels: Tensor,
        assigned_bboxes: Tensor,
        assigned_scores: Tensor,
        reg_max: int,
    ) -> Tuple[Tensor, Tensor]:
        mask_positive = assigned_labels != self.num_classes
        num_pos = mask_positive.sum()
        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape(
                [-1, 4]
            )
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask
            ).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive
            ).unsqueeze(-1)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum()

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape(
                [-1, 4, reg_max + 1]
            )
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes, reg_max)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape(
                [-1, 4]
            )
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum()
        else:
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = pred_dist.sum() * 0.0
        return loss_iou, loss_dfl

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor):
        batch_size, num_locations, *_ = pred_dist.size()
        pred_dist = pred_dist.reshape([batch_size, num_locations, 4, -1])
        reg_max = pred_dist.size(-1) - 1
        proj_conv = self.get_proj_conv_for_reg_max(reg_max, device=pred_dist.device)
        pred_dist = torch.softmax(pred_dist, dim=-1)
        pred_dist = F.conv2d(pred_dist.permute(0, 3, 1, 2), proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist), reg_max, proj_conv

    def _bbox2distance(self, points: Tensor, bbox: Tensor, reg_max: int):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)

    @staticmethod
    def _focal_loss(
        pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0
    ) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = (pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = weight * F.binary_cross_entropy_with_logits(
            pred_logits, label, reduction="none"
        )
        return loss.sum()

    @staticmethod
    def _varifocal_loss(
        pred_logits: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0
    ) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = weight * F.binary_cross_entropy_with_logits(
            pred_logits, gt_score, reduction="none"
        )
        return loss.sum()


# ===========================================================================
# Pose estimation loss
#
# Ported from SuperGradients ``YoloNASPoseLoss`` / ``yolo_nas_pose_loss.py``.
# The assignment, OKS, and keypoint loss math are upstream-faithful; only the
# target adapter differs — LibreYOLO feeds padded ``(B, max_labels, 5 + 3K)``
# tensors instead of the SuperGradients flat ``(boxes, joints, crowd)`` tuple.
# ===========================================================================


def bbox_ciou_loss(pred_bboxes: Tensor, target_bboxes: Tensor, eps: float) -> Tensor:
    """CIoU loss between predicted and target boxes (both XYXY, ``[..., 4]``).

    Returns the per-box loss as a ``[..., 1]`` tensor.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, dim=-1)

    iou, _, _ = GIoULoss().bbox_overlap(
        [b1_x1, b1_y1, b1_x2, b1_y2], [b2_x1, b2_y1, b2_x2, b2_y2], eps
    )
    iou_term = 1 - iou

    xc1 = torch.minimum(b1_x1, b2_x1)
    yc1 = torch.minimum(b1_y1, b2_y1)
    xc2 = torch.maximum(b1_x2, b2_x2)
    yc2 = torch.maximum(b1_y2, b2_y2)

    cw = xc2 - xc1
    ch = yc2 - yc1
    diagonal_distance_squared = cw**2 + ch**2

    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    centers_distance_squared = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2
    distance_term = centers_distance_squared / (diagonal_distance_squared + eps)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    v = (4 / math.pi**2) * torch.pow(
        torch.atan(w2 / h2.clamp_min(eps)) - torch.atan(w1 / h1.clamp_min(eps)), 2
    )
    with torch.no_grad():
        alpha = v / ((1 - iou) + v).clamp_min(eps)

    return iou_term + distance_term + v * alpha


class CIoULoss(nn.Module):
    """Complete-IoU loss with optional per-box weighting."""

    def __init__(self, eps: float = 1e-10, reduction: str = "none"):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.eps = eps
        self.reduction = reduction

    def __call__(
        self, predictions: Tensor, targets: Tensor, loc_weights: Optional[Tensor] = None
    ) -> Tensor:
        loss = bbox_ciou_loss(predictions, targets, eps=self.eps)
        if loc_weights is not None:
            loss = loss * loc_weights
        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        return loss


def batch_pose_oks(
    gt_keypoints: Tensor,
    pred_keypoints: Tensor,
    gt_bboxes_xyxy: Tensor,
    sigmas: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """Batched OKS between ground-truth and predicted keypoints.

    :param gt_keypoints:   ``[N, M1, K, 3]`` (x, y, visibility)
    :param pred_keypoints: ``[N, M2, K, 2]`` (x, y)
    :param gt_bboxes_xyxy: ``[N, M1, 4]`` XYXY
    :param sigmas:         ``[K]`` per-keypoint sigmas
    :return:               OKS of shape ``[N, M1, M2]``
    """
    joints1_xy = gt_keypoints[:, :, :, 0:2].unsqueeze(2)  # [N, M1, 1, K, 2]
    joints2_xy = pred_keypoints[:, :, :, 0:2].unsqueeze(1)  # [N, 1, M2, K, 2]

    d = ((joints1_xy - joints2_xy) ** 2).sum(dim=-1, keepdim=False)  # [N, M1, M2, K]

    area = (
        (gt_bboxes_xyxy[:, :, 2] - gt_bboxes_xyxy[:, :, 0])
        * (gt_bboxes_xyxy[:, :, 3] - gt_bboxes_xyxy[:, :, 1])
        * 0.53
    )  # [N, M1]
    area = area[:, :, None, None]
    sigmas = sigmas.reshape([1, 1, 1, -1])

    e = d / (2 * sigmas) ** 2 / (area + eps) / 2
    oks = torch.exp(-e)

    joints1_visibility = gt_keypoints[:, :, :, 2].gt(0).float().unsqueeze(2)
    num_visible = joints1_visibility.sum(dim=-1, keepdim=False)
    oks = (oks * joints1_visibility).sum(dim=-1, keepdim=False) / (
        num_visible + eps
    )
    # If an object has no labelled keypoints, do not suppress its bbox
    # assignment. That target should train boxes/classes but contribute no
    # keypoint-regression signal.
    return torch.where(num_visible > 0, oks, torch.ones_like(oks))


@dataclass
class PoseAssignmentResult:
    """Result of assigning predicted boxes/poses to ground truth."""

    assigned_labels: Tensor
    assigned_bboxes: Tensor
    assigned_poses: Tensor
    assigned_scores: Tensor
    assigned_gt_index: Tensor
    assigned_crowd: Tensor


class YoloNASPoseTaskAlignedAssigner(nn.Module):
    """Task-aligned assigner for pose: assignment score blends OKS and IoU."""

    def __init__(
        self,
        sigmas: Tensor,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        multiply_by_pose_oks: bool = False,
    ):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.sigmas = sigmas
        self.multiply_by_pose_oks = multiply_by_pose_oks

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        pred_pose_coords: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        gt_poses: Tensor,
        gt_crowd: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
    ) -> PoseAssignmentResult:
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, _, num_keypoints, _ = pred_pose_coords.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        if num_max_boxes == 0:
            device = gt_labels.device
            return PoseAssignmentResult(
                assigned_labels=torch.full(
                    [batch_size, num_anchors], bg_index, dtype=torch.long, device=device
                ),
                assigned_bboxes=torch.zeros(
                    [batch_size, num_anchors, 4], device=device
                ),
                assigned_poses=torch.zeros(
                    [batch_size, num_anchors, num_keypoints, 3], device=device
                ),
                assigned_scores=torch.zeros(
                    [batch_size, num_anchors, num_classes], device=device
                ),
                assigned_gt_index=torch.zeros(
                    [batch_size, num_anchors], dtype=torch.long, device=device
                ),
                assigned_crowd=torch.zeros(
                    [batch_size, num_anchors], dtype=torch.bool, device=device
                ),
            )

        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        if self.multiply_by_pose_oks:
            pose_oks = batch_pose_oks(
                gt_poses, pred_pose_coords, gt_bboxes, self.sigmas.to(pred_pose_coords.device)
            )
            ious = ious * pose_oks

        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device
        ).unsqueeze(-1)
        gt_labels_ind = torch.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1
        )
        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask
        )
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile(
                [1, num_max_boxes, 1]
            )
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(
            gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0
        )
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0,
            assigned_labels,
            torch.full_like(assigned_labels, bg_index),
        )

        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_poses = gt_poses.reshape([-1, num_keypoints, 3])[
            assigned_gt_index.flatten(), :
        ]
        assigned_poses = assigned_poses.reshape(
            [batch_size, num_anchors, num_keypoints, 3]
        )

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = [i for i in range(num_classes + 1) if i != bg_index]
        assigned_scores = torch.index_select(
            assigned_scores,
            index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long),
            dim=-1,
        )
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = (
            alignment_metrics
            / (max_metrics_per_instance + self.eps)
            * max_ious_per_instance
        )
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        assigned_crowd = torch.gather(
            gt_crowd.flatten(), index=assigned_gt_index.flatten(), dim=0
        )
        assigned_crowd = assigned_crowd.reshape([batch_size, num_anchors])
        assigned_scores = assigned_scores * assigned_crowd.eq(0).unsqueeze(-1)

        return PoseAssignmentResult(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assigned_scores=assigned_scores,
            assigned_poses=assigned_poses,
            assigned_gt_index=assigned_gt_index,
            assigned_crowd=assigned_crowd,
        )


class YoloNASPoseLoss(nn.Module):
    """Native YOLO-NAS pose loss for LibreYOLO.

    Combines classification, box (IoU + DFL), and keypoint (OKS regression +
    visibility classification) terms. Defaults match the SuperGradients
    ``coco2017_yolo_nas_pose`` recipe.
    """

    def __init__(
        self,
        oks_sigmas: Union[List[float], np.ndarray, Tensor],
        classification_loss_type: str = "focal",
        regression_iou_loss_type: str = "ciou",
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.01,
        pose_cls_loss_weight: float = 1.0,
        pose_reg_loss_weight: float = 34.0,
        pose_classification_loss_type: str = "focal",
        bbox_assigner_topk: int = 13,
        bbox_assigned_alpha: float = 1.0,
        bbox_assigned_beta: float = 6.0,
        assigner_multiply_by_pose_oks: bool = True,
        rescale_pose_loss_with_assigned_score: bool = True,
    ):
        super().__init__()
        self.classification_loss_type = classification_loss_type
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.iou_loss = {"giou": GIoULoss, "ciou": CIoULoss}[regression_iou_loss_type]()
        self.num_keypoints = len(oks_sigmas)
        self.num_classes = 1  # pose estimation is single-class
        self.register_buffer("oks_sigmas", torch.as_tensor(oks_sigmas, dtype=torch.float32))
        self.pose_cls_loss_weight = pose_cls_loss_weight
        self.pose_reg_loss_weight = pose_reg_loss_weight
        self.assigner = YoloNASPoseTaskAlignedAssigner(
            sigmas=self.oks_sigmas,
            topk=bbox_assigner_topk,
            alpha=bbox_assigned_alpha,
            beta=bbox_assigned_beta,
            multiply_by_pose_oks=assigner_multiply_by_pose_oks,
        )
        self.pose_classification_loss_type = pose_classification_loss_type
        self.rescale_pose_loss_with_assigned_score = rescale_pose_loss_with_assigned_score

    @property
    def component_names(self) -> List[str]:
        return ["cls", "iou", "dfl", "pose_cls", "pose_reg", "loss"]

    @torch.no_grad()
    def _unpack_padded_targets(self, targets: Tensor) -> dict:
        """Convert padded ``(B, M, 5 + 3K)`` targets to per-image tensors.

        Each row is ``[cls, cx, cy, w, h, kx, ky, v, ...]`` in image pixel
        coordinates. Valid rows are packed at the front of each image's slab
        (guaranteed by the pose collate), so a single front slice suffices.
        """
        if targets.ndim != 3 or targets.shape[-1] != 5 + 3 * self.num_keypoints:
            raise ValueError(
                f"Pose loss expects targets shaped (B, M, {5 + 3 * self.num_keypoints}), "
                f"got {tuple(targets.shape)}"
            )
        batch_size = targets.shape[0]
        device = targets.device

        valid = (targets[..., 3] > 0) & (targets[..., 4] > 0)  # (B, M)
        n_max = int(valid.sum(dim=1).max().item()) if valid.numel() else 0

        if n_max == 0:
            return {
                "gt_class": torch.zeros(batch_size, 0, 1, dtype=torch.long, device=device),
                "gt_bbox": torch.zeros(batch_size, 0, 4, device=device),
                "pad_gt_mask": torch.zeros(batch_size, 0, 1, device=device),
                "gt_poses": torch.zeros(
                    batch_size, 0, self.num_keypoints, 3, device=device
                ),
                "gt_crowd": torch.zeros(batch_size, 0, 1, device=device),
            }

        t = targets[:, :n_max]
        gt_bbox = cxcywh_to_xyxy(t[..., 1:5])
        pad_gt_mask = ((t[..., 3] > 0) & (t[..., 4] > 0)).unsqueeze(-1).float()
        gt_class = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        gt_poses = t[..., 5:].reshape(batch_size, n_max, self.num_keypoints, 3)
        gt_crowd = torch.zeros(batch_size, n_max, 1, device=device)
        return {
            "gt_class": gt_class,
            "gt_bbox": gt_bbox,
            "pad_gt_mask": pad_gt_mask,
            "gt_poses": gt_poses,
            "gt_crowd": gt_crowd,
        }

    def forward(self, outputs, targets: Tensor) -> Tuple[Tensor, Tensor]:
        # outputs is (decoded_predictions, raw_predictions) from the pose head.
        _, predictions = outputs
        (
            pred_scores,
            pred_distri,
            pred_pose_coords,  # [B, A, K, 2]
            pred_pose_logits,  # [B, A, K]
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions
        del anchors, num_anchors_list

        tgt = self._unpack_padded_targets(targets)
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, reg_max = self._bbox_decode(anchor_points_s, pred_distri)

        assign_result = self.assigner(
            pred_scores=pred_scores.detach().sigmoid(),
            pred_bboxes=pred_bboxes.detach() * stride_tensor,
            pred_pose_coords=pred_pose_coords.detach(),
            anchor_points=anchor_points,
            gt_labels=tgt["gt_class"],
            gt_bboxes=tgt["gt_bbox"],
            gt_poses=tgt["gt_poses"],
            gt_crowd=tgt["gt_crowd"],
            pad_gt_mask=tgt["pad_gt_mask"],
            bg_index=self.num_classes,
        )

        assigned_scores = assign_result.assigned_scores
        if self.classification_loss_type == "focal":
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha=-1)
        elif self.classification_loss_type == "bce":
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_scores, assigned_scores, reduction="sum"
            )
        else:
            raise ValueError(
                f"Unknown classification loss type: {self.classification_loss_type}"
            )

        assigned_scores_sum = torch.clip(assigned_scores.sum(), min=1.0)
        loss_cls = loss_cls / assigned_scores_sum

        loss_iou, loss_dfl, loss_pose_cls, loss_pose_reg = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            pred_pose_coords=pred_pose_coords,
            pred_pose_logits=pred_pose_logits,
            stride_tensor=stride_tensor,
            anchor_points=anchor_points_s,
            assign_result=assign_result,
            assigned_scores_sum=assigned_scores_sum,
            reg_max=reg_max,
        )

        loss_cls = loss_cls * self.classification_loss_weight
        loss_iou = loss_iou * self.iou_loss_weight
        loss_dfl = loss_dfl * self.dfl_loss_weight
        loss_pose_cls = loss_pose_cls * self.pose_cls_loss_weight
        loss_pose_reg = loss_pose_reg * self.pose_reg_loss_weight

        loss = loss_cls + loss_iou + loss_dfl + loss_pose_cls + loss_pose_reg
        log_losses = torch.stack(
            [
                loss_cls.detach(),
                loss_iou.detach(),
                loss_dfl.detach(),
                loss_pose_cls.detach(),
                loss_pose_reg.detach(),
                loss.detach(),
            ]
        )
        return loss, log_losses

    def _df_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        pred_dist = torch.moveaxis(pred_dist, -1, 1)
        loss_left = F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        loss_right = (
            F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        )
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    def _keypoint_loss(
        self,
        predicted_coords: Tensor,
        target_coords: Tensor,
        predicted_logits: Tensor,
        target_visibility: Tensor,
        area: Tensor,
        sigmas: Tensor,
        assigned_scores: Optional[Tensor] = None,
        assigned_scores_sum: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        sigmas = sigmas.reshape([1, -1, 1])
        area = area.reshape([-1, 1, 1])

        visible_targets_mask = (target_visibility > 0).float()

        d = ((predicted_coords - target_coords) ** 2).sum(dim=-1, keepdim=True)
        e = d / (2 * sigmas) ** 2 / (area + 1e-9) / 2
        regression_loss_unreduced = 1 - torch.exp(-e)

        regression_loss_reduced = (regression_loss_unreduced * visible_targets_mask).sum(
            dim=1, keepdim=False
        ) / (visible_targets_mask.sum(dim=1, keepdim=False) + 1e-9)

        if self.pose_classification_loss_type == "bce":
            classification_loss = F.binary_cross_entropy_with_logits(
                predicted_logits, visible_targets_mask, reduction="none"
            ).mean(dim=1)
        elif self.pose_classification_loss_type == "focal":
            classification_loss = self._focal_loss(
                predicted_logits, visible_targets_mask, alpha=0.25, gamma=2.0, reduction="none"
            ).mean(dim=1)
        else:
            raise ValueError(
                f"Unsupported pose classification loss type "
                f"{self.pose_classification_loss_type}"
            )

        if assigned_scores is None:
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss_reduced.mean()
        else:
            classification_loss = (
                classification_loss * assigned_scores
            ).sum() / assigned_scores_sum
            regression_loss = (
                regression_loss_reduced * assigned_scores
            ).sum() / assigned_scores_sum
        return regression_loss, classification_loss

    @staticmethod
    def _xyxy_box_area(boxes: Tensor) -> Tensor:
        return (boxes[..., 2:4] - boxes[..., 0:2]).prod(dim=-1, keepdim=True)

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        pred_pose_coords,
        pred_pose_logits,
        stride_tensor,
        anchor_points,
        assign_result: PoseAssignmentResult,
        assigned_scores_sum,
        reg_max: int,
    ):
        # Positives that are neither background nor crowd.
        mask_positive = (
            assign_result.assigned_labels != self.num_classes
        ) * assign_result.assigned_crowd.eq(0)
        num_pos = mask_positive.sum()
        assigned_bboxes_div_stride = assign_result.assigned_bboxes / stride_tensor

        if num_pos > 0:
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes_div_stride, bbox_mask
            ).reshape([-1, 4])
            assigned_bboxes_pos_image = torch.masked_select(
                assign_result.assigned_bboxes, bbox_mask
            ).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assign_result.assigned_scores.sum(-1), mask_positive
            ).unsqueeze(-1)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape(
                [-1, 4, reg_max + 1]
            )
            assigned_ltrb = self._bbox2distance(
                anchor_points, assigned_bboxes_div_stride, reg_max
            )
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape(
                [-1, 4]
            )
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum

            # Poses stay in image coordinates (do not divide by stride).
            pred_pose_coords_pos = pred_pose_coords[mask_positive]
            pred_pose_logits_pos = pred_pose_logits[mask_positive].unsqueeze(-1)
            gt_pose_coords = assign_result.assigned_poses[..., 0:2][mask_positive]
            gt_pose_visibility = assign_result.assigned_poses[mask_positive][:, :, 2:3]

            area = self._xyxy_box_area(assigned_bboxes_pos_image).reshape([-1, 1]) * 0.53
            loss_pose_reg, loss_pose_cls = self._keypoint_loss(
                predicted_coords=pred_pose_coords_pos,
                target_coords=gt_pose_coords,
                predicted_logits=pred_pose_logits_pos,
                target_visibility=gt_pose_visibility,
                assigned_scores=bbox_weight
                if self.rescale_pose_loss_with_assigned_score
                else None,
                assigned_scores_sum=assigned_scores_sum
                if self.rescale_pose_loss_with_assigned_score
                else None,
                area=area,
                sigmas=self.oks_sigmas.to(pred_pose_logits.device),
            )
        else:
            loss_iou = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0
            loss_pose_cls = pred_pose_logits.sum() * 0.0
            loss_pose_reg = pred_pose_coords.sum() * 0.0
        return loss_iou, loss_dfl, loss_pose_cls, loss_pose_reg

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tuple[Tensor, int]:
        batch_size, num_locations, *_ = pred_dist.size()
        pred_dist = torch.softmax(
            pred_dist.reshape([batch_size, num_locations, 4, -1]), dim=-1
        )
        reg_max = pred_dist.size(-1) - 1
        proj_conv = torch.linspace(
            0, reg_max, reg_max + 1, device=pred_dist.device
        ).reshape([1, reg_max + 1, 1, 1])
        pred_dist = F.conv2d(pred_dist.permute(0, 3, 1, 2), proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist), reg_max

    def _bbox2distance(self, points: Tensor, bbox: Tensor, reg_max: int) -> Tensor:
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)

    @staticmethod
    def _focal_loss(
        pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum"
    ) -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = torch.abs(pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight = weight * alpha_t
        loss = weight * F.binary_cross_entropy_with_logits(
            pred_logits, label, reduction="none"
        )
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction type {reduction}")
