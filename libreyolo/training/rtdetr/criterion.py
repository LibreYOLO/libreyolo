"""
RT-DETR Loss Criterion.

Implements VFL (Varifocal Loss) + L1 + GIoU losses for RT-DETR training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, generalized_box_iou


class RTDETRCriterion(nn.Module):
    """
    RT-DETR loss computation.

    Computes:
    - VFL (Varifocal Loss) for classification
    - L1 loss for box coordinates
    - GIoU loss for box regression
    """

    def __init__(
        self,
        num_classes: int = 80,
        loss_vfl_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
        matcher_cost_class: float = 2.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of object classes
            loss_vfl_weight: Weight for VFL classification loss
            loss_bbox_weight: Weight for L1 box loss
            loss_giou_weight: Weight for GIoU loss
            alpha: VFL alpha parameter
            gamma: VFL gamma parameter
            matcher_cost_class: Matcher classification cost weight
            matcher_cost_bbox: Matcher L1 cost weight
            matcher_cost_giou: Matcher GIoU cost weight
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_vfl_weight = loss_vfl_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.alpha = alpha
        self.gamma = gamma

        # Build matcher
        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of dicts with 'labels' and 'boxes'

        Returns:
            Dict with 'loss_vfl', 'loss_bbox', 'loss_giou', 'loss' (total)
        """
        # Hungarian matching
        indices = self.matcher(outputs, targets)

        # Number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        # Compute losses
        loss_vfl = self._loss_vfl(outputs, targets, indices, num_boxes)
        loss_bbox, loss_giou = self._loss_boxes(outputs, targets, indices, num_boxes)

        # Total loss
        total_loss = (
            self.loss_vfl_weight * loss_vfl
            + self.loss_bbox_weight * loss_bbox
            + self.loss_giou_weight * loss_giou
        )

        return {
            "loss_vfl": loss_vfl,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss": total_loss,
        }

    def _loss_vfl(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> torch.Tensor:
        """Varifocal Loss for classification."""
        pred_logits = outputs["pred_logits"]  # (B, Q, C)
        device = pred_logits.device
        bs, num_queries = pred_logits.shape[:2]

        # Get matched indices
        idx = self._get_src_permutation_idx(indices)

        # Create target classification tensor
        target_classes = torch.full(
            (bs, num_queries),
            self.num_classes,  # Background class
            dtype=torch.int64,
            device=device,
        )

        if len(idx[0]) > 0:
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)]
            )
            target_classes[idx] = target_classes_o.to(device)

        # One-hot target (excluding background)
        target_onehot = F.one_hot(
            target_classes.clamp(max=self.num_classes), self.num_classes + 1
        )[..., :-1].float()

        # Compute IoU for VFL weighting
        pred_boxes = outputs["pred_boxes"]
        target_score = torch.zeros_like(target_onehot)

        if len(idx[0]) > 0:
            src_boxes = pred_boxes[idx]
            target_boxes = torch.cat(
                [t["boxes"][J].to(device) for t, (_, J) in zip(targets, indices)], dim=0
            )

            # Compute IoU between matched predictions and targets
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

            # Compute pairwise IoU and take diagonal
            giou_matrix = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
            ious = torch.diag(giou_matrix).clamp(min=0).detach()

            # Set IoU as target score for matched positions
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices)]
            ).to(device)
            target_score[idx[0], idx[1], target_classes_o] = ious

        # Varifocal loss computation
        pred_score = pred_logits.sigmoid().detach()
        weight = (
            self.alpha * pred_score.pow(self.gamma) * (1 - target_onehot)
            + target_score
        )

        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_score, weight=weight.detach(), reduction="none"
        )
        loss = loss.mean(1).sum() * pred_logits.shape[1] / num_boxes

        return loss

    def _loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L1 and GIoU losses for box regression."""
        device = outputs["pred_boxes"].device
        idx = self._get_src_permutation_idx(indices)

        if len(idx[0]) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, zero

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][J].to(device) for t, (_, J) in zip(targets, indices)], dim=0
        )

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes

        return loss_bbox, loss_giou

    @staticmethod
    def _get_src_permutation_idx(
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get batch and source indices from matching."""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
