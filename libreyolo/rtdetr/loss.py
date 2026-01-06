"""
RT-DETR Loss with Hungarian Matcher.

Implements VFL (Varifocal Loss) + L1 + GIoU losses for RT-DETR training
with Hungarian bipartite matching for label assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


# =============================================================================
# Box Utilities
# =============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (M, 4) boxes in xyxy format

    Returns:
        (N, M) GIoU matrix
    """
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = rb_enc - lt_enc
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    giou = iou - (area_enc - union) / area_enc.clamp(min=1e-6)
    return giou


# =============================================================================
# Hungarian Matcher
# =============================================================================

class HungarianMatcher(nn.Module):
    """
    Performs bipartite matching between predictions and ground truth.

    Cost = cost_class * C_class + cost_bbox * C_bbox + cost_giou * C_giou

    The matching is performed using the Hungarian algorithm which finds
    the optimal one-to-one assignment that minimizes the total cost.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_bbox: Weight for L1 box cost
            cost_giou: Weight for GIoU cost
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            outputs: Dict with:
                - 'pred_logits': (B, num_queries, num_classes)
                - 'pred_boxes': (B, num_queries, 4) in cxcywh normalized format
            targets: List of dicts (one per image) with:
                - 'labels': (N,) class indices
                - 'boxes': (N, 4) in cxcywh normalized format

        Returns:
            List of (pred_indices, target_indices) tuples for each batch item
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Handle empty batch
        if bs == 0:
            return []

        # Flatten predictions for cost computation
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*Q, C)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (B*Q, 4)

        # Concatenate all targets
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        # Handle empty targets
        if len(tgt_ids) == 0:
            return [
                (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
                for _ in range(bs)
            ]

        # Classification cost (focal loss style)
        out_prob_selected = out_prob[:, tgt_ids]
        neg_cost = (
            (1 - self.alpha)
            * (out_prob_selected**self.gamma)
            * (-(1 - out_prob_selected + 1e-8).log())
        )
        pos_cost = (
            self.alpha
            * ((1 - out_prob_selected) ** self.gamma)
            * (-(out_prob_selected + 1e-8).log())
        )
        cost_class = pos_cost - neg_cost

        # L1 bbox cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        # Hungarian matching per batch item
        sizes = [len(t["boxes"]) for t in targets]
        indices = []

        for i, c in enumerate(C.split(sizes, -1)):
            c_i = c[i]
            if c_i.shape[1] == 0:
                # No targets in this image
                indices.append(
                    (
                        torch.tensor([], dtype=torch.int64),
                        torch.tensor([], dtype=torch.int64),
                    )
                )
            else:
                row_ind, col_ind = linear_sum_assignment(c_i.numpy())
                indices.append(
                    (
                        torch.as_tensor(row_ind, dtype=torch.int64),
                        torch.as_tensor(col_ind, dtype=torch.int64),
                    )
                )

        return indices


# =============================================================================
# RT-DETR Loss
# =============================================================================

class RTDETRLoss(nn.Module):
    """
    RT-DETR loss computation.

    Computes:
    - VFL (Varifocal Loss) for classification
    - L1 loss for box coordinates
    - GIoU loss for box regression

    Uses Hungarian matching for label assignment.
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
            Dict with 'loss_vfl', 'loss_bbox', 'loss_giou', 'total_loss'
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
            "total_loss": total_loss,
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
