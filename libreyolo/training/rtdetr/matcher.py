"""
Hungarian Matcher for RT-DETR training.

Performs bipartite matching between predictions and ground truth
using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


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
