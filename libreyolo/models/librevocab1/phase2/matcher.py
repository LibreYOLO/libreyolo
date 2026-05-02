"""LibreVocab1 Phase 2 — Hungarian matcher with cosine-classification cost.

The standard DETR matcher uses a per-query class-probability cost. In an
open-vocab setting the "class" is a text embedding, and the predicted class
score is the cosine of the query against that text embedding. This file
implements that variant.

Output of the decoder per image:
    pred_logits: (Q, K) — cosine-similarity-based class scores against the
                          K text prompts present in this image's batch.
    pred_boxes:  (Q, 4) — cxcywh in [0, 1].

Targets per image:
    target_labels: (N,) long, indices into the K-prompt list (i.e. "this GT
                   belongs to the j-th prompt name").
    target_boxes:  (N, 4) cxcywh in [0, 1].

Matcher output per image:
    Tuple of (pred_indices, target_indices), each (M,) long, M = min(Q, N).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou_pair


class OpenVocabHungarianMatcher(nn.Module):
    """Hungarian matcher for the open-vocab detector.

    Cost = ``cost_class * cls_cost + cost_bbox * box_l1 + cost_giou * (-giou)``.

    The class cost is a focal-style cost on the cosine logit:
        positive_cost = (-α * (1-p)^γ * log(p))[t]
        negative_cost = (-(1-α) * p^γ * log(1-p))[t]
        cls_cost      = positive_cost - negative_cost
    matching DEIMv2 / RT-DETRv2's matcher recipe.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if cost_class <= 0 and cost_bbox <= 0 and cost_giou <= 0:
            raise ValueError("at least one cost weight must be positive")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Compute (pred_idx, target_idx) per image.

        Args:
            outputs: ``{'pred_logits': (B, Q, K), 'pred_boxes': (B, Q, 4)}``.
                ``pred_logits`` are pre-sigmoid scores (cosine / temperature).
            targets: list of B dicts, each:
                ``{'labels': (N_i,) long, 'boxes_cxcywh': (N_i, 4)}``.

        Returns:
            List of B tuples ``(pred_idx, tgt_idx)`` of matched indices.
        """
        # scipy is only required at training time. Import lazily so unit tests
        # without scipy can still import the module.
        from scipy.optimize import linear_sum_assignment

        bs, num_queries, _ = outputs["pred_logits"].shape
        results: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Iterate per image — the K dimension and the target sets vary by image.
        for b in range(bs):
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes_cxcywh"]
            n_tgt = tgt_labels.shape[0]

            if n_tgt == 0:
                empty = torch.zeros(0, dtype=torch.long)
                results.append((empty.clone(), empty.clone()))
                continue

            logits_b = outputs["pred_logits"][b]   # (Q, K)
            boxes_b = outputs["pred_boxes"][b]     # (Q, 4) cxcywh

            # Class cost (focal-style on cosine logits).
            probs = logits_b.sigmoid()                                   # (Q, K)
            pos_cost = -self.alpha * (1 - probs) ** self.gamma * (probs.clamp(min=1e-8)).log()
            neg_cost = -(1 - self.alpha) * probs ** self.gamma * ((1 - probs).clamp(min=1e-8)).log()
            cls_cost = pos_cost - neg_cost                                # (Q, K)
            cls_cost = cls_cost[:, tgt_labels]                            # (Q, n_tgt)

            # L1 box cost (cxcywh, normalized).
            box_cost = torch.cdist(boxes_b, tgt_boxes, p=1)               # (Q, n_tgt)

            # GIoU cost.
            giou = generalized_box_iou_pair(
                box_cxcywh_to_xyxy(boxes_b), box_cxcywh_to_xyxy(tgt_boxes)
            )                                                              # (Q, n_tgt)
            giou_cost = -giou

            cost = (
                self.cost_class * cls_cost
                + self.cost_bbox * box_cost
                + self.cost_giou * giou_cost
            )

            cost_np = cost.detach().cpu().numpy()
            pred_idx, tgt_idx = linear_sum_assignment(cost_np)
            results.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.long),
                    torch.as_tensor(tgt_idx, dtype=torch.long),
                )
            )
        return results
