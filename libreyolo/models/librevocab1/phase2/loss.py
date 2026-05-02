"""LibreVocab1 Phase 2 — open-vocab DETR set criterion.

Three loss terms, weighted sum:
    - Varifocal Loss (VFL) on cosine-similarity classification logits.
    - L1 on cxcywh boxes (normalized [0, 1]).
    - GIoU on xyxy boxes.

The classification target is a *soft* IoU-weighted target (Varifocal style):
matched queries get a target equal to their IoU with the matched GT box,
and all other (query, prompt) pairs get target 0. Unmatched queries
contribute only background-class signal.

Hungarian assignment is delegated to :class:`OpenVocabHungarianMatcher`.

Usage:

    matcher = OpenVocabHungarianMatcher()
    criterion = OpenVocabSetCriterion(matcher,
                                       weight_class=1.0,
                                       weight_bbox=5.0,
                                       weight_giou=2.0)
    loss_dict = criterion(outputs, targets)
    total = sum(loss_dict.values())
    total.backward()
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou_pair
from .matcher import OpenVocabHungarianMatcher


def _sigmoid_varifocal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """VFL — Varifocal Loss on sigmoid logits.

    Args:
        logits: (..., K) raw scores.
        target: (..., K) soft target in [0, 1]. Positives carry the IoU,
            negatives are 0.
        alpha:  positive sample weight.
        gamma:  focal exponent applied only to negatives.

    Returns:
        (..., K) per-element loss; caller reduces.
    """
    p = logits.sigmoid()
    # Per-element focal weighting:
    #   positives (target > 0): alpha * target
    #   negatives (target = 0): (1 - alpha) * p^gamma
    pos_weight = alpha * target
    neg_weight = (1 - alpha) * p.pow(gamma)
    # Per-element BCE (numerically stable).
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(target > 0, pos_weight, neg_weight)
    return weight * bce


class OpenVocabSetCriterion(nn.Module):
    """Hungarian-matched set loss for the LibreVocab1 Phase 2 decoder."""

    def __init__(
        self,
        matcher: OpenVocabHungarianMatcher,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        vfl_alpha: float = 0.75,
        vfl_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.vfl_alpha = vfl_alpha
        self.vfl_gamma = vfl_gamma

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute the set-loss dict.

        Args:
            outputs:
                ``pred_logits`` : (B, Q, K)  pre-sigmoid cosine logits
                ``pred_boxes``  : (B, Q, 4)  cxcywh in [0, 1]
            targets: list of B dicts, each:
                ``labels``       : (N_i,) long, indices into [0, K)
                ``boxes_cxcywh`` : (N_i, 4) in [0, 1]

        Returns:
            ``{'loss_cls': ..., 'loss_bbox': ..., 'loss_giou': ...}``.
        """
        bs, Q, K = outputs["pred_logits"].shape
        device = outputs["pred_logits"].device
        indices = self.matcher(outputs, targets)

        # Build VFL soft targets across the whole batch in one tensor.
        cls_target = outputs["pred_logits"].new_zeros((bs, Q, K))

        # Accumulators for box terms.
        all_pred_boxes: List[torch.Tensor] = []
        all_tgt_boxes: List[torch.Tensor] = []
        num_matched = 0

        for b in range(bs):
            pred_idx, tgt_idx = indices[b]
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes_cxcywh"]
            if pred_idx.numel() == 0:
                continue

            pred_idx = pred_idx.to(device)
            tgt_idx = tgt_idx.to(device)

            # Matched predicted boxes vs matched target boxes — used for both
            # the box loss and the IoU we feed into the VFL soft target.
            pred_boxes_m = outputs["pred_boxes"][b][pred_idx]   # (M, 4) cxcywh
            tgt_boxes_m = tgt_boxes[tgt_idx]                     # (M, 4) cxcywh
            tgt_labels_m = tgt_labels[tgt_idx]                   # (M,)

            # GIoU between matched pairs (paired, not pairwise full).
            pair_giou = torch.diag(
                generalized_box_iou_pair(
                    box_cxcywh_to_xyxy(pred_boxes_m),
                    box_cxcywh_to_xyxy(tgt_boxes_m),
                )
            )
            # IoU in [0, 1] for the soft VFL target.
            iou_target = pair_giou.clamp(min=0).detach()

            # Place soft targets at (b, pred_idx, tgt_label_idx).
            cls_target[b, pred_idx, tgt_labels_m] = iou_target

            all_pred_boxes.append(pred_boxes_m)
            all_tgt_boxes.append(tgt_boxes_m)
            num_matched += pred_idx.numel()

        # Classification: VFL averaged by total predictions (B*Q*K) then
        # rescaled by num_matched to match DETR-style normalization.
        cls_loss = _sigmoid_varifocal_loss(
            outputs["pred_logits"],
            cls_target,
            alpha=self.vfl_alpha,
            gamma=self.vfl_gamma,
        )
        cls_loss = cls_loss.sum() / max(num_matched, 1)

        if num_matched > 0:
            pred_box_cat = torch.cat(all_pred_boxes, dim=0)
            tgt_box_cat = torch.cat(all_tgt_boxes, dim=0)
            box_loss = F.l1_loss(pred_box_cat, tgt_box_cat, reduction="sum") / num_matched
            giou_pair = torch.diag(
                generalized_box_iou_pair(
                    box_cxcywh_to_xyxy(pred_box_cat),
                    box_cxcywh_to_xyxy(tgt_box_cat),
                )
            )
            giou_loss = (1 - giou_pair).sum() / num_matched
        else:
            zero = outputs["pred_boxes"].sum() * 0.0  # connect to graph
            box_loss = zero
            giou_loss = zero

        return {
            "loss_cls": self.weight_class * cls_loss,
            "loss_bbox": self.weight_bbox * box_loss,
            "loss_giou": self.weight_giou * giou_loss,
        }
