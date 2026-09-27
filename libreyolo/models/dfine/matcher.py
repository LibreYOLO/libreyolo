"""Hungarian matcher for D-FINE training.

Ported from D-FINE (https://github.com/Peterande/D-FINE).
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified from facebookresearch/detr.

Differences from upstream:
- ``@register()`` and ``__share__`` stripped; pass ``use_focal_loss`` as a
  plain ctor kwarg.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def dice_cost(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_masks = pred_masks.flatten(1).float()
    gt_masks = gt_masks.flatten(1).float()
    numerator = 2 * torch.einsum("qp,tp->qt", pred_masks, gt_masks)
    denominator = pred_masks.sum(dim=1, keepdim=True) + gt_masks.sum(
        dim=1, keepdim=False
    )
    return 1 - (numerator + eps) / (denominator + eps)


def sigmoid_focal_cost(
    pred_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    pred_logits = pred_logits.float()
    gt_labels = gt_labels.float()
    pred_prob = pred_logits.sigmoid()
    neg_cost = (1 - alpha) * (pred_prob**gamma) * (-(1 - pred_prob + 1e-8).log())
    pos_cost = alpha * ((1 - pred_prob) ** gamma) * (-(pred_prob + 1e-8).log())
    cost = torch.einsum("qp,tp->qt", pos_cost, gt_labels) + torch.einsum(
        "qp,tp->qt", neg_cost, (1 - gt_labels)
    )
    return cost / pred_logits.shape[1]


class HungarianMatcher(nn.Module):
    """One-to-one matching between predictions and targets via LSAP.

    Returns ``{"indices": [(pred_idx, tgt_idx) per image]}``. Same contract as
    upstream D-FINE.
    """

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_mask = weight_dict.get("cost_mask", 0.0)
        self.cost_mask_dice = weight_dict.get("cost_mask_dice", 0.0)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def compute_cost_matrix(self, outputs: Dict[str, torch.Tensor], targets):
        """Build the pairwise matching cost on the predictions' device.

        Split out from :meth:`forward` so a caller matching several output
        levels (main + aux + enc) can enqueue the next level's cost on the
        device before draining the current one's ``.cpu()`` transfer, paying
        one overlapped drain per level instead of a full pipeline stall (see
        ``DFINECriterion.forward``).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha)
                * (out_prob**self.gamma)
                * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # The broadcast form is bit-identical to ``torch.cdist(out_bbox,
        # tgt_bbox, p=1)`` but avoids cdist's slow one-thread-per-pair p=1
        # CUDA kernel (issue #763; measured on rfdetr in PR #761).
        cost_bbox = (out_bbox[:, None, :] - tgt_bbox[None, :, :]).abs().sum(-1)

        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1)

        if (self.cost_mask > 0 or self.cost_mask_dice > 0) and "pred_masks" in outputs:
            pred_masks = outputs["pred_masks"]
            has_any_masks = any(
                "masks" in t and t["masks"] is not None and t["masks"].numel() > 0
                for t in targets
            )
            if pred_masks is not None and has_any_masks:
                _, q_mask, hm, wm = pred_masks.shape
                if q_mask != num_queries:
                    dn_num = q_mask - num_queries
                    if dn_num > 0:
                        pred_masks = pred_masks[:, dn_num:]

                sizes_local = [len(v["boxes"]) for v in targets]
                offset = 0
                for b in range(bs):
                    n_tgt = sizes_local[b]
                    if n_tgt == 0:
                        continue
                    target = targets[b]
                    if (
                        "masks" not in target
                        or target["masks"] is None
                        or target["masks"].numel() == 0
                    ):
                        offset += n_tgt
                        continue

                    tgt_masks = target["masks"].float().to(pred_masks.device)
                    if tgt_masks.shape[-2:] != (hm, wm):
                        tgt_masks = F.interpolate(
                            tgt_masks.unsqueeze(1),
                            size=(hm, wm),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(1)

                    pred_probs = pred_masks[b].sigmoid()
                    mask_cost = pred_probs.new_zeros((num_queries, n_tgt))
                    if self.cost_mask_dice > 0:
                        mask_cost = mask_cost + self.cost_mask_dice * dice_cost(
                            pred_probs, tgt_masks
                        )
                    if self.cost_mask > 0:
                        mask_cost = mask_cost + self.cost_mask * sigmoid_focal_cost(
                            pred_masks[b].flatten(1),
                            tgt_masks.flatten(1),
                            alpha=self.alpha,
                            gamma=self.gamma,
                        )
                    C[b, :, offset : offset + n_tgt] = (
                        C[b, :, offset : offset + n_tgt] + mask_cost
                    )
                    offset += n_tgt

        return C

    @torch.no_grad()
    def solve(self, cost_matrix, targets, return_topk=False):
        """Run the Hungarian assignment on a CPU cost matrix.

        ``cost_matrix`` is the [bs, num_queries, total_targets] output of
        :meth:`compute_cost_matrix` after ``.cpu()``.
        """
        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(cost_matrix, nan=1.0)
        indices_pre = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices_pre
        ]

        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        cost_matrix = self.compute_cost_matrix(outputs, targets)
        return self.solve(cost_matrix.cpu(), targets, return_topk=return_topk)

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (
                        torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64),
                    )
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        return indices_list
