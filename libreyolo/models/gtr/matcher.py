# Adapted from GTR 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Modified local imports and distributed validation normalization. See NOTICE.
"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)
from .segmentation_head import point_sample


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        mask_point_sample_ratio=None,
    ):
        """Creates the matcher

        Params:
            cost_class: Relative weight of the classification error in the matching cost
            cost_bbox:  Relative weight of the L1 error of the bounding box coordinates
            cost_giou:  Relative weight of the giou loss of the bounding box
            mask_point_sample_ratio: enables mask BCE/Dice costs (point-sampled) when set
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.mask_point_sample_ratio = mask_point_sample_ratio
        if self.mask_point_sample_ratio:
            self.cost_mask_ce = weight_dict.get("cost_mask_ce", 0)
            self.cost_mask_dice = weight_dict.get("cost_mask_dice", 0)
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], targets, return_topk=False, group_detr=1
    ):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # We flatten to compute the cost matrices in a batch. For focal loss the
        # class cost only needs the target classes present in this mini-batch, so
        # gather first and avoid sigmoid over every Objects365 class.
        pred_logits = outputs["pred_logits"].flatten(0, 1)
        if self.use_focal_loss:
            out_prob = pred_logits[:, tgt_ids].sigmoid()
        else:
            out_prob = pred_logits.softmax(
                -1
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
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

        # Compute the L1 cost between boxes. For 4D boxes, explicit elementwise
        # L1 is much faster than torch.cdist(p=1) on CUDA while matching it
        # numerically up to normal fp32 rounding.
        if os.environ.get("GTR_FAST_L1_COST", "1") == "1":
            cost_bbox = (out_bbox[:, None, :] - tgt_bbox[None, :, :]).abs().sum(-1)
        else:
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Mask costs (instance segmentation): point-sampled BCE / Dice, same as GTR.
        masks_present = (
            self.mask_point_sample_ratio is not None
            and "pred_masks" in outputs
            and outputs["pred_masks"] is not None
            and len(targets) > 0
            and "masks" in targets[0]
        )
        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])
            out_masks = outputs["pred_masks"].flatten(0, 1)
            num_points = (
                out_masks.shape[-2]
                * out_masks.shape[-1]
                // self.mask_point_sample_ratio
            )
            tgt_masks = tgt_masks.to(out_masks.dtype)

            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(
                out_masks.unsqueeze(1),
                point_coords.repeat(out_masks.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            tgt_masks_flat = point_sample(
                tgt_masks.unsqueeze(1),
                point_coords.repeat(tgt_masks.shape[0], 1, 1),
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        if masks_present:
            C = (
                C
                + self.cost_mask_ce * cost_mask_ce
                + self.cost_mask_dice * cost_mask_dice
            )

        sizes = [len(v["boxes"]) for v in targets]
        # Group DETR: split cost matrix into group_detr chunks and match each independently.
        # Each group has g_num_queries = num_queries // group_detr queries.
        # Query indices for group g are offset by g * g_num_queries.
        g_num_queries = num_queries // group_detr

        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)
        C_groups = C.split(g_num_queries, dim=1)
        indices = None
        for g_i, C_g in enumerate(C_groups):
            indices_g_pre = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))
            ]
            indices_g = [
                (
                    torch.as_tensor(i + g_i * g_num_queries, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices_g_pre
            ]
            if indices is None:
                indices = indices_g
            else:
                indices = [
                    (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                    for idx1, idx2 in zip(indices, indices_g)
                ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C,
                    sizes=sizes,
                    k=return_topk,
                    initial_indices=[
                        (idx[0].numpy(), idx[1].numpy()) for idx in indices
                    ],
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
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
        # C.copy_(C_original)
        return indices_list
