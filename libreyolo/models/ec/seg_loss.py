"""ECSegCriterion — EC criterion extended with an instance-mask loss.

Builds on :class:`ECCriterion` (which itself extends D-FINE's box/class/local
criterion with the MAL classification loss) and adds a point-sampled
BCE + Dice mask loss on the Hungarian-matched queries.

The mask machinery (uncertainty-based point sampling, dice/sigmoid-CE) is
reused verbatim from LibreYOLO's RF-DETR seg port (Apache-2.0, the same head
EC's :class:`SegmentationHead` is derived from) so the two seg families share
one tested implementation. EC's seg head emits masks in the *deferred* form
(``{spatial_features, query_features, bias}``) during training; the matched
masks are materialized here for the handful of matched queries only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..dfine.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from ..rfdetr.box_ops import batch_dice_loss, batch_sigmoid_ce_loss
from ..rfdetr.loss import dice_loss_jit, sigmoid_ce_loss_jit
from ..rfdetr.segmentation import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .loss import ECCriterion


class ECSegHungarianMatcher(nn.Module):
    """EC matcher with EdgeCrafter's mask-aware segmentation costs.

    The return value intentionally matches D-FINE's matcher contract
    (``{"indices": ...}``) so :class:`ECCriterion` can keep its shared forward
    path, while the mask cost follows EdgeCrafter/RF-DETR's point-sampled
    CE+Dice assignment recipe.
    """

    def __init__(
        self,
        weight_dict,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        mask_point_sample_ratio: int | None = 16,
    ):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_mask_ce = weight_dict.get("cost_mask_ce", 0.0)
        self.cost_mask_dice = weight_dict.get("cost_mask_dice", 0.0)
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.mask_point_sample_ratio = mask_point_sample_ratio

        if not any(
            (
                self.cost_class,
                self.cost_bbox,
                self.cost_giou,
                self.cost_mask_ce,
                self.cost_mask_dice,
            )
        ):
            raise ValueError("at least one matching cost must be non-zero")

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(v["boxes"]) for v in targets]
        if sum(sizes) == 0:
            empty = [
                (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
                for _ in targets
            ]
            return {"indices": empty}

        flat_logits = outputs["pred_logits"].flatten(0, 1)
        if self.use_focal_loss:
            out_prob = flat_logits.sigmoid()
        else:
            out_prob = flat_logits.softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_focal_loss:
            tgt_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha)
                * (tgt_prob**self.gamma)
                * (-(1 - tgt_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha
                * ((1 - tgt_prob) ** self.gamma)
                * (-(tgt_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        cost_matrix = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )

        masks_present = (
            self.mask_point_sample_ratio
            and "pred_masks" in outputs
            and targets
            and "masks" in targets[0]
        )
        if masks_present:
            cost_mask_ce, cost_mask_dice = self._mask_costs(outputs["pred_masks"], targets)
            cost_matrix = (
                cost_matrix
                + self.cost_mask_ce * cost_mask_ce
                + self.cost_mask_dice * cost_mask_dice
            )

        cost_matrix = torch.nan_to_num(
            cost_matrix.view(bs, num_queries, -1).float().cpu(),
            nan=1.0,
            posinf=1e6,
            neginf=1e6,
        )
        indices_pre = [
            linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))
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
                    cost_matrix, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }
        return {"indices": indices}

    def _mask_costs(self, pred_masks, targets):
        tgt_masks = torch.cat([v["masks"] for v in targets])

        if isinstance(pred_masks, torch.Tensor):
            out_masks = pred_masks.flatten(0, 1)
            num_points = max(
                1,
                out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio,
            )
            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(
                out_masks.unsqueeze(1),
                point_coords.repeat(out_masks.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
        else:
            spatial_features = pred_masks["spatial_features"]
            query_features = pred_masks["query_features"]
            bias = pred_masks["bias"]
            num_points = max(
                1,
                spatial_features.shape[-2]
                * spatial_features.shape[-1]
                // self.mask_point_sample_ratio,
            )
            point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
            sampled_spatial = point_sample(
                spatial_features,
                point_coords.repeat(spatial_features.shape[0], 1, 1),
                align_corners=False,
            )
            pred_masks_logits = (
                torch.einsum("bcp,bnc->bnp", sampled_spatial, query_features) + bias
            ).flatten(0, 1)

        tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
        tgt_masks_flat = point_sample(
            tgt_masks.unsqueeze(1),
            point_coords.repeat(tgt_masks.shape[0], 1, 1),
            align_corners=False,
            mode="nearest",
        ).squeeze(1)

        return (
            batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat),
            batch_dice_loss(pred_masks_logits, tgt_masks_flat),
        )

    def get_top_k_matches(self, cost_matrix, sizes, k=1, initial_indices=None):
        indices_list = []
        for i in range(k):
            indices_k = (
                [
                    linear_sum_assignment(c[i])
                    for i, c in enumerate(cost_matrix.split(sizes, -1))
                ]
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
            for c, idx_k in zip(cost_matrix.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        return [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]


class ECSegCriterion(ECCriterion):
    """ECCriterion + instance-segmentation mask loss (``loss_mask_ce`` / ``loss_mask_dice``)."""

    def __init__(self, *args, mask_point_sample_ratio: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_point_sample_ratio = mask_point_sample_ratio

    def loss_masks(self, outputs, targets, indices, num_boxes, **kwargs):
        """Point-sampled BCE + Dice on matched masks.

        ``outputs["pred_masks"]`` is the deferred mask form
        ``{spatial_features: (B, C, Hm, Wm), query_features: (B, N, C),
        bias: (1,)}``. Masks for matched queries are computed on the fly via
        einsum, mirroring RF-DETR's ``loss_masks`` sparse branch.
        """
        # The pre/enc auxiliary outputs carry no masks — skip cleanly so the
        # shared D-FINE forward loop can still call ``loss_masks`` for them.
        if "pred_masks" not in outputs:
            return {}

        pred_masks = outputs["pred_masks"]
        idx = self._get_src_permutation_idx(indices)

        if isinstance(pred_masks, torch.Tensor):
            src_masks = pred_masks[idx]
        else:
            spatial_features = pred_masks["spatial_features"]
            query_features = pred_masks["query_features"]
            bias = pred_masks["bias"]
            if idx[0].numel() == 0:
                # Keep every mask-head parameter in the autograd graph (a bare
                # empty tensor has no grad_fn → breaks DDP static_graph). Return
                # zeros that still depend on the head's tensors.
                zero = (
                    spatial_features.sum() * 0.0
                    + query_features.sum() * 0.0
                    + bias.sum() * 0.0
                )
                return {"loss_mask_ce": zero, "loss_mask_dice": zero}
            selected = []
            per_batch_counts = idx[0].unique(return_counts=True)[1]
            batch_indices = torch.cat(
                (torch.zeros_like(per_batch_counts[:1]), per_batch_counts), dim=0
            ).cumsum(0)
            for i in range(per_batch_counts.shape[0]):
                batch_indicator = idx[0][batch_indices[i] : batch_indices[i + 1]]
                box_indicator = idx[1][batch_indices[i] : batch_indices[i + 1]]
                this_queries = query_features[(batch_indicator, box_indicator)]
                this_spatial = spatial_features[idx[0][batch_indices[i + 1] - 1]]
                this_masks = (
                    torch.einsum("chw,nc->nhw", this_spatial, this_queries) + bias
                )
                selected.append(this_masks)
            src_masks = torch.cat(selected)

        if src_masks.numel() == 0:
            return {
                "loss_mask_ce": src_masks.sum(),
                "loss_mask_dice": src_masks.sum(),
            }

        target_masks = torch.cat(
            [t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )

        # Normalized point coordinates make the pred (imgsz/downsample) and
        # target (imgsz) resolutions interchangeable — no upsampling needed.
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(
            src_masks.shape[-2],
            src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio,
        )

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )

        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        with torch.no_grad():
            point_labels = point_sample(
                target_masks, point_coords, align_corners=False, mode="nearest"
            ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }
        del src_masks, target_masks
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "masks":
            return self.loss_masks(outputs, targets, indices, num_boxes, **kwargs)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
