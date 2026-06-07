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

import torch

from ..rfdetr.loss import dice_loss_jit, sigmoid_ce_loss_jit
from ..rfdetr.segmentation import (
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .loss import ECCriterion


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
