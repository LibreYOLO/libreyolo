"""ECCriterion — D-FINE-style criterion with the upstream MAL classification loss.

EC's classification loss in the released YAML is ``mal`` (Modified Align
Loss). LibreYOLO's D-FINE port only carries ``focal`` / ``vfl``; the MAL form is
ported here so the EC recipe can be reproduced.

Other losses (``focal``, ``vfl``, ``boxes``, ``local`` = FGL+DDF) are inherited
from ``DFINECriterion`` unchanged. The forward path is identical: D-FINE's hard-
coded ``if loss in ["boxes", "local"]`` for the cross-layer match-union is
exactly what upstream means by ``use_uni_set=True``, which is the only setting
the released configs use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..dfine.box_ops import box_cxcywh_to_xyxy, box_iou
from ..dfine.loss import DFINECriterion


class ECCriterion(DFINECriterion):
    """D-FINE-style criterion with the upstream MAL classification loss.

    Adds ``loss_labels_mal`` and routes ``"mal"`` in the loss map. ``mal_alpha``
    is exposed as an optional override; when ``None`` the unweighted-positive
    form (matching upstream's default) is used.
    """

    def __init__(self, *args, mal_alpha: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mal_alpha = mal_alpha

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """Modified Align Loss (MAL).

        Differs from VFL in two places:

        1. The positive target score is raised to gamma before being used as
           the BCE target (``target_score = target_score ** gamma``).
        2. The negative weight is the un-alpha-scaled prediction
           ``pred_score**gamma * (1 - target)`` (i.e., MAL drops the alpha=0.75
           that VFL applies). When ``mal_alpha`` is set, that alpha is
           re-applied.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        # Extend D-FINE's IoU-pass-through to also feed MAL.
        if self.boxes_weight_format is None:
            return {}
        meta = super().get_loss_meta_info(loss, outputs, targets, indices)
        if loss == "mal" and "values" not in meta:
            src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
            target_boxes = torch.cat(
                [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            meta = {"values": torch.diag(iou)}
        return meta
