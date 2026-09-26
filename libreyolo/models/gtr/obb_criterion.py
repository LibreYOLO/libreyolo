# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT):
# engine/gtr/obb/criterion.py and engine/gtr/obb/matcher.py.
# Changes: local imports, no registry, per-rank normalization switch for
# validation loss. See NOTICE.
"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Criterion and Hungarian matcher for the OBB task.

Losses follow the paper: classification (MAL / focal, soft target weighted by
the IoU of the enclosing horizontal boxes as in ai4rs), L1 on the normalized
5-dim rbox, KLD loss (gaussian, log1p/tau=1) instead of GIoU, and the
Fine-Grained Localization loss over the 6 ADR distributions. Matching cost is
focal classification + Chamfer distance (corner sets) + KLD.
"""

import copy
import os
from typing import Dict

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

from ..dfine.loss import _get_world_size as get_world_size
from ..dfine.loss import (
    _is_dist_available_and_initialized as is_dist_available_and_initialized,
)
from .box_ops import paired_box_iou
from .obb_decoder import NUM_DIST
from .obb_rbox import (
    chamfer_cost_pairwise,
    kld_cost_pairwise,
    kld_loss_paired,
    rbox2distance,
    rbox_norm_to_rad,
    rbox_to_hbox_xyxy,
)


class OBBHungarianMatcher(nn.Module):
    """Computes 1-to-1 assignment between oriented-box predictions and targets."""

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_chamfer = weight_dict["cost_chamfer"]
        self.cost_kld = weight_dict["cost_kld"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_chamfer != 0 or self.cost_kld != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, group_detr=1):
        """
        Args:
            outputs: dict with "pred_logits" [bs, q, num_classes], "pred_boxes" [bs, q, 5]
                     (sigmoid-domain oriented boxes).
            targets: list of dicts with "labels" [n] and "boxes" [n, 5] (sigmoid-domain).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        pred_logits = outputs["pred_logits"].flatten(0, 1)
        if self.use_focal_loss:
            out_prob = pred_logits[:, tgt_ids].sigmoid()
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
            out_prob = pred_logits.softmax(-1)
            cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*q, 5]

        out_rad = rbox_norm_to_rad(out_bbox)
        tgt_rad = rbox_norm_to_rad(tgt_bbox)
        cost_chamfer = chamfer_cost_pairwise(out_rad, tgt_rad)
        cost_kld = kld_cost_pairwise(out_rad, tgt_rad)

        C = (
            self.cost_chamfer * cost_chamfer
            + self.cost_class * cost_class
            + self.cost_kld * cost_kld
        )

        sizes = [len(v["boxes"]) for v in targets]
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

        return {"indices": indices}


def paired_hbox_iou_of_rbox(
    src_rbox: torch.Tensor, tgt_rbox: torch.Tensor
) -> torch.Tensor:
    """Paired IoU of the enclosing horizontal boxes of two sigmoid-domain rbox sets."""
    src_xyxy = rbox_to_hbox_xyxy(rbox_norm_to_rad(src_rbox))
    tgt_xyxy = rbox_to_hbox_xyxy(rbox_norm_to_rad(tgt_rbox))
    iou, _ = paired_box_iou(src_xyxy, tgt_xyxy)
    return iou


class OBBGTRCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=15,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        group_detr=1,
        angle_periodic_l1=False,
        distributed_normalize=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        # Shortest-path Periodic L1 Loss: measure the angle dim on the circle so the
        # 0<->1 seam (a = theta/pi) is not over-penalized (paper Sec 3.3). Off -> plain L1.
        self.angle_periodic_l1 = angle_periodic_l1
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.group_detr = group_detr
        self._loss_meta_cache = {}
        # LibreYOLO: validation loss runs per rank without a collective.
        self.distributed_normalize = distributed_normalize

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        if src_logits.shape[1] == 0:
            # 0-query DN branch (the whole batch lost its GT to augmentation):
            # `.mean(1)` over the empty query dim would yield NaN.
            return {"loss_focal": src_logits.sum()}
        idx = self._get_src_permutation_idx(indices)
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
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        if outputs["pred_logits"].shape[1] == 0:
            # 0-query DN branch (the whole batch lost its GT to augmentation):
            # `.mean(1)` over the empty query dim would yield NaN.
            return {"loss_mal": outputs["pred_logits"].sum()}
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            ious = paired_hbox_iou_of_rbox(src_boxes, target_boxes).detach()
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
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """L1 on the normalized 5-dim rbox + KLD loss (replaces GIoU)."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        losses = {}
        loss_bbox = (src_boxes - target_boxes).abs()
        if self.angle_periodic_l1:
            # a = theta / pi in [0, 1) with period 1: take the shorter arc on the circle.
            ang = loss_bbox[..., 4]
            loss_bbox = torch.cat(
                [loss_bbox[..., :4], torch.minimum(ang, 1.0 - ang).unsqueeze(-1)],
                dim=-1,
            )
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_kld = kld_loss_paired(
            rbox_norm_to_rad(src_boxes), rbox_norm_to_rad(target_boxes)
        )
        loss_kld = loss_kld if boxes_weight is None else loss_kld * boxes_weight
        losses["loss_kld"] = loss_kld.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, values=None):
        """Fine-Grained Localization (FGL) over 6 ADR distributions + DDF distillation."""
        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = rbox2distance(
                        ref_points,
                        target_boxes,
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = rbox2distance(
                        ref_points,
                        target_boxes,
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            if values is None:
                ious = paired_hbox_iou_of_rbox(outputs["pred_boxes"][idx], target_boxes)
            else:
                ious = values
            weight_targets = (
                ious.unsqueeze(-1).repeat(1, 1, NUM_DIST).reshape(-1).detach()
            )

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            # DDF self-distillation: the paper does not use it ("The distillation
            # loss is not used"); computed only when a weight is configured.
            if "loss_ddf" in self.weight_dict and "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(
                    -1, (self.reg_max + 1)
                )
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = (
                        outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                    )

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, NUM_DIST).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(
                        weight_targets_local[idx]
                    ).to(weight_targets_local.dtype)
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1)
                        .repeat(1, 1, NUM_DIST)
                        .reshape(-1)
                        .detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    mask_f = mask.to(loss_match_local.dtype)
                    pos_count = mask_f.sum().clamp_min(1)
                    neg_mask_f = 1 - mask_f
                    neg_count = neg_mask_f.sum().clamp_min(1)
                    loss_match_local1 = (loss_match_local * mask_f).sum() / pos_count
                    loss_match_local2 = (
                        loss_match_local * neg_mask_f
                    ).sum() / neg_count
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos
                        + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _world_size(self):
        return get_world_size() if self.distributed_normalize else 1

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        layers = (indices, *indices_aux_list)
        for batch_i in range(len(indices)):
            rows = torch.cat([layer[batch_i][0] for layer in layers])
            cols = torch.cat([layer[batch_i][1] for layer in layers])
            if rows.numel() == 0:
                results.append((rows.long(), cols.long()))
                continue

            key_base = int(cols.max().item()) + 1
            keys = rows.long() * key_base + cols.long()
            unique_keys, counts = torch.unique(keys, sorted=True, return_counts=True)
            unique_sorted = unique_keys[torch.argsort(counts, descending=True)]
            row_sorted = torch.div(unique_sorted, key_base, rounding_mode="floor")
            col_sorted = unique_sorted.remainder(key_base)

            seen_rows = set()
            final_rows, final_cols = [], []
            for row_idx, col_idx in zip(row_sorted.tolist(), col_sorted.tolist()):
                if row_idx not in seen_rows:
                    seen_rows.add(row_idx)
                    final_rows.append(row_idx)
                    final_cols.append(col_idx)

            results.append(
                (
                    torch.as_tensor(final_rows, dtype=torch.int64, device=rows.device),
                    torch.as_tensor(final_cols, dtype=torch.int64, device=cols.device),
                )
            )
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self._loss_meta_cache = {}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)[
            "indices"
        ]
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs["aux_outputs"]
            if "pre_outputs" in outputs:
                aux_outputs_list = outputs["aux_outputs"] + [outputs["pre_outputs"]]
            if self.share_matched_indices:
                cached_indices = [indices for _ in aux_outputs_list]
                cached_indices_enc = [indices for _ in outputs["enc_aux_outputs"]]
                indices_go = indices
            else:
                for i, aux_outputs in enumerate(aux_outputs_list):
                    indices_aux = self.matcher(
                        aux_outputs, targets, group_detr=group_detr
                    )["indices"]
                    cached_indices.append(indices_aux)
                    indices_aux_list.append(indices_aux)
                for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                    indices_enc = self.matcher(
                        aux_outputs, targets, group_detr=group_detr
                    )["indices"]
                    cached_indices_enc.append(indices_enc)
                    indices_aux_list.append(indices_enc)
                indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if self.distributed_normalize and is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / self._world_size(), min=1)
        else:
            assert "aux_outputs" in outputs, ""

        num_boxes = sum(len(t["labels"]) for t in targets) * group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if self.distributed_normalize and is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / self._world_size(), min=1)

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(
                loss, outputs, targets, indices_in, num_boxes_in, **meta
            )
            l_dict = {
                k: l_dict[k] * self.weight_dict[k]
                for k in l_dict
                if k in self.weight_dict
            }
            losses.update(l_dict)

        # Auxiliary losses of each intermediate decoder layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if "local" in self.losses:
                    aux_outputs["up"], aux_outputs["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_in
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Auxiliary traditional head output at the first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(
                    loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                )

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k]
                    for k in l_dict
                    if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # Encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    if loss == "local":
                        continue  # encoder stage has no distribution predictions
                    use_uni_set = (
                        self.use_uni_set and (loss == "boxes") and group_detr == 1
                    )
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, enc_targets, indices_in
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # Denoising auxiliary losses.
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes.clamp_min(1)
            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:
                    aux_outputs["is_dn"] = True
                    aux_outputs["up"], aux_outputs["reg_scale"] = (
                        outputs["up"],
                        outputs["reg_scale"],
                    )
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(
                        loss, aux_outputs, targets, indices_dn
                    )
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k]
                        for k in l_dict
                        if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if os.getenv("GTR_SANITIZE_LOSSES", "1") == "1":
            losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        needs_matched_iou = loss in ("mal", "local")
        if self.boxes_weight_format is None and not needs_matched_iou:
            return {}

        cache_key = (id(outputs), id(indices))
        iou = self._loss_meta_cache.get(cache_key)
        if iou is None:
            src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
            target_boxes = torch.cat(
                [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )
            iou = paired_hbox_iou_of_rbox(src_boxes.detach(), target_boxes).detach()
            self._loss_meta_cache[cache_key] = iou

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("mal", "local"):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = (
            dn_meta["dn_positive_idx"],
            dn_meta["dn_num_group"],
        )
        dn_group_detr = dn_meta.get("dn_group_detr", 1)
        effective_tile = dn_num_group * dn_group_detr
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(effective_tile)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def unimodal_distribution_focal_loss(
        self,
        pred,
        label,
        weight_right,
        weight_left,
        weight=None,
        reduction="sum",
        avg_factor=None,
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(
            -1
        )

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
