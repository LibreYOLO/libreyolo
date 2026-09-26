# Adapted from GTR 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Modified local imports and distributed validation normalization. See NOTICE.
"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import os

import torch
import torch.distributed
import torch.nn.functional as F
import torchvision
from torch import nn

from ..dfine.loss import _get_world_size as get_world_size
from ..dfine.loss import (
    _is_dist_available_and_initialized as is_dist_available_and_initialized,
)
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    generalized_box_iou,
    paired_box_iou,
    paired_generalized_box_iou,
)
from .segmentation_head import get_uncertain_point_coords_with_randomness, point_sample
from .utils import bbox2distance


class GTRCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        group_detr=1,
        mask_point_sample_ratio=None,
        distributed_normalize=True,
    ):
        super().__init__()
        self.distributed_normalize = distributed_normalize
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.group_detr = group_detr
        # Prefer matcher's setting if it has one, else fall back to ctor arg.
        self.mask_point_sample_ratio = (
            getattr(matcher, "mask_point_sample_ratio", None) or mask_point_sample_ratio
        )
        self._loss_meta_cache = {}

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
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

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
                ious, _ = paired_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            else:
                ious, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
                ious = torch.diag(ious)
            ious = ious.detach()
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
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
                ious, _ = paired_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            else:
                ious, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
                ious = torch.diag(ious)
            ious = ious.detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        if os.getenv("GTR_FAST_MAL_LOSS", "0") == "1":
            pred_score = F.sigmoid(src_logits).detach()
            neg_weight = pred_score.pow(self.gamma)
            if self.mal_alpha is not None:
                neg_weight = self.mal_alpha * neg_weight
            loss = F.softplus(src_logits) * neg_weight

            if target_classes_o.numel() > 0:
                matched_logits = src_logits[idx]
                matched_neg = loss[idx]
                arange = torch.arange(
                    target_classes_o.numel(), device=src_logits.device
                )
                pos_logits = matched_logits[arange, target_classes_o]
                pos_neg_loss = matched_neg[arange, target_classes_o]
                target_score = ious.to(src_logits.dtype).pow(self.gamma)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, target_score, reduction="none"
                )
                total_loss = loss.sum() - pos_neg_loss.sum() + pos_loss.sum()
            else:
                total_loss = loss.sum()

            return {"loss_mal": total_loss / num_boxes}

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
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
            loss_giou = 1 - paired_generalized_box_iou(
                src_boxes_xyxy, target_boxes_xyxy
            )
        else:
            loss_giou = 1 - torch.diag(
                generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """BCE-with-logits + Dice losses on matched masks. Uses point-rend style sampling.
        Expects outputs['pred_masks'] of shape [B, Q, H, W] and targets[i]['masks'] of shape [Ti, Ht, Wt].
        """
        assert "pred_masks" in outputs and outputs["pred_masks"] is not None, (
            "pred_masks missing"
        )
        pred_masks = outputs["pred_masks"]  # [B, Q, H, W]
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H, W]

        if src_masks.numel() == 0:
            return {
                "loss_mask_ce": src_masks.sum(),
                "loss_mask_dice": src_masks.sum(),
            }

        target_masks = torch.cat(
            [t["masks"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )
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
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        point_logits = point_sample(
            src_masks, point_coords, align_corners=False
        ).squeeze(1)

        # NOTE: plain (non-jit) loss fns: num_boxes is a Tensor when GTR_TENSOR_LOSS_NORM=1,
        # which torch.jit.script's `num_masks: float` signature would reject.
        losses = {
            "loss_mask_ce": sigmoid_ce_loss(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_boxes),
        }
        del src_masks, target_masks
        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, values=None):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

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
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            if values is None:
                src_boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"][idx])
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
                if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
                    ious, _ = paired_box_iou(src_boxes_xyxy, target_boxes_xyxy)
                else:
                    ious = torch.diag(box_iou(src_boxes_xyxy, target_boxes_xyxy)[0])
            else:
                ious = values
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
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
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(
                        weight_targets_local[idx]
                    ).to(weight_targets_local.dtype)
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1)
                        .repeat(1, 1, 4)
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
                    if os.getenv("GTR_SYNC_FREE_DDF", "1") == "1":
                        mask_f = mask.to(loss_match_local.dtype)
                        pos_count = mask_f.sum().clamp_min(1)
                        neg_mask_f = 1 - mask_f
                        neg_count = neg_mask_f.sum().clamp_min(1)
                        loss_match_local1 = (
                            loss_match_local * mask_f
                        ).sum() / pos_count
                        loss_match_local2 = (
                            loss_match_local * neg_mask_f
                        ).sum() / neg_count
                    else:
                        loss_match_local1 = (
                            loss_match_local[mask].mean() if mask.any() else 0
                        )
                        loss_match_local2 = (
                            loss_match_local[~mask].mean() if (~mask).any() else 0
                        )
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos
                        + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        if os.getenv("GTR_FAST_GO_INDICES", "1") == "1":
            return self._get_go_indices_fast(indices, indices_aux_list)

        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [
            torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices
        ]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _get_go_indices_fast(self, indices, indices_aux_list):
        """Equivalent GO-index union with cheaper 1D unique keys.

        The legacy path does torch.unique(..., dim=0) on [row, col] pairs and
        then keeps the highest-count col per row. Keep the same count argsort
        tie behavior by applying torch.argsort to the same count vector order.
        """
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
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self._loss_meta_cache = {}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        share_matched_indices = (
            self.share_matched_indices
            or os.getenv("GTR_SHARE_MATCHED_INDICES", "0") == "1"
        )

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
            if share_matched_indices:
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
            if os.getenv("GTR_TENSOR_LOSS_NORM", "1") == "1":
                num_boxes_go = torch.clamp(
                    num_boxes_go
                    / (get_world_size() if self.distributed_normalize else 1),
                    min=1,
                )
            else:
                num_boxes_go = torch.clamp(
                    num_boxes_go
                    / (get_world_size() if self.distributed_normalize else 1),
                    min=1,
                ).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes.
        # Scale by group_detr so each group contributes equally (matches RF-DETR's criterion).
        num_boxes = sum(len(t["labels"]) for t in targets) * group_detr
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if self.distributed_normalize and is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        if os.getenv("GTR_TENSOR_LOSS_NORM", "1") == "1":
            num_boxes = torch.clamp(
                num_boxes / (get_world_size() if self.distributed_normalize else 1),
                min=1,
            )
        else:
            num_boxes = torch.clamp(
                num_boxes / (get_world_size() if self.distributed_normalize else 1),
                min=1,
            ).item()

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

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if "local" in self.losses:  # only work for local loss
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

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
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

        # In case of encoder auxiliary losses.
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

            num_boxes_enc = num_boxes
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    if loss == "masks":
                        continue  # encoder stage has no mask predictions
                    use_uni_set = (
                        self.use_uni_set and (loss == "boxes") and group_detr == 1
                    )
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes_enc
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

        # In case of cdn auxiliary losses.
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            if isinstance(dn_num_boxes, torch.Tensor):
                dn_num_boxes = dn_num_boxes.clamp_min(1)
            else:
                dn_num_boxes = 1 if dn_num_boxes == 0 else dn_num_boxes
            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                if "local" in self.losses:  # only work for local loss
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

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
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

        # Legacy behavior masks NaN losses to 0. Fast runs can disable it and
        # let the outer finite-loss check fail loudly while avoiding many tiny
        # scalar nan_to_num kernels.
        if os.getenv("GTR_SANITIZE_LOSSES", "1") == "1":
            losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        needs_matched_iou = loss in ("vfl", "mal", "local")
        if self.boxes_weight_format is None and not needs_matched_iou:
            return {}

        cache_key = (id(outputs), id(indices), self.boxes_weight_format or "iou")
        iou = self._loss_meta_cache.get(cache_key)
        if iou is None:
            src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
            target_boxes = torch.cat(
                [t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0
            )

            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes.detach())
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

            if self.boxes_weight_format in (None, "iou"):
                if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
                    iou, _ = paired_box_iou(src_boxes_xyxy, target_boxes_xyxy)
                else:
                    iou, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
                    iou = torch.diag(iou)
                iou = iou.detach()
            elif self.boxes_weight_format == "giou":
                if os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1") == "1":
                    iou = paired_generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
                else:
                    iou = torch.diag(
                        generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)
                    )
                iou = iou.detach()
            else:
                raise AttributeError()
            self._loss_meta_cache[cache_key] = iou

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl", "mal", "local"):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = (
            dn_meta["dn_positive_idx"],
            dn_meta["dn_num_group"],
        )
        # Group DETR replicates the whole DN block once per group, so gt_idx must be
        # tiled an extra ``dn_group_detr`` times to match the layout of dn_positive_idx.
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

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

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

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = (
            [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        )
        return opt_list


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks
