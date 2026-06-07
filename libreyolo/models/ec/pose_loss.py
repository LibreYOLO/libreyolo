"""ECPoseCriterion — faithful port of DETRPose's keypoint criterion (Apache-2.0).

EdgeCrafter's ECPose is a DETRPose model, so this mirrors DETRPose's released
recipe (``configs/detrpose/include/detrpose_hgnetv2.py``):

  losses      = ['vfl', 'keypoints']
  weight_dict = {'loss_vfl': 2.0, 'loss_keypoints': 10.0, 'loss_oks': 4.0}
  matcher     = HungarianMatcher(cost_class=2, cost_keypoints=10, cost_oks=4)
  + contrastive denoising (dn_number=20, label_noise_ratio=0.5)
  + GO-LSD union indices for the keypoint losses across decoder layers
  + deep supervision: decoder aux layers + encoder (interm) + pre-head + DN group

DETRPose's config does NOT enable the keypoint-DFL ``loss_local`` term, so it is
intentionally not used here either. Keypoints are the flattened interleaved form
``[x1,y1,...,xK,yK]`` and targets carry ``keypoints=[xy(2K) | vis(K)]`` plus a
normalized ``area`` and xyxy ``boxes`` (the latter used by the denoising group).

Lineage (all Apache-2.0): DETRPose criterion/matcher + COCO OKS metric; the
varifocal weight and focal matching cost are the generic DETR/RT-DETR/D-FINE
formulas already present in LibreYOLO's D-FINE port.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

COCO17_OKS_SIGMAS = (
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
)


def default_oks_sigmas(num_keypoints: int) -> List[float]:
    if num_keypoints == 17:
        return list(COCO17_OKS_SIGMAS)
    return [1.0 / num_keypoints] * num_keypoints


def _sigmas_for(num_keypoints: int, sigmas: Sequence[float] | None = None) -> np.ndarray:
    if sigmas is not None:
        if len(sigmas) != num_keypoints:
            raise ValueError(
                f"sigmas has {len(sigmas)} entries but num_keypoints={num_keypoints}"
            )
        return np.asarray([float(s) for s in sigmas], dtype=np.float32)
    if num_keypoints == 17:
        return np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
             1.07, .87, .87, .89, .89], dtype=np.float32) / 10.0
    if num_keypoints == 14:
        return np.array(
            [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
             .79, .79], dtype=np.float32) / 10.0
    return np.full((num_keypoints,), 1.0 / num_keypoints, dtype=np.float32)


def oks_overlaps(kpt_preds, kpt_gts, kpt_valids, kpt_areas, sigmas):
    """COCO Object Keypoint Similarity. Faithful port of DETRPose's helper.

    ``kpt_preds`` / ``kpt_gts`` are flattened ``(N, 2K)``; ``kpt_valids`` is
    ``(N, K)``; ``kpt_areas`` is ``(N,)`` (normalized box area, matching the
    normalized keypoint coordinates).
    """
    sigmas = kpt_preds.new_tensor(sigmas)
    variances = (sigmas * 2) ** 2
    kpt_preds = kpt_preds.reshape(-1, kpt_preds.size(-1) // 2, 2)
    kpt_gts = kpt_gts.reshape(-1, kpt_gts.size(-1) // 2, 2)
    squared_distance = (kpt_preds[:, :, 0] - kpt_gts[:, :, 0]) ** 2 + \
        (kpt_preds[:, :, 1] - kpt_gts[:, :, 1]) ** 2
    squared_distance0 = squared_distance / (kpt_areas[:, None] * variances[None, :] * 2)
    squared_distance1 = torch.exp(-squared_distance0) * kpt_valids
    oks = squared_distance1.sum(dim=1) / (kpt_valids.sum(dim=1) + 1e-6)
    return oks


class OKSLoss(nn.Module):
    """OKS overlap, ``linear`` form returns the similarity itself."""

    def __init__(
        self,
        num_keypoints: int = 17,
        linear: bool = True,
        eps: float = 1e-6,
        sigmas: Sequence[float] | None = None,
    ):
        super().__init__()
        self.linear = linear
        self.eps = eps
        self.sigmas = _sigmas_for(num_keypoints, sigmas)

    def forward(self, pred, target, valid, area):
        oks = oks_overlaps(pred, target, valid, area, self.sigmas).clamp(min=self.eps)
        return oks if self.linear else -oks.log()


class PoseHungarianMatcher(nn.Module):
    """Hungarian matcher: focal class cost + visibility-weighted keypoint L1 + OKS.

    Faithful to DETRPose's matcher (flattened keypoint format)."""

    def __init__(
        self,
        num_keypoints: int = 17,
        cost_class: float = 2.0,
        cost_keypoints: float = 10.0,
        cost_oks: float = 4.0,
        focal_alpha: float = 0.25,
        sigmas: Sequence[float] | None = None,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.cost_class = cost_class
        self.cost_keypoints = cost_keypoints
        self.cost_oks = cost_oks
        self.focal_alpha = focal_alpha
        self.sigmas = _sigmas_for(num_keypoints, sigmas)

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*Q, C)
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)     # (B*Q, 2K)

        sizes = [len(v["labels"]) for v in targets]
        if sum(sizes) == 0:
            return [
                (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
                for _ in targets
            ]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])  # (G, 2K+K)
        tgt_area = torch.cat([v["area"] for v in targets])            # (G,)

        alpha, gamma = self.focal_alpha, 2.0
        neg = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]  # (B*Q, G)

        K2 = self.num_keypoints * 2
        Z_pred = out_keypoints[:, 0:K2]
        Z_gt = tgt_keypoints[:, 0:K2]
        V_gt = tgt_keypoints[:, K2:]
        if Z_pred.sum() > 0:
            sigmas = Z_pred.new_tensor(self.sigmas)
            variances = (sigmas * 2) ** 2
            kp = Z_pred.reshape(-1, self.num_keypoints, 2)
            kg = Z_gt.reshape(-1, self.num_keypoints, 2)
            sq = (kp[:, None, :, 0] - kg[None, :, :, 0]) ** 2 + \
                 (kp[:, None, :, 1] - kg[None, :, :, 1]) ** 2  # (B*Q, G, K)
            sq0 = sq / (tgt_area[:, None] * variances[None, :] * 2)
            oks = (torch.exp(-sq0) * V_gt).sum(-1) / (V_gt.sum(-1) + 1e-6)
            cost_oks = 1 - oks.clamp(min=1e-6)
            cost_kpt = torch.abs(Z_pred[:, None, :] - Z_gt[None])  # (B*Q, G, 2K)
            cost_kpt = (cost_kpt * V_gt.repeat_interleave(2, dim=1)[None]).sum(-1)
            C = (
                self.cost_class * cost_class
                + self.cost_keypoints * cost_kpt
                + self.cost_oks * cost_oks
            )
        else:
            C = self.cost_class * cost_class
        C = torch.nan_to_num(C.view(bs, num_queries, -1), nan=1e4).cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


def _is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _world_size():
    return torch.distributed.get_world_size() if _is_dist() else 1


class ECPoseCriterion(nn.Module):
    """DETRPose criterion: VFL classification (OKS-keyed) + keypoint L1 + OKS,
    with GO-LSD union matching, deep supervision, and contrastive denoising."""

    def __init__(
        self,
        matcher: PoseHungarianMatcher,
        num_keypoints: int = 17,
        num_classes: int = 2,
        weight_dict: dict | None = None,
        losses: Sequence[str] = ("vfl", "keypoints"),
        focal_alpha: float = 0.25,
        gamma: float = 2.0,
        # accepted for trainer convenience; folded into weight_dict if given
        cls_loss_weight: float | None = None,
        keypoint_l1_loss_weight: float | None = None,
        oks_loss_weight: float | None = None,
        sigmas: Sequence[float] | None = None,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_keypoints = num_keypoints
        self.num_body_points = num_keypoints
        self.num_classes = num_classes
        self.losses = list(losses)
        self.focal_alpha = focal_alpha
        self.gamma = gamma
        if weight_dict is None:
            weight_dict = {
                "loss_vfl": cls_loss_weight if cls_loss_weight is not None else 2.0,
                "loss_keypoints": keypoint_l1_loss_weight if keypoint_l1_loss_weight is not None else 10.0,
                "loss_oks": oks_loss_weight if oks_loss_weight is not None else 4.0,
            }
        self.weight_dict = weight_dict
        self.oks = OKSLoss(num_keypoints=num_keypoints, linear=True, sigmas=sigmas)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _gather_kpt_targets(self, targets, indices):
        K2 = self.num_body_points * 2
        tk = torch.cat([t["keypoints"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ta = torch.cat([t["area"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return tk[:, 0:K2], tk[:, K2:], ta

    def loss_vfl(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(outputs["pred_logits"].device), idx[1].to(outputs["pred_logits"].device))
        src_kpts = outputs["pred_keypoints"][idx]
        Z_gt, V_gt, area = self._gather_kpt_targets(targets, indices)
        Z_gt, V_gt, area = Z_gt.to(src_kpts.device), V_gt.to(src_kpts.device), area.to(src_kpts.device)
        oks = self.oks(src_kpts[:, 0:self.num_body_points * 2], Z_gt, V_gt, area).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        ).to(src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = src_logits.sigmoid().detach()
        weight = self.focal_alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(outputs["pred_keypoints"].device), idx[1].to(outputs["pred_keypoints"].device))
        src_kpts = outputs["pred_keypoints"][idx]  # (N, 2K)
        device = outputs["pred_logits"].device
        if len(src_kpts) == 0:
            zero = torch.as_tensor(0.0, device=device) + src_kpts.sum() * 0
            return {"loss_keypoints": zero, "loss_oks": zero}
        Z_pred = src_kpts[:, 0:self.num_body_points * 2]
        Z_gt, V_gt, area = self._gather_kpt_targets(targets, indices)
        Z_gt, V_gt, area = Z_gt.to(device), V_gt.to(device), area.to(device)
        oks_loss = 1 - self.oks(Z_pred, Z_gt, V_gt, area)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction="none")
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        return {
            "loss_keypoints": pose_loss.sum() / num_boxes,
            "loss_oks": oks_loss.sum() / num_boxes,
        }

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        return {"vfl": self.loss_vfl, "keypoints": self.loss_keypoints}[loss](
            outputs, targets, indices, num_boxes
        )

    def _get_go_indices(self, indices, indices_aux_list):
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([i1[0], i2[0]]), torch.cat([i1[1], i2[1]]))
                for i1, i2 in zip(indices.copy(), indices_aux.copy())
            ]
        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            unique_sorted = unique[torch.argsort(counts, descending=True)]
            col_to_row = {}
            for u in unique_sorted:
                r, c = u[0].item(), u[1].item()
                if r not in col_to_row:
                    col_to_row[r] = c
            rows = torch.tensor(list(col_to_row.keys()), device=ind.device).long()
            cols = torch.tensor(list(col_to_row.values()), device=ind.device).long()
            results.append((rows, cols))
        return results

    @staticmethod
    def _prep_for_dn(dn_meta):
        n_groups, pad = dn_meta["num_dn_group"], dn_meta["pad_size"]
        return pad // n_groups, n_groups

    def _num_boxes(self, targets, device):
        n = sum(len(t["labels"]) for t in targets)
        n = torch.as_tensor([n], dtype=torch.float, device=device)
        if _is_dist():
            torch.distributed.all_reduce(n)
        return torch.clamp(n / _world_size(), min=1).item()

    def _num_index_pairs(self, indices, device):
        n = sum(len(x[0]) for x in indices)
        n = torch.as_tensor([n], dtype=torch.float, device=device)
        if _is_dist():
            torch.distributed.all_reduce(n)
        return torch.clamp(n / _world_size(), min=1).item()

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        indices = self.matcher(outputs_without_aux, targets)

        # GO-LSD union over decoder + pre + interm layers (for keypoint losses).
        cached_indices, cached_indices_enc, indices_aux_list = [], [], []
        if "aux_outputs" in outputs:
            for aux in outputs["aux_outputs"] + [outputs["aux_pre_outputs"]]:
                ia = self.matcher(aux, targets)
                cached_indices.append(ia)
                indices_aux_list.append(ia)
            for aux in outputs["aux_interm_outputs"]:
                ie = self.matcher(aux, targets)
                cached_indices_enc.append(ie)
                indices_aux_list.append(ie)
            indices_go = self._get_go_indices(indices, indices_aux_list)
            num_boxes_go = self._num_index_pairs(indices_go, device)
        else:
            indices_go = indices
            num_boxes_go = self._num_index_pairs(indices, device)

        num_boxes = self._num_boxes(targets, device)

        def _apply(out, tgts, idx_per_layer, suffix):
            d = {}
            for loss in self.losses:
                idx_in = indices_go if loss == "keypoints" else idx_per_layer
                nb_in = num_boxes_go if loss == "keypoints" else num_boxes
                l = self.get_loss(loss, out, tgts, idx_in, nb_in)
                l = {k: v * self.weight_dict[k] for k, v in l.items() if k in self.weight_dict}
                d.update({k + suffix: v for k, v in l.items()})
            return d

        losses = {}
        losses.update(_apply(outputs, targets, indices, ""))

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                losses.update(_apply(aux, targets, cached_indices[i], f"_aux_{i}"))
            losses.update(_apply(outputs["aux_pre_outputs"], targets, cached_indices[-1], "_pre"))
            for i, aux in enumerate(outputs["aux_interm_outputs"]):
                losses.update(_apply(aux, targets, cached_indices_enc[i], f"_enc_{i}"))

        # Contrastive denoising losses.
        if "dn_aux_outputs" in outputs:
            dn_meta = outputs["dn_meta"]
            single_pad, scalar = self._prep_for_dn(dn_meta)
            dn_pos_idx = []
            for i in range(len(targets)):
                n = len(targets[i]["labels"])
                if n > 0:
                    t = torch.arange(n, device=device).long().unsqueeze(0).repeat(scalar, 1)
                    out_idx = (
                        torch.arange(scalar, device=device).long().unsqueeze(1) * single_pad + t
                    ).flatten()
                    tgt_idx = t.flatten()
                else:
                    out_idx = tgt_idx = torch.zeros(0, dtype=torch.long, device=device)
                dn_pos_idx.append((out_idx, tgt_idx))

            dn_nb = num_boxes * scalar
            for i, aux in enumerate(outputs["dn_aux_outputs"]):
                losses.update(self._dn_apply(aux, targets, dn_pos_idx, dn_nb, f"_dn_{i}"))
            losses.update(
                self._dn_apply(outputs["dn_aux_pre_outputs"], targets, dn_pos_idx, dn_nb, "_dn_pre")
            )

        return {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}

    def _dn_apply(self, out, targets, dn_pos_idx, dn_nb, suffix):
        d = {}
        for loss in self.losses:
            l = self.get_loss(loss, out, targets, dn_pos_idx, dn_nb)
            l = {k: v * self.weight_dict[k] for k, v in l.items() if k in self.weight_dict}
            d.update({k + suffix: v for k, v in l.items()})
        return d
