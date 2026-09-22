"""HungarianMatcher: matching cost + LSAP solver for RF-DETR.

Ported from RF-DETR (https://github.com/roboflow/rf-detr).
Copyright (c) 2025 Roboflow, Inc. All Rights Reserved.
Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR).
Copyright (c) 2024 Baidu. All Rights Reserved.
Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR).
Copyright (c) 2021 Microsoft. All Rights Reserved.
Modified from DETR (https://github.com/facebookresearch/detr).
Copyright (c) Facebook, Inc. and its affiliates.
Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR).
Copyright (c) 2020 SenseTime. All Rights Reserved.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    pairwise_box_l1_cost,
)
from .keypoints import compute_keypoint_matching_cost, map_labels_to_keypoint_schema
from .segmentation import point_sample

logger = logging.getLogger(__name__)
_SANITIZED_COST_MARGIN = 1.0


def _classic_keypoint_matching_cost(
    all_pred_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise keypoint costs for classic RF-DETR pose heads."""
    if all_pred_keypoints.shape[-1] < 3:
        raise ValueError("classic RF-DETR keypoints must have x, y, and visibility")
    if all_pred_keypoints.shape[-2] != target_keypoints.shape[-2]:
        raise ValueError(
            "classic RF-DETR keypoint count does not match target keypoint count: "
            f"{all_pred_keypoints.shape[-2]} vs {target_keypoints.shape[-2]}"
        )

    pred = all_pred_keypoints
    target = target_keypoints.to(device=pred.device, dtype=pred.dtype)
    target_xy = target[..., :2]
    target_vis = target[..., 2]

    finite_xy = torch.isfinite(target_xy).all(dim=-1)
    finite_vis = torch.isfinite(target_vis)
    visible = finite_xy & finite_vis & (target_vis > 0)
    visible_f = visible.to(pred.dtype)
    visible_count = visible_f.sum(dim=-1).clamp(min=1.0)

    l1 = (pred[:, :, None, :, :2] - target_xy[None, None, :, :, :]).abs().sum(dim=-1)
    cost_l1 = (l1 * visible_f[None, None]).sum(dim=-1) / visible_count[None, None]

    pred_vis_logits = pred[..., 2][:, :, None, :].expand(
        -1, -1, target.shape[0], -1
    )
    valid_vis_f = finite_vis.to(pred.dtype)
    vis_count = valid_vis_f.sum(dim=-1).clamp(min=1.0)
    target_findable = (target_vis > 0).to(pred.dtype)
    cost_findable = (
        F.binary_cross_entropy_with_logits(
            pred_vis_logits,
            target_findable[None, None].expand_as(pred_vis_logits),
            reduction="none",
        )
        * valid_vis_f[None, None]
    ).sum(dim=-1) / vis_count[None, None]
    cost_visible = torch.zeros_like(cost_findable)
    cost_nll = torch.zeros_like(cost_l1)
    return cost_l1, cost_findable, cost_visible, cost_nll


def _get_rng_state(device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return torch.get_rng_state()


def _set_rng_state(device, state):
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)
    else:
        torch.set_rng_state(state)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,  # reserved for future use; not yet implemented
        use_position_modulated_cost: bool = False,  # reserved for future use; not yet implemented
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
        cost_angle: float = 0,
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # Zero defaults disable keypoint matching terms so the detection/seg/obb
        # cost matrix is byte-identical when keypoints are off.
        num_keypoints_per_class=None,
        keypoint_l1_loss_coef: float = 0.0,
        keypoint_findable_loss_coef: float = 0.0,
        keypoint_visible_loss_coef: float = 0.0,
        keypoint_nll_loss_coef: float = 0.0,
    ):
        """Creates the matcher.

        Args:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates.
            cost_giou: Relative weight of the GIoU loss of the bounding box.
            focal_alpha: Alpha parameter for focal loss used in the classification cost.
            use_pos_only: Reserved for future use; currently has no effect.
            use_position_modulated_cost: Reserved for future use; currently has no effect.
            mask_point_sample_ratio: Downsampling ratio for mask point sampling.
            cost_mask_ce: Relative weight of the binary cross-entropy mask cost.
            cost_mask_dice: Relative weight of the Dice mask cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self.cost_angle = cost_angle
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        self.num_keypoints_per_class = list(num_keypoints_per_class) if num_keypoints_per_class else []
        self.keypoint_l1_loss_coef = keypoint_l1_loss_coef
        self.keypoint_findable_loss_coef = keypoint_findable_loss_coef
        self.keypoint_visible_loss_coef = keypoint_visible_loss_coef
        self.keypoint_nll_loss_coef = keypoint_nll_loss_coef
        self._warned_non_finite_costs = False

    @staticmethod
    def _sanitize_cost_matrix(cost_matrix: torch.Tensor) -> torch.Tensor:
        """Replace non-finite cost entries with a large finite sentinel.

        >>> HungarianMatcher._sanitize_cost_matrix(
        ...     torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
        ... ).tolist()
        [[1.0, 4.0], [4.0, -2.0]]

        Args:
            cost_matrix: Cost matrix to sanitize before Hungarian assignment.

        Returns:
            Cost matrix with all non-finite entries replaced by a finite
            sentinel that is no smaller than any valid entry.
        """
        finite_mask = torch.isfinite(cost_matrix)
        if finite_mask.all():
            return cost_matrix

        dtype_info = torch.finfo(cost_matrix.dtype)
        if finite_mask.any():
            finite_costs = cost_matrix[finite_mask]
            max_cost = finite_costs.max()
            # Add the largest absolute finite cost so the replacement stays
            # strictly larger than every valid entry, even if all costs are negative.
            replacement_cost = max_cost + finite_costs.abs().max() + _SANITIZED_COST_MARGIN
            # Guard against overflow to inf/NaN and clamp to the maximum finite value.
            if not torch.isfinite(replacement_cost):
                replacement_cost = cost_matrix.new_tensor(dtype_info.max)
            else:
                replacement_cost = torch.clamp(replacement_cost, max=dtype_info.max)
        else:
            # If all entries are non-finite, fall back to a large finite sentinel.
            replacement_cost = cost_matrix.new_tensor(dtype_info.max)

        sanitized_cost_matrix = cost_matrix.clone()
        sanitized_cost_matrix[~finite_mask] = replacement_cost
        return sanitized_cost_matrix

    @torch.no_grad()
    def compute_cost_matrix(self, outputs, targets):
        """Build the pairwise matching cost on the predictions' device.

        Split out from :meth:`forward` so a caller matching several output
        levels (main + aux + enc) can enqueue every level's cost on the GPU
        before the first host transfer, paying one pipeline drain instead of
        one per level (see ``SetCriterion.forward``).

        Returns:
            Cost matrix of dim [batch_size, num_queries, total_num_targets],
            float32, on the predictions' device.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_prob = flat_pred_logits.sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_angles = outputs.get("pred_angles")
        if out_angles is not None:
            out_angles = out_angles.flatten(0, 1).squeeze(-1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_angles = torch.cat([v["angles"] for v in targets]) if out_angles is not None and "angles" in targets[0] else None

        masks_present = "masks" in targets[0]
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        # Gate keypoint matching cost on both the prediction tensor and target key.
        keypoints_present = "pred_keypoints" in outputs and "keypoints" in targets[0]
        tgt_keypoints = None
        if keypoints_present:
            tgt_keypoints = torch.cat([v["keypoints"] for v in targets], dim=0)

        # Compute the giou cost between boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0

        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-F.logsigmoid(-flat_pred_logits))
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
        # --- GroupPose keypoint additions (adapted from RF-DETR v1.8.0). ---
        # The GroupPose detection head has one logit column per keypoint-schema
        # class, and the per-keypoint class-logit boost is added to the
        # keypoint-bearing column (internal index 1 for ``[0, 17]``). A
        # person-only dataset labels people as contiguous class 0, which would
        # index the empty schema slot (column 0), so the classification cost must
        # supervise the same internal column that the boost targets. Lift the
        # contiguous label to its schema index before indexing the cost columns.
        # Detection/seg/obb leave ``num_keypoints_per_class`` empty and so are
        # byte-identical (``cls_tgt_ids is tgt_ids``).
        cls_tgt_ids = tgt_ids
        if self.num_keypoints_per_class:
            cls_tgt_ids = map_labels_to_keypoint_schema(tgt_ids, self.num_keypoints_per_class)
        cost_class = pos_cost_class[:, cls_tgt_ids] - neg_cost_class[:, cls_tgt_ids]

        # Compute the L1 cost between boxes. The broadcast form is bit-identical
        # to ``torch.cdist(out_bbox, tgt_bbox, p=1)`` but avoids cdist's slow
        # one-thread-per-pair p=1 CUDA kernel, which alone cost 15% of all GPU
        # time in an rfdetr-s training step.
        cost_bbox = pairwise_box_l1_cost(out_bbox, tgt_bbox)
        cost_angle = 0
        if out_angles is not None and tgt_angles is not None and self.cost_angle:
            cost_angle = 1.0 - torch.cos(2.0 * (out_angles[:, None] - tgt_angles[None, :]))

        if masks_present:
            cost_mask_ce, cost_mask_dice = self._mask_costs(outputs, targets)

        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        if keypoints_present and tgt_keypoints is not None:
            if self.num_keypoints_per_class:
                target_areas = tgt_bbox[:, 2] * tgt_bbox[:, 3]
                # Class-index remap at the matcher boundary: lift the LibreYOLO
                # contiguous pose label (person = 0) to the GroupPose internal schema
                # class (person = 1 for ``[0, 17]``) so it indexes
                # ``num_keypoints_per_class`` at the keypoint-bearing slot. Label 0
                # would otherwise select the empty slot (0 keypoints) and the keypoint
                # matching cost would collapse to zero. The classification cost above
                # uses the same schema-space ids (``cls_tgt_ids``); reuse them here.
                kp_target_classes = map_labels_to_keypoint_schema(
                    tgt_ids, self.num_keypoints_per_class
                )
                cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
                    all_pred_keypoints=outputs["pred_keypoints"],
                    target_keypoints=tgt_keypoints,
                    target_classes=kp_target_classes,
                    target_areas=target_areas,
                    num_keypoints_per_class=self.num_keypoints_per_class,
                )
            else:
                cost_l1, cost_findable, cost_visible, cost_nll = (
                    _classic_keypoint_matching_cost(
                        outputs["pred_keypoints"],
                        tgt_keypoints,
                    )
                )
            cost_l1 = cost_l1.flatten(0, 1)
            cost_findable = cost_findable.flatten(0, 1)
            cost_visible = cost_visible.flatten(0, 1)
            cost_nll = cost_nll.flatten(0, 1)

        # Final cost matrix
        cost_matrix = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
            + self.cost_angle * cost_angle
        )
        if masks_present:
            cost_matrix = cost_matrix + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        # --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
        if keypoints_present and tgt_keypoints is not None:
            cost_matrix = (
                cost_matrix
                + self.keypoint_l1_loss_coef * cost_l1
                + self.keypoint_findable_loss_coef * cost_findable
                + self.keypoint_visible_loss_coef * cost_visible
                + self.keypoint_nll_loss_coef * cost_nll
            )
        # convert to float because bfloat16 doesn't play nicely with CPU
        return cost_matrix.view(bs, num_queries, -1).float()

    def solve(self, cost_matrix, targets, group_detr=1):
        """Run the Hungarian assignment on a CPU cost matrix.

        Args:
            cost_matrix: [batch_size, num_queries, total_num_targets] float32
                CPU tensor from :meth:`compute_cost_matrix` (after ``.cpu()``).
            targets: Same target list the cost matrix was built from.
            group_detr: Number of groups used for matching.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j).
        """
        # We assume any good match will not cause NaN or Inf, so replace invalid
        # entries with a finite value that is larger than every valid cost.
        finite_mask = torch.isfinite(cost_matrix)
        if not finite_mask.all():
            if not self._warned_non_finite_costs:
                logger.warning(
                    "Non-finite values detected in matcher cost matrix; "
                    "replacing with finite sentinel. "
                    "Check for numerical instability."
                )
                self._warned_non_finite_costs = True
            cost_matrix = self._sanitize_cost_matrix(cost_matrix)

        num_queries = cost_matrix.shape[1]
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        cost_matrix_list = cost_matrix.split(g_num_queries, dim=1)
        for g_i in range(group_detr):
            grouped_cost_matrix = cost_matrix_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(grouped_cost_matrix.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (
                        np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]),
                        np.concatenate([indice1[1], indice2[1]]),
                    )
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def _mask_costs(self, outputs, targets):
        tgt_masks = torch.cat([v["masks"] for v in targets])

        if isinstance(outputs["pred_masks"], torch.Tensor):
            out_masks = outputs["pred_masks"].flatten(0, 1)

            num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(
                out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
            ).squeeze(1)
        else:
            spatial_features = outputs["pred_masks"]["spatial_features"]
            query_features = outputs["pred_masks"]["query_features"]
            bias = outputs["pred_masks"]["bias"]

            num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
            point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
            pred_masks_logits = point_sample(
                spatial_features, point_coords.repeat(spatial_features.shape[0], 1, 1), align_corners=False
            )
            # print(f"pred_masks_logits.shape: {pred_masks_logits.shape}")
            pred_masks_logits = torch.einsum("bcp,bnc->bnp", pred_masks_logits, query_features) + bias
            pred_masks_logits = pred_masks_logits.flatten(0, 1)

        tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
        tgt_masks_flat = point_sample(
            tgt_masks.unsqueeze(1),
            point_coords.repeat(tgt_masks.shape[0], 1, 1),
            align_corners=False,
            mode="nearest",
        ).squeeze(1)

        # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
        cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

        # Dice loss cost (1 - dice coefficient)
        cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)
        return cost_mask_ce, cost_mask_dice

    def _compact_eligible(self, outputs, targets):
        """Keep extended tasks and heterogeneous target tensors on their existing path."""
        if not targets or len(targets) != outputs["pred_boxes"].shape[0]:
            return False
        if any(key in outputs for key in ("pred_keypoints", "pred_angles")):
            return False
        first = targets[0]["boxes"]
        return all(
            set(target) >= {"boxes", "labels"}
            and target["boxes"].dtype == first.dtype
            and target["boxes"].device == outputs["pred_boxes"].device
            and target["labels"].device == outputs["pred_logits"].device
            and target["labels"].dtype == torch.int64
            and len(target["labels"]) == len(target["boxes"])
            for target in targets
        )

    def _compact_cost(self, outputs, targets):
        """Build only same-image detection costs, padded to the largest target count.

        Equivalent to selecting the diagonal image blocks from the full
        Cartesian matrix. Preserve the existing focal and box formulas.
        """
        from torch.nn.utils.rnn import pad_sequence

        labels = pad_sequence([target["labels"] for target in targets], batch_first=True)
        # Advanced indexing in the original path accepts negative class ids.
        # Preserve that behavior rather than giving gather a different contract.
        labels = torch.where(labels < 0, labels + outputs["pred_logits"].shape[-1], labels)
        boxes = pad_sequence([target["boxes"] for target in targets], batch_first=True)
        logits = outputs["pred_logits"].gather(
            2, labels[:, None, :].expand(-1, outputs["pred_logits"].shape[1], -1)
        )
        prob = logits.sigmoid()
        cost_class = 0.25 * ((1 - prob) ** 2) * (-F.logsigmoid(logits)) - 0.75 * (prob ** 2) * (-F.logsigmoid(-logits))
        cost_bbox = pairwise_box_l1_cost(outputs["pred_boxes"], boxes)
        cost_giou = -torch.vmap(generalized_box_iou)(
            box_cxcywh_to_xyxy(outputs["pred_boxes"]), box_cxcywh_to_xyxy(boxes)
        )
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        compact = torch.cat([cost[i, :, :len(target["boxes"])] for i, target in enumerate(targets)], -1)
        if "masks" in targets[0]:
            # Keep mask sampling and its RNG draw in the original batch order.
            mask_ce, mask_dice = self._mask_costs(outputs, targets)
            batch, queries = outputs["pred_boxes"].shape[:2]
            sizes = [len(target["boxes"]) for target in targets]
            def diagonal_blocks(values):
                return torch.cat([block[i] for i, block in enumerate(values.view(batch, queries, -1).split(sizes, -1))], -1)
            compact = compact + self.cost_mask_ce * diagonal_blocks(mask_ce) + self.cost_mask_dice * diagonal_blocks(mask_dice)
        return compact.float()

    def _try_batched_assignment(self, matrices, sizes, group_detr):
        backend = getattr(self, "matcher_backend", "scipy")
        if backend not in {"scipy", "auto", "torch"}:
            raise ValueError(f"Unknown RF-DETR matcher backend: {backend}")
        use_torch = backend == "torch" or (backend == "auto" and matrices[0].is_cuda)
        if use_torch and torch.stack([torch.isfinite(matrix).all() for matrix in matrices]).all():
            try:
                from .assignment import assign_compact
                return assign_compact(matrices, sizes, group_detr)
            except ModuleNotFoundError as exc:
                if exc.name != "torch_linear_assignment":
                    raise
                if backend == "torch":
                    raise ImportError("matcher_backend='torch' requires libreyolo[rfdetr-accel]") from exc
                if not getattr(self, "_warned_assignment_fallback", False):
                    logger.warning("GPU assignment requires libreyolo[rfdetr-accel]; using SciPy")
                    self._warned_assignment_fallback = True
        elif backend == "auto" and not matrices[0].is_cuda and not getattr(self, "_warned_assignment_fallback", False):
            logger.info("GPU assignment is unavailable on %s; using SciPy", matrices[0].device.type)
            self._warned_assignment_fallback = True
        return None

    def _solve_dense(self, matrix, targets, group_detr):
        # Extended tasks keep their own cost equations, but can still opt into
        # the same assignment backend. Do not silently ignore the public choice.
        if getattr(self, "matcher_backend", "scipy") != "scipy":
            sizes = [len(target["boxes"]) for target in targets]
            compact = torch.cat([block[i] for i, block in enumerate(matrix.split(sizes, -1))], -1)
            assigned = self._try_batched_assignment([compact], sizes, group_detr)
            if assigned is not None:
                return assigned[0]
        return self.solve(matrix.cpu(), targets, group_detr)

    @torch.no_grad()
    def match_many(self, levels, targets, group_detr=1):
        """Match output layers with one compact host transfer when memory permits.

        Large/extended-task batches retain the depth-two pipeline. Non-finite
        compact costs rerun the original full matrix so sentinel selection
        and its warning retain their original global semantics.
        """
        total_targets = sum(len(target["boxes"]) for target in targets)
        elements = total_targets * sum(level["pred_boxes"].shape[1] for level in levels)
        compact = elements <= 16 * 1024 * 1024 and all(
            self._compact_eligible(level, targets) for level in levels
        )
        if compact:
            sizes = [len(target["boxes"]) for target in targets]
            # Preserve per-level RNG states for the exceptional full-matrix
            # fallback; mask matching draws coordinates once per level.
            rng_states = []
            matrices = []
            for level in levels:
                device = level["pred_boxes"].device
                rng = None
                if "masks" in targets[0]:
                    rng = _get_rng_state(device)
                rng_states.append(rng)
                matrices.append(self._compact_cost(level, targets))
            assigned = self._try_batched_assignment(matrices, sizes, group_detr)
            if assigned is not None:
                return assigned
            shapes = [matrix.shape for matrix in matrices]
            host = torch.cat([matrix.flatten() for matrix in matrices]).cpu()
            results, offset = [], 0
            for level, shape, rng_state in zip(levels, shapes, rng_states):
                count = shape.numel()
                matrix = host[offset:offset + count].view(shape)
                offset += count
                if not torch.isfinite(matrix).all():
                    device = level["pred_boxes"].device
                    if rng_state is None:
                        full = self.compute_cost_matrix(level, targets).cpu()
                    else:
                        current_rng = _get_rng_state(device)
                        try:
                            _set_rng_state(device, rng_state)
                            full = self.compute_cost_matrix(level, targets).cpu()
                        finally:
                            _set_rng_state(device, current_rng)
                    results.append(self.solve(full, targets, group_detr))
                    continue
                if shape[0] % group_detr:
                    raise ValueError("RF-DETR query count must be divisible by group_detr")
                width = shape[0] // group_detr
                image_results = []
                for image in matrix.split(sizes, dim=1):
                    pairs = [linear_sum_assignment(part) for part in image.split(width, dim=0)]
                    rows = np.concatenate([pair[0] + group * width for group, pair in enumerate(pairs)])
                    columns = np.concatenate([pair[1] for pair in pairs])
                    image_results.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(columns, dtype=torch.int64)))
                results.append(image_results)
            return results
        pending = self.compute_cost_matrix(levels[0], targets)
        results = []
        for level in levels[1:]:
            following = self.compute_cost_matrix(level, targets)
            results.append(self._solve_dense(pending, targets, group_detr))
            pending = following
        results.append(self._solve_dense(pending, targets, group_detr))
        return results

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                 "masks": Tensor of dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.match_many([outputs], targets, group_detr=group_detr)[0]


def build_matcher(args):
    # Detection-only matcher args may omit keypoint costs; zero defaults disable
    # keypoint matching terms. --- GroupPose keypoint additions (ported from RF-DETR v1.8.0). ---
    common_kwargs = {
        "cost_class": args.set_cost_class,
        "cost_bbox": args.set_cost_bbox,
        "cost_giou": args.set_cost_giou,
        "focal_alpha": args.focal_alpha,
        "cost_angle": getattr(args, "set_cost_angle", 0.0),
        "num_keypoints_per_class": getattr(args, "num_keypoints_per_class", []),
        "keypoint_l1_loss_coef": getattr(args, "keypoint_l1_loss_coef", 0.0),
        "keypoint_findable_loss_coef": getattr(args, "keypoint_findable_loss_coef", 0.0),
        "keypoint_visible_loss_coef": getattr(args, "keypoint_visible_loss_coef", 0.0),
        "keypoint_nll_loss_coef": getattr(args, "keypoint_nll_loss_coef", 0.0),
    }
    if args.segmentation_head:
        return HungarianMatcher(
            **common_kwargs,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(**common_kwargs)
