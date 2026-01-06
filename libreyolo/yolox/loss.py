"""
YOLOX Loss with SimOTA assigner.

This module contains the loss computation and label assignment for YOLOX training.
Extracted from the original YOLOX implementation (Megvii).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def meshgrid(*tensors):
    """Create a meshgrid (compatible with different PyTorch versions)."""
    _TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) tensor
        bboxes_b: (M, 4) tensor
        xyxy: If True, boxes are in xyxy format. If False, cxcywh format.

    Returns:
        (N, M) tensor of IoU values
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Bounding boxes must have 4 coordinates")

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # cxcywh format
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).to(device=tl.device, dtype=tl.dtype).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    """IoU loss for bounding box regression."""

    def __init__(self, reduction: str = "none", loss_type: str = "iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)

        # Convert cxcywh to corners for IoU calculation
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).to(device=tl.device, dtype=tl.dtype).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOXLoss:
    """
    YOLOX Loss with SimOTA label assignment.

    This loss function handles:
    - IoU loss for bounding box regression
    - BCE loss for objectness
    - BCE loss for classification
    - Optional L1 loss (enabled in final training epochs)
    - SimOTA dynamic label assignment

    Args:
        num_classes: Number of object classes
        strides: Feature map strides (default: [8, 16, 32])
        use_l1: Whether to use L1 loss (typically enabled in final epochs)
    """

    def __init__(
        self,
        num_classes: int,
        strides: List[int] = [8, 16, 32],
        use_l1: bool = False
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.use_l1 = use_l1

        # Loss functions
        self.iou_loss = IOUloss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

        # Grid cache
        self.grids = [torch.zeros(1)] * len(strides)

    def __call__(
        self,
        outputs: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLOX losses.

        Args:
            outputs: List of head outputs, each (B, 5+num_classes, H, W)
                     Format: [reg(4), obj(1), cls(num_classes)]
            targets: Ground truth (B, max_obj, 5) in format [class, cx, cy, w, h]
                     Coordinates are in pixel space.

        Returns:
            Dict with total_loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg
        """
        # Process outputs and build grids
        processed_outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        dtype = outputs[0].dtype
        device = outputs[0].device

        for k, output in enumerate(outputs):
            batch_size = output.shape[0]
            hsize, wsize = output.shape[-2:]
            stride = self.strides[k]

            # Reshape output: (B, C, H, W) -> (B, H*W, C)
            # Output format: [reg(4), obj(1), cls(num_classes)]
            n_ch = 5 + self.num_classes
            output_reshaped = output.view(batch_size, n_ch, hsize, wsize)
            output_reshaped = output_reshaped.permute(0, 2, 3, 1).reshape(batch_size, hsize * wsize, n_ch)

            # Generate grid
            grid = self.grids[k]
            if grid.shape[0] != hsize or (len(grid.shape) > 1 and grid.shape[1] != wsize):
                yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
                grid = torch.stack((xv, yv), 2).view(1, hsize * wsize, 2).to(device=device, dtype=dtype)
                self.grids[k] = grid

            # Decode boxes: (grid + offset) * stride
            output_decoded = output_reshaped.clone()
            output_decoded[..., :2] = (output_reshaped[..., :2] + grid) * stride
            output_decoded[..., 2:4] = torch.exp(output_reshaped[..., 2:4]) * stride

            processed_outputs.append(output_decoded)
            x_shifts.append(grid[..., 0])
            y_shifts.append(grid[..., 1])
            expanded_strides.append(
                torch.full((1, hsize * wsize), stride, device=device, dtype=dtype)
            )

            # For L1 loss, keep original predictions
            if self.use_l1:
                origin_preds.append(output_reshaped[..., :4].clone())

        # Concatenate all scales
        outputs_cat = torch.cat(processed_outputs, dim=1)  # (B, total_anchors, 5+num_classes)
        x_shifts = torch.cat(x_shifts, dim=1)  # (1, total_anchors)
        y_shifts = torch.cat(y_shifts, dim=1)
        expanded_strides = torch.cat(expanded_strides, dim=1)

        if self.use_l1:
            origin_preds = torch.cat(origin_preds, dim=1)

        return self._compute_losses(
            outputs_cat,
            targets,
            x_shifts,
            y_shifts,
            expanded_strides,
            origin_preds if self.use_l1 else None,
            dtype,
        )

    def _compute_losses(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        expanded_strides: torch.Tensor,
        origin_preds: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses after output processing."""

        bbox_preds = outputs[:, :, :4]  # (B, total_anchors, 4)
        obj_preds = outputs[:, :, 4:5]  # (B, total_anchors, 1)
        cls_preds = outputs[:, :, 5:]   # (B, total_anchors, num_classes)

        # Count objects per image
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt

            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                gt_classes = targets[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self._get_assignments(
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self._get_assignments(
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        mode="cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self._get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        # IoU loss
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg

        # Objectness loss
        loss_obj = (
            self.bce_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        # Classification loss
        loss_cls = (
            self.bce_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        # L1 loss (optional)
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = torch.tensor(0.0, device=outputs.device)

        # Total loss with weights
        reg_weight = 5.0
        total_loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return {
            "total_loss": total_loss,
            "iou_loss": reg_weight * loss_iou,
            "obj_loss": loss_obj,
            "cls_loss": loss_cls,
            "l1_loss": loss_l1,
            "num_fg": num_fg / max(num_gts, 1),
        }

    def _get_l1_target(
        self,
        l1_target: torch.Tensor,
        gt: torch.Tensor,
        stride: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """Compute L1 regression targets."""
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def _get_assignments(
        self,
        batch_idx: int,
        num_gt: int,
        gt_bboxes_per_image: torch.Tensor,
        gt_classes: torch.Tensor,
        bboxes_preds_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        cls_preds: torch.Tensor,
        obj_preds: torch.Tensor,
        mode: str = "gpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        SimOTA label assignment.

        Assigns ground truth boxes to predictions using the SimOTA algorithm.
        """
        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self._get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes).float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self._simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def _get_geometry_constraint(
        self,
        gt_bboxes_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate geometry constraints for anchor-GT matching.

        Filters anchors to only those whose centers are within a fixed radius
        of the ground truth box centers.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def _simota_matching(
        self,
        cost: torch.Tensor,
        pair_wise_ious: torch.Tensor,
        gt_classes: torch.Tensor,
        num_gt: int,
        fg_mask: torch.Tensor,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SimOTA matching algorithm.

        Dynamically assigns predictions to ground truths based on cost matrix.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)

        # Handle case where one anchor matches multiple GTs
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1

        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
