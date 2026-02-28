"""
YOLOv9 Loss functions for training.

Ported from https://github.com/WongKinYiu/YOLO (MIT License)
Adapted for LibreYOLO's v9 architecture.

Includes:
- Task Aligned Assignment (TAL) via BoxMatcher
- CIoU box loss
- Distribution Focal Loss (DFL) for anchor-free regression
- BCE classification loss
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_iou(bbox1: Tensor, bbox2: Tensor, metrics: str = "iou") -> Tensor:
    """
    Calculate IoU, DIoU, or CIoU between two sets of bounding boxes.

    Args:
        bbox1: Bounding boxes in xyxy format. Shape: (A, 4) or (B, A, 4)
        bbox2: Bounding boxes in xyxy format. Shape: (B, 4) or (B, B, 4)
        metrics: IoU variant - "iou", "diou", or "ciou"

    Returns:
        IoU matrix. Shape depends on input dimensions.
    """
    metrics = metrics.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(
        ymax_inter - ymin_inter, min=0
    )

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term (CIoU)
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


def generate_anchors(image_size: List[int], strides: List[int]) -> Tuple[Tensor, Tensor]:
    """
    Generate anchor grid points for all feature map scales.

    Args:
        image_size: Image size as [W, H]
        strides: List of stride values for each feature level [8, 16, 32]

    Returns:
        anchor_grid: Tensor of shape [total_anchors, 2] with (x, y) coordinates
        scaler: Tensor of shape [total_anchors] with stride values
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride, dtype=torch.float32))
        shift = stride // 2
        h = torch.arange(0, H, stride, dtype=torch.float32) + shift
        w = torch.arange(0, W, stride, dtype=torch.float32) + shift
        anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


# =============================================================================
# Loss Classes
# =============================================================================

class BCELoss(nn.Module):
    """Binary Cross Entropy loss for classification, normalized by positive samples."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Tensor:
        """
        Args:
            predicts_cls: Predicted class logits [B, anchors, num_classes]
            targets_cls: Target class distribution [B, anchors, num_classes]
            cls_norm: Normalization factor (total positive samples)
        """
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    """CIoU loss for bounding box regression."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        """
        Args:
            predicts_bbox: Predicted boxes [B, anchors, 4] in xyxy (normalized)
            targets_bbox: Target boxes [B, anchors, 4] in xyxy (normalized)
            valid_masks: Boolean mask [B, anchors] indicating positive anchors
            box_norm: Per-anchor weights [num_positive]
            cls_norm: Normalization factor
        """
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    """Distribution Focal Loss for anchor-free box regression."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(
        self,
        predicts_anc: Tensor,
        targets_bbox: Tensor,
        anchors_norm: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        """
        Args:
            predicts_anc: Predicted anchor distributions [B, anchors, 4, reg_max]
            targets_bbox: Target boxes [B, anchors, 4] in xyxy (normalized)
            anchors_norm: Normalized anchor grid [1, anchors, 2]
            valid_masks: Boolean mask [B, anchors]
            box_norm: Per-anchor weights
            cls_norm: Normalization factor
        """
        # valid_masks is (B, anchors), expand to (B, anchors, 4) for bbox selection
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)

        # Compute target distances from anchors: (anchors, 2) broadcast with (B, anchors, 2)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)  # each (B, anchors, 2)
        targets_dist = torch.cat(
            ((anchors_norm - bbox_lt), (bbox_rb - anchors_norm)), -1
        ).clamp(0, self.reg_max - 1.01)  # (B, anchors, 4)

        # Select valid targets: (B, anchors, 4)[mask] -> (num_valid, 4) -> flatten to (num_valid * 4,)
        picked_targets = targets_dist[valid_bbox].view(-1)

        # predicts_anc is (B, anchors, 4, reg_max)
        # Select valid predictions: need to expand mask to (B, anchors, 4, reg_max)
        valid_anc = valid_bbox[..., None].expand(-1, -1, -1, self.reg_max)
        picked_predict = predicts_anc[valid_anc].view(-1, self.reg_max)

        # Soft label assignment between adjacent bins
        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


# =============================================================================
# Vec2Box - Prediction Converter
# =============================================================================

class Vec2Box:
    """
    Convert raw detection outputs to decoded format for loss computation.

    Handles LibreYOLO v9's output format:
    - Input: List of tensors [P3, P4, P5], each (B, nc + 4*reg_max, H, W)
    - Output: (preds_cls, preds_anc, preds_box) all flattened across scales
    """

    def __init__(
        self,
        strides: List[int],
        image_size: List[int],
        reg_max: int,
        num_classes: int,
        device: torch.device,
    ):
        self.strides = strides
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.device = device

        anchor_grid, scaler = generate_anchors(image_size, strides)
        self.image_size = image_size
        self.anchor_grid = anchor_grid.to(device)
        self.scaler = scaler.to(device)

    def update(self, image_size: List[int]):
        """Update anchors for new image size."""
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid = anchor_grid.to(self.device)
        self.scaler = scaler.to(self.device)

    def __call__(self, predicts: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert raw predictions to loss-ready format.

        Args:
            predicts: List of [P3, P4, P5] tensors, each (B, nc + 4*reg_max, H, W)
                     Channel order: first 4*reg_max channels are box, last nc are class

        Returns:
            preds_cls: (B, total_anchors, num_classes) - class logits
            preds_anc: (B, total_anchors, reg_max, 4) - raw anchor distributions
            preds_box: (B, total_anchors, 4) - decoded boxes in xyxy (pixel coords)
        """
        preds_cls_list = []
        preds_anc_list = []
        preds_box_list = []
        box_channels = 4 * self.reg_max

        for pred in predicts:
            B, C, H, W = pred.shape

            # Split channels: box (4*reg_max) | class (nc)
            pred_box_raw = pred[:, :box_channels, :, :]  # (B, 4*reg_max, H, W)
            pred_cls = pred[:, box_channels:, :, :]  # (B, nc, H, W)

            # Reshape class predictions: (B, nc, H, W) -> (B, H*W, nc)
            pred_cls = pred_cls.permute(0, 2, 3, 1).reshape(B, H * W, -1)
            preds_cls_list.append(pred_cls)

            # Reshape box predictions for DFL: (B, 4*reg_max, H, W) -> (B, H*W, 4, reg_max)
            # Format: (B, anchors, 4, reg_max) matches YOLO repo for DFL loss
            pred_anc = pred_box_raw.view(B, 4, self.reg_max, H, W)
            pred_anc = pred_anc.permute(0, 3, 4, 1, 2).reshape(B, H * W, 4, self.reg_max)
            preds_anc_list.append(pred_anc)

            # Decode boxes using DFL (softmax + weighted sum)
            # (B, H*W, 4, reg_max) -> softmax over reg_max -> (B, H*W, 4)
            pred_dist = F.softmax(pred_anc, dim=3)
            # Weighted sum: multiply by [0, 1, 2, ..., reg_max-1]
            proj = torch.arange(self.reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
            pred_box = (pred_dist * proj.view(1, 1, 1, -1)).sum(dim=3)  # (B, H*W, 4)
            preds_box_list.append(pred_box)

        # Concatenate across scales
        preds_cls = torch.cat(preds_cls_list, dim=1)  # (B, total_anchors, nc)
        preds_anc = torch.cat(preds_anc_list, dim=1)  # (B, total_anchors, 4, reg_max)
        preds_box = torch.cat(preds_box_list, dim=1)  # (B, total_anchors, 4) - LTRB distances

        # Convert LTRB distances to xyxy coordinates (pixel space)
        # pred_box is in "grid units", scale by stride
        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)

        return preds_cls, preds_anc, preds_box


# =============================================================================
# BoxMatcher - Task Aligned Assignment (TAL)
# =============================================================================

class BoxMatcher:
    """
    Task Aligned Assignment (TAL) for matching ground truths to anchors.

    Uses combined IoU and classification score to determine best anchor matches.
    """

    def __init__(
        self,
        num_classes: int,
        anchor_grid: Tensor,
        scaler: Tensor,
        reg_max: int,
        topk: int = 10,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
    ):
        self.num_classes = num_classes
        self.anchor_grid = anchor_grid
        self.scaler = scaler
        self.reg_max = reg_max
        self.topk = topk
        self.iou_factor = iou_factor
        self.cls_factor = cls_factor

    def get_valid_matrix(self, target_bbox: Tensor) -> Tensor:
        """
        Get valid anchor mask based on whether anchor can predict the target.

        Args:
            target_bbox: Target boxes [B, targets, 4] in xyxy format

        Returns:
            Valid mask [B, targets, anchors]
        """
        x_min, y_min, x_max, y_max = target_bbox[:, :, None].unbind(3)
        anchors = self.anchor_grid[None, None]  # (1, 1, anchors, 2)
        anchors_x, anchors_y = anchors.unbind(dim=3)

        x_min_dist, x_max_dist = anchors_x - x_min, x_max - anchors_x
        y_min_dist, y_max_dist = anchors_y - y_min, y_max - anchors_y
        targets_dist = torch.stack((x_min_dist, y_min_dist, x_max_dist, y_max_dist), dim=-1)
        targets_dist /= self.scaler[None, None, :, None]

        min_reg_dist, max_reg_dist = targets_dist.amin(dim=-1), targets_dist.amax(dim=-1)
        target_on_anchor = min_reg_dist >= 0
        target_in_reg_max = max_reg_dist <= self.reg_max - 1.01
        return target_on_anchor & target_in_reg_max

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get predicted class probabilities for target classes.

        Args:
            predict_cls: Predicted class probs [B, anchors, num_classes]
            target_cls: Target class indices [B, targets, 1]

        Returns:
            Class probabilities [B, targets, anchors]
        """
        predict_cls = predict_cls.transpose(1, 2)  # (B, nc, anchors)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))  # (B, targets, anchors)
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox: Tensor, target_bbox: Tensor) -> Tensor:
        """
        Compute IoU matrix between predictions and targets.

        Args:
            predict_bbox: Predicted boxes [B, anchors, 4]
            target_bbox: Target boxes [B, targets, 4]

        Returns:
            IoU matrix [B, targets, anchors]
        """
        return calculate_iou(target_bbox, predict_bbox, "ciou").clamp(0, 1)

    def filter_topk(
        self, target_matrix: Tensor, grid_mask: Tensor, topk: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """Filter top-k anchors for each target."""
        masked_target_matrix = grid_mask * target_matrix
        values, indices = masked_target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_mask = topk_targets > 0
        return topk_targets, topk_mask

    def ensure_one_anchor(self, target_matrix: Tensor, topk_mask: Tensor) -> Tensor:
        """Ensure each valid target gets at least one anchor."""
        values, indices = target_matrix.max(dim=-1)
        best_anchor_mask = torch.zeros_like(target_matrix, dtype=torch.bool)
        best_anchor_mask.scatter_(-1, index=indices[..., None], src=~best_anchor_mask)
        matched_anchor_num = torch.sum(topk_mask, dim=-1)
        target_without_anchor = (matched_anchor_num == 0) & (values > 0)
        topk_mask = torch.where(target_without_anchor[..., None], best_anchor_mask, topk_mask)
        return topk_mask

    def filter_duplicates(
        self, iou_mat: Tensor, topk_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Remove duplicate assignments (one anchor -> multiple targets)."""
        duplicates = (topk_mask.sum(1, keepdim=True) > 1).repeat([1, topk_mask.size(1), 1])
        masked_iou_mat = topk_mask * iou_mat
        best_indices = masked_iou_mat.argmax(1)[:, None, :]
        best_target_mask = torch.zeros_like(duplicates, dtype=torch.bool)
        best_target_mask.scatter_(1, index=best_indices, src=~best_target_mask)
        topk_mask = torch.where(duplicates, best_target_mask, topk_mask)
        unique_indices = topk_mask.to(torch.uint8).argmax(dim=1)
        return unique_indices[..., None], topk_mask.any(dim=1), topk_mask

    def __call__(
        self, target: Tensor, predict: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Match targets to anchors using Task Aligned Assignment.

        Args:
            target: Ground truth [B, targets, 5] with [class_id, x1, y1, x2, y2]
            predict: Tuple of (pred_cls, pred_bbox)
                - pred_cls: [B, anchors, num_classes] - class logits (will be sigmoided)
                - pred_bbox: [B, anchors, 4] - decoded boxes in pixel coords

        Returns:
            anchor_matched_targets: [B, anchors, num_classes + 4]
            valid_mask: [B, anchors] boolean mask
        """
        predict_cls, predict_bbox = predict

        n_targets = target.shape[1]
        if n_targets == 0:
            device = predict_bbox.device
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros(predict_cls.shape[:2], dtype=bool, device=device)
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            return anchor_matched_targets, valid_mask

        target_cls, target_bbox = target.split([1, 4], dim=-1)
        target_cls = target_cls.long().clamp(0)

        # Get valid matrix (which anchors can predict which targets)
        grid_mask = self.get_valid_matrix(target_bbox)

        # Get IoU matrix
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # Get class probability matrix
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        # Compute task-aligned score
        target_matrix = (iou_mat ** self.iou_factor) * (cls_mat ** self.cls_factor)

        # Select top-k anchors per target
        topk_targets, topk_mask = self.filter_topk(target_matrix, grid_mask, topk=self.topk)

        # Ensure each target has at least one anchor
        topk_mask = self.ensure_one_anchor(target_matrix, topk_mask)

        # Remove duplicate assignments
        unique_indices, valid_mask, topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        # Gather assigned targets
        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls_indices = torch.gather(target_cls, 1, unique_indices)
        align_cls = torch.zeros_like(align_cls_indices, dtype=torch.bool).repeat(
            1, 1, self.num_classes
        )
        align_cls.scatter_(-1, index=align_cls_indices, src=~align_cls)

        # Normalize class distribution by task-aligned score
        iou_mat *= topk_mask
        target_matrix *= topk_mask
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
        return anchor_matched_targets, valid_mask


# =============================================================================
# YOLOv9Loss - Main Loss Class
# =============================================================================

class YOLOv9Loss:
    """
    Combined loss for YOLOv9 training (single head, no auxiliary).

    Computes:
    - Box loss (CIoU)
    - DFL loss (Distribution Focal Loss)
    - Classification loss (BCE)
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int,
        strides: List[int],
        image_size: Optional[List[int]],
        device: torch.device,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        cls_weight: float = 0.5,
        topk: int = 10,
        iou_factor: float = 6.0,
        cls_factor: float = 0.5,
    ):
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.device = device

        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.cls_weight = cls_weight

        # TAL matcher parameters - must be set before _init_vec2box
        self.topk = topk
        self.iou_factor = iou_factor
        self.cls_factor = cls_factor

        # Loss functions
        self.cls_loss = BCELoss()
        self.box_loss = BoxLoss()
        self.dfl_loss = DFLoss(reg_max)

        # Matcher will be created when needed
        self.matcher = None
        self.vec2box = None

        # Initialize Vec2Box if image_size provided
        if image_size is not None:
            self._init_vec2box(image_size)

    def _init_vec2box(self, image_size: List[int]):
        """Initialize or update Vec2Box converter."""
        self.vec2box = Vec2Box(
            strides=self.strides,
            image_size=image_size,
            reg_max=self.reg_max,
            num_classes=self.num_classes,
            device=self.device,
        )
        # Initialize matcher with new anchors
        self.matcher = BoxMatcher(
            num_classes=self.num_classes,
            anchor_grid=self.vec2box.anchor_grid,
            scaler=self.vec2box.scaler,
            reg_max=self.reg_max,
            topk=self.topk,
            iou_factor=self.iou_factor,
            cls_factor=self.cls_factor,
        )

    def update_anchors(self, image_size: List[int]):
        """Update anchors for new image size."""
        if self.vec2box is None or self.vec2box.image_size != image_size:
            self._init_vec2box(image_size)

    def __call__(
        self, predictions: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute YOLOv9 loss.

        Args:
            predictions: List of [P3, P4, P5] tensors from DDetect head
                        Each tensor: (B, nc + 4*reg_max, H, W)
            targets: Ground truth [B, max_targets, 5] with [class_id, x1, y1, x2, y2]
                    Coordinates are normalized (0-1)

        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dict with individual loss values for logging
        """
        if self.vec2box is None:
            raise RuntimeError("Vec2Box not initialized. Call update_anchors() first.")

        # Convert predictions to loss-ready format
        preds_cls, preds_anc, preds_box = self.vec2box(predictions)

        # Scale targets from normalized to pixel coordinates
        B = targets.shape[0]
        W, H = self.vec2box.image_size
        scale = torch.tensor([1, W, H, W, H], device=targets.device, dtype=targets.dtype)
        targets_scaled = targets * scale

        # Run Task Aligned Assignment
        align_targets, valid_masks = self.matcher(
            targets_scaled, (preds_cls.detach(), preds_box.detach())
        )

        # Separate class and box targets
        targets_cls, targets_bbox = torch.split(
            align_targets, (self.num_classes, 4), dim=-1
        )

        # Normalize predicted boxes to same scale as targets
        preds_box_norm = preds_box / self.vec2box.scaler[None, :, None]
        targets_bbox_norm = targets_bbox / self.vec2box.scaler[None, :, None]

        # Compute normalization factors
        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        # Compute losses
        loss_cls = self.cls_loss(preds_cls, targets_cls, cls_norm)
        loss_box = self.box_loss(preds_box_norm, targets_bbox_norm, valid_masks, box_norm, cls_norm)

        # DFL loss needs normalized anchor grid
        anchors_norm = (self.vec2box.anchor_grid / self.vec2box.scaler[:, None])[None]
        loss_dfl = self.dfl_loss(
            preds_anc, targets_bbox_norm, anchors_norm, valid_masks, box_norm, cls_norm
        )

        # Apply weights
        loss_box_weighted = self.box_weight * loss_box
        loss_dfl_weighted = self.dfl_weight * loss_dfl
        loss_cls_weighted = self.cls_weight * loss_cls

        total_loss = loss_box_weighted + loss_dfl_weighted + loss_cls_weighted

        # Return dict format consistent with YOLOX
        loss_dict = {
            "total_loss": total_loss,
            # Loss components (tensor form for backward)
            "box_loss": loss_box_weighted,
            "dfl_loss": loss_dfl_weighted,
            "cls_loss": loss_cls_weighted,
            # Scalar values for logging
            "box": loss_box_weighted.item() if isinstance(loss_box_weighted, Tensor) else loss_box_weighted,
            "dfl": loss_dfl_weighted.item() if isinstance(loss_dfl_weighted, Tensor) else loss_dfl_weighted,
            "cls": loss_cls_weighted.item() if isinstance(loss_cls_weighted, Tensor) else loss_cls_weighted,
            "num_fg": valid_masks.sum().item() / max(B, 1),
        }

        return loss_dict
