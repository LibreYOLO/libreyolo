"""
Standard Non-Maximum Suppression (NMS) post-processor.

This module provides the classic NMS algorithm used in object detection
to remove redundant overlapping bounding boxes.
"""

import torch
from typing import Tuple, Optional

from .base import BasePostProcessor, PostProcessorConfig


def nms_kernel(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Core NMS algorithm using torch operations.

    This is a pure-torch implementation that doesn't require torchvision.
    It's used as a fallback when torchvision is not available.

    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Sort by scores (descending)
    _, order = scores.sort(0, descending=True)
    keep = []

    while len(order) > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i.item())

        if len(order) == 1:
            break

        # Calculate IoU with remaining boxes
        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]

        # Calculate intersection
        x1_i, y1_i, x2_i, y2_i = box_i
        x1_r = boxes_remaining[:, 0]
        y1_r = boxes_remaining[:, 1]
        x2_r = boxes_remaining[:, 2]
        y2_r = boxes_remaining[:, 3]

        x1_inter = torch.max(x1_i, x1_r)
        y1_inter = torch.max(y1_i, y1_r)
        x2_inter = torch.min(x2_i, x2_r)
        y2_inter = torch.min(y2_i, y2_r)

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

        # Calculate union
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        union_area = area_i + area_r - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)

        # Keep boxes with IoU < threshold
        order = order[1:][iou < iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# Check for torchvision availability once at module load
try:
    import torchvision.ops
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


class StandardNMS(BasePostProcessor):
    """
    Standard Non-Maximum Suppression (NMS).

    This is the classic NMS algorithm that keeps the highest-scoring box
    and removes all overlapping boxes with IoU above the threshold.

    Features:
        - Per-class NMS by default (agnostic=False)
        - Class-agnostic NMS option (agnostic=True)
        - Uses torchvision.ops.nms when available for performance
        - Falls back to pure-torch implementation otherwise

    Args:
        config: PostProcessorConfig with thresholds
        **kwargs: Override config parameters

    Example:
        >>> processor = StandardNMS(iou_thres=0.5)
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "nms"
    description = "Standard NMS - removes overlapping boxes above IoU threshold"
    supports_batched = False

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply standard NMS to detections.

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores
            class_ids: (N,) class indices

        Returns:
            Tuple of filtered (boxes, scores, class_ids)
        """
        if len(boxes) == 0:
            return self._empty_result(boxes.device)

        if self.config.agnostic:
            boxes, scores, class_ids = self._nms_agnostic(boxes, scores, class_ids)
        else:
            boxes, scores, class_ids = self._nms_per_class(boxes, scores, class_ids)

        return self._apply_max_det(boxes, scores, class_ids)

    def _nms_per_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply NMS separately for each class."""
        unique_classes = torch.unique(class_ids)
        keep_indices_list = []

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            # Use torchvision if available (faster)
            if _HAS_TORCHVISION:
                cls_keep = torchvision.ops.nms(cls_boxes, cls_scores, self.config.iou_thres)
            else:
                cls_keep = nms_kernel(cls_boxes, cls_scores, self.config.iou_thres)

            # Map back to original indices
            cls_indices = torch.where(cls_mask)[0]
            keep_indices_list.append(cls_indices[cls_keep])

        if not keep_indices_list:
            return self._empty_result(boxes.device)

        keep_indices = torch.cat(keep_indices_list)
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    def _nms_agnostic(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply NMS across all classes (class-agnostic)."""
        if _HAS_TORCHVISION:
            keep = torchvision.ops.nms(boxes, scores, self.config.iou_thres)
        else:
            keep = nms_kernel(boxes, scores, self.config.iou_thres)

        return boxes[keep], scores[keep], class_ids[keep]


class BatchedNMS(BasePostProcessor):
    """
    Batched NMS using torchvision's batched_nms.

    This is an optimized version that uses class offsets to perform
    per-class NMS in a single operation. Requires torchvision.

    The implementation offsets boxes by class ID to prevent cross-class
    suppression, then applies a single NMS pass.

    Args:
        config: PostProcessorConfig with thresholds
        **kwargs: Override config parameters
    """

    name = "batched_nms"
    description = "Optimized batched NMS using class offsets (requires torchvision)"
    supports_batched = True

    def __init__(self, config: Optional[PostProcessorConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        if not _HAS_TORCHVISION:
            raise ImportError(
                "BatchedNMS requires torchvision. Install with: pip install torchvision"
            )

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply batched NMS to detections.

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores
            class_ids: (N,) class indices

        Returns:
            Tuple of filtered (boxes, scores, class_ids)
        """
        if len(boxes) == 0:
            return self._empty_result(boxes.device)

        if self.config.agnostic:
            keep = torchvision.ops.nms(boxes, scores, self.config.iou_thres)
        else:
            keep = torchvision.ops.batched_nms(
                boxes, scores, class_ids, self.config.iou_thres
            )

        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        return self._apply_max_det(boxes, scores, class_ids)
