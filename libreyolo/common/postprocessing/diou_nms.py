"""
Distance-IoU (DIoU) and Complete-IoU (CIoU) based NMS.

These methods use enhanced IoU metrics that consider box center distances
and aspect ratios, leading to better performance on small and overlapping objects.

Reference: https://arxiv.org/abs/1911.08287
"""

import torch
from typing import Tuple, Optional, Literal

from .base import BasePostProcessor, PostProcessorConfig
from .box_utils import box_diou, box_ciou


def diou_nms_kernel(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    use_ciou: bool = False
) -> torch.Tensor:
    """
    Core DIoU/CIoU-NMS algorithm.

    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: DIoU/CIoU threshold for suppression
        use_ciou: If True, use CIoU instead of DIoU

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

        # Calculate DIoU/CIoU with remaining boxes
        box_i = boxes[i:i+1]
        boxes_remaining = boxes[order[1:]]

        if use_ciou:
            metric = box_ciou(box_i, boxes_remaining)[0]
        else:
            metric = box_diou(box_i, boxes_remaining)[0]

        # Keep boxes with DIoU/CIoU < threshold
        order = order[1:][metric < iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class DIoUNMS(BasePostProcessor):
    """
    Distance-IoU (DIoU) based NMS.

    DIoU extends standard IoU by adding a penalty term based on the
    normalized distance between box centers. This helps handle cases
    where boxes don't overlap but are close together.

    Benefits:
        - Better performance on small objects
        - Improved handling of closely positioned objects
        - Reported +7.6% mAP for small UAV targets

    Args:
        config: PostProcessorConfig with thresholds
        **kwargs: Override config parameters

    Example:
        >>> processor = DIoUNMS(iou_thres=0.5)
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "diou_nms"
    description = "DIoU-NMS - distance-aware suppression, better for small objects"
    supports_batched = False

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply DIoU-NMS to detections.

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
            boxes, scores, class_ids = self._diou_nms_agnostic(boxes, scores, class_ids)
        else:
            boxes, scores, class_ids = self._diou_nms_per_class(boxes, scores, class_ids)

        return self._apply_max_det(boxes, scores, class_ids)

    def _diou_nms_per_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DIoU-NMS separately for each class."""
        unique_classes = torch.unique(class_ids)
        keep_indices_list = []

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            cls_keep = diou_nms_kernel(cls_boxes, cls_scores, self.config.iou_thres, use_ciou=False)

            # Map back to original indices
            cls_indices = torch.where(cls_mask)[0]
            keep_indices_list.append(cls_indices[cls_keep])

        if not keep_indices_list:
            return self._empty_result(boxes.device)

        keep_indices = torch.cat(keep_indices_list)
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    def _diou_nms_agnostic(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DIoU-NMS across all classes (class-agnostic)."""
        keep = diou_nms_kernel(boxes, scores, self.config.iou_thres, use_ciou=False)
        return boxes[keep], scores[keep], class_ids[keep]


class CIoUNMS(BasePostProcessor):
    """
    Complete-IoU (CIoU) based NMS.

    CIoU extends DIoU by adding an aspect ratio consistency term,
    which considers both the distance between centers and the
    similarity in box shapes.

    Benefits:
        - All benefits of DIoU
        - Better handling of boxes with different aspect ratios
        - More robust in diverse object detection scenarios

    Args:
        config: PostProcessorConfig with thresholds
        **kwargs: Override config parameters

    Example:
        >>> processor = CIoUNMS(iou_thres=0.5)
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "ciou_nms"
    description = "CIoU-NMS - distance and aspect ratio aware suppression"
    supports_batched = False

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply CIoU-NMS to detections.

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
            boxes, scores, class_ids = self._ciou_nms_agnostic(boxes, scores, class_ids)
        else:
            boxes, scores, class_ids = self._ciou_nms_per_class(boxes, scores, class_ids)

        return self._apply_max_det(boxes, scores, class_ids)

    def _ciou_nms_per_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CIoU-NMS separately for each class."""
        unique_classes = torch.unique(class_ids)
        keep_indices_list = []

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            cls_keep = diou_nms_kernel(cls_boxes, cls_scores, self.config.iou_thres, use_ciou=True)

            # Map back to original indices
            cls_indices = torch.where(cls_mask)[0]
            keep_indices_list.append(cls_indices[cls_keep])

        if not keep_indices_list:
            return self._empty_result(boxes.device)

        keep_indices = torch.cat(keep_indices_list)
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    def _ciou_nms_agnostic(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CIoU-NMS across all classes (class-agnostic)."""
        keep = diou_nms_kernel(boxes, scores, self.config.iou_thres, use_ciou=True)
        return boxes[keep], scores[keep], class_ids[keep]
