"""
Soft-NMS post-processor.

Soft-NMS decays the scores of overlapping boxes instead of completely
removing them, leading to better recall in crowded scenes.

Reference: https://arxiv.org/abs/1704.04503
"""

import torch
from typing import Tuple, Optional, Literal

from .base import BasePostProcessor, PostProcessorConfig
from .box_utils import box_iou


class SoftNMS(BasePostProcessor):
    """
    Soft Non-Maximum Suppression (Soft-NMS).

    Instead of completely removing overlapping boxes, Soft-NMS decays their
    confidence scores based on the IoU with higher-scoring boxes. This
    preserves more detections and improves recall in dense scenes.

    Benefits:
        - Better recall in crowded/overlapping scenarios
        - Typically +1-2% mAP improvement over standard NMS
        - No additional training required (drop-in replacement)

    Decay methods:
        - "gaussian": score *= exp(-(iou^2) / sigma)
        - "linear": score *= (1 - iou) if iou > threshold

    Args:
        sigma: Gaussian decay parameter (default: 0.5). Lower values = more aggressive decay.
        method: Decay method, either "gaussian" or "linear" (default: "gaussian")
        score_threshold: Minimum score to keep after decay (default: uses conf_thres)
        config: PostProcessorConfig with thresholds
        **kwargs: Override config parameters

    Example:
        >>> processor = SoftNMS(sigma=0.5, method="gaussian")
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "soft_nms"
    description = "Soft-NMS - score decay for overlapping boxes (+1-2% mAP, better recall)"
    supports_batched = False

    def __init__(
        self,
        sigma: float = 0.5,
        method: Literal["gaussian", "linear"] = "gaussian",
        score_threshold: Optional[float] = None,
        config: Optional[PostProcessorConfig] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.sigma = sigma
        self.method = method
        self.score_threshold = score_threshold

    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Soft-NMS to detections.

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
            boxes, scores, class_ids = self._soft_nms_agnostic(boxes, scores, class_ids)
        else:
            boxes, scores, class_ids = self._soft_nms_per_class(boxes, scores, class_ids)

        return self._apply_max_det(boxes, scores, class_ids)

    def _soft_nms_per_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Soft-NMS separately for each class."""
        unique_classes = torch.unique(class_ids)
        keep_boxes, keep_scores, keep_classes = [], [], []

        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            if len(cls_boxes) == 0:
                continue

            final_boxes, final_scores = self._apply_soft_nms(cls_boxes, cls_scores)

            keep_boxes.append(final_boxes)
            keep_scores.append(final_scores)
            keep_classes.append(torch.full(
                (len(final_boxes),), cls.item(),
                dtype=torch.int64, device=boxes.device
            ))

        if not keep_boxes:
            return self._empty_result(boxes.device)

        return (
            torch.cat(keep_boxes),
            torch.cat(keep_scores),
            torch.cat(keep_classes)
        )

    def _soft_nms_agnostic(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Soft-NMS across all classes (class-agnostic)."""
        final_boxes, final_scores = self._apply_soft_nms(boxes, scores)

        # Need to track which original indices survived
        # This is more complex for agnostic mode - we recompute
        if len(final_boxes) == 0:
            return self._empty_result(boxes.device)

        # Match surviving boxes back to original indices
        # Using a simple approach: find closest matches
        keep_classes = []
        for fb in final_boxes:
            dists = ((boxes - fb) ** 2).sum(dim=1)
            idx = dists.argmin()
            keep_classes.append(class_ids[idx].item())

        keep_classes = torch.tensor(keep_classes, dtype=torch.int64, device=boxes.device)
        return final_boxes, final_scores, keep_classes

    def _apply_soft_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core Soft-NMS algorithm.

        Args:
            boxes: (N, 4) boxes in xyxy format
            scores: (N,) confidence scores

        Returns:
            Tuple of (filtered_boxes, filtered_scores)
        """
        # Clone scores since we'll modify them
        scores = scores.clone()

        # Get score threshold
        score_thresh = self.score_threshold or self.config.conf_thres

        # Sort by scores initially
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        scores = scores[order]

        # Track which boxes to keep
        keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

        for i in range(len(boxes)):
            if not keep_mask[i]:
                continue

            # Compute IoU with all remaining boxes
            if i + 1 >= len(boxes):
                break

            remaining_mask = keep_mask.clone()
            remaining_mask[:i+1] = False
            remaining_indices = torch.where(remaining_mask)[0]

            if len(remaining_indices) == 0:
                break

            # Get IoU between current box and remaining boxes
            ious = box_iou(boxes[i:i+1], boxes[remaining_indices])[0]

            # Apply decay based on method
            if self.method == "gaussian":
                decay = torch.exp(-(ious ** 2) / self.sigma)
            else:  # linear
                decay = torch.where(
                    ious > self.config.iou_thres,
                    1 - ious,
                    torch.ones_like(ious)
                )

            # Update scores
            scores[remaining_indices] *= decay

            # Remove boxes below threshold
            keep_mask[remaining_indices] &= scores[remaining_indices] >= score_thresh

        # Return kept boxes and their scores
        final_boxes = boxes[keep_mask]
        final_scores = scores[keep_mask]

        return final_boxes, final_scores


class LinearSoftNMS(SoftNMS):
    """
    Soft-NMS with linear decay method.

    Convenience class that pre-configures Soft-NMS with linear decay.
    Linear decay: score *= (1 - iou) when iou > threshold

    Example:
        >>> processor = LinearSoftNMS(iou_thres=0.5)
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "linear_soft_nms"
    description = "Soft-NMS with linear score decay"

    def __init__(
        self,
        score_threshold: Optional[float] = None,
        config: Optional[PostProcessorConfig] = None,
        **kwargs
    ):
        super().__init__(
            sigma=0.5,  # Not used for linear, but required
            method="linear",
            score_threshold=score_threshold,
            config=config,
            **kwargs
        )


class GaussianSoftNMS(SoftNMS):
    """
    Soft-NMS with Gaussian decay method.

    Convenience class that pre-configures Soft-NMS with Gaussian decay.
    Gaussian decay: score *= exp(-(iou^2) / sigma)

    Args:
        sigma: Gaussian decay parameter (default: 0.5)

    Example:
        >>> processor = GaussianSoftNMS(sigma=0.3)  # More aggressive decay
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """

    name = "gaussian_soft_nms"
    description = "Soft-NMS with Gaussian score decay"

    def __init__(
        self,
        sigma: float = 0.5,
        score_threshold: Optional[float] = None,
        config: Optional[PostProcessorConfig] = None,
        **kwargs
    ):
        super().__init__(
            sigma=sigma,
            method="gaussian",
            score_threshold=score_threshold,
            config=config,
            **kwargs
        )
