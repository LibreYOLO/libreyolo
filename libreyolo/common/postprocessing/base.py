"""
Base classes for post-processing pipelines.

This module defines the abstract base class and configuration for all post-processors.
All post-processing methods (NMS, Soft-NMS, DIoU-NMS, etc.) inherit from BasePostProcessor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import torch


@dataclass
class PostProcessorConfig:
    """
    Configuration for post-processors.

    This dataclass holds all common parameters used by post-processing methods.
    Following the immutable pattern from training/config.py.

    Attributes:
        conf_thres: Confidence threshold for filtering detections (default: 0.25)
        iou_thres: IoU threshold for NMS-based methods (default: 0.45)
        max_det: Maximum number of detections to return (default: 300)
        agnostic: If True, apply class-agnostic NMS (default: False)
    """
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    max_det: int = 300
    agnostic: bool = False

    def update(self, **kwargs) -> "PostProcessorConfig":
        """
        Create a new config with updated values (immutable pattern).

        Args:
            **kwargs: Parameters to update

        Returns:
            New PostProcessorConfig with updated values
        """
        current = {
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
            "max_det": self.max_det,
            "agnostic": self.agnostic,
        }
        current.update(kwargs)
        return PostProcessorConfig(**current)


class BasePostProcessor(ABC):
    """
    Abstract base class for all post-processors.

    All post-processing methods (NMS, Soft-NMS, Confluence, etc.) must inherit
    from this class and implement the __call__ method.

    Class Attributes:
        name: Short identifier for the post-processor
        description: Human-readable description of what this method does
        supports_batched: Whether this method supports batched processing

    Example:
        >>> class MyNMS(BasePostProcessor):
        ...     name = "my_nms"
        ...     description = "My custom NMS implementation"
        ...
        ...     def __call__(self, boxes, scores, class_ids, **kwargs):
        ...         # Custom NMS logic
        ...         return filtered_boxes, filtered_scores, filtered_class_ids
    """

    # Metadata - subclasses should override these
    name: str = "base"
    description: str = "Base post-processor (not usable directly)"
    supports_batched: bool = False

    def __init__(self, config: Optional[PostProcessorConfig] = None, **kwargs):
        """
        Initialize the post-processor.

        Args:
            config: PostProcessorConfig instance. If None, creates default config.
            **kwargs: Additional parameters to update in the config.
        """
        self.config = config or PostProcessorConfig()
        if kwargs:
            # Filter only valid config keys
            valid_keys = {"conf_thres", "iou_thres", "max_det", "agnostic"}
            config_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            if config_kwargs:
                self.config = self.config.update(**config_kwargs)

    @abstractmethod
    def __call__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply post-processing to detections.

        This is the core method that each post-processor must implement.
        It receives already-decoded and confidence-filtered boxes.

        Args:
            boxes: Tensor of shape (N, 4) in xyxy format
            scores: Tensor of shape (N,) with confidence scores
            class_ids: Tensor of shape (N,) with class indices
            **kwargs: Additional method-specific parameters

        Returns:
            Tuple of (boxes, scores, class_ids) after post-processing:
                - boxes: (M, 4) filtered boxes in xyxy format
                - scores: (M,) filtered scores
                - class_ids: (M,) filtered class IDs
        """
        pass

    def _empty_result(self, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return empty tensors for no-detection case.

        Args:
            device: Target device for tensors

        Returns:
            Tuple of empty tensors for boxes, scores, and class_ids
        """
        device = device or torch.device("cpu")
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
        )

    def _apply_max_det(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Limit detections to max_det by keeping highest scores.

        Args:
            boxes: Tensor of shape (N, 4)
            scores: Tensor of shape (N,)
            class_ids: Tensor of shape (N,)

        Returns:
            Filtered tensors limited to max_det detections
        """
        if len(boxes) <= self.config.max_det:
            return boxes, scores, class_ids

        _, top_indices = torch.topk(scores, self.config.max_det)
        return boxes[top_indices], scores[top_indices], class_ids[top_indices]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
