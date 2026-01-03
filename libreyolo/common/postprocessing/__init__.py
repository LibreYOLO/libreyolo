"""
Post-processing module for LibreYOLO.

This module provides a pluggable post-processing architecture that allows
users to select different NMS and box filtering methods during inference.

Usage:
    # Using the factory function
    from libreyolo.common.postprocessing import get_postprocessor

    processor = get_postprocessor("soft_nms", sigma=0.5, iou_thres=0.45)
    boxes, scores, classes = processor(boxes, scores, classes)

    # Listing available methods
    from libreyolo.common.postprocessing import list_postprocessors
    print(list_postprocessors())

Available post-processors:
    - nms: Standard NMS (default)
    - batched_nms: Optimized batched NMS using torchvision
    - soft_nms: Soft-NMS with Gaussian/linear decay
    - diou_nms: Distance-IoU based NMS
    - ciou_nms: Complete-IoU based NMS

Example with model:
    # The model's __call__ method accepts a postprocessor parameter
    results = model(image, postprocessor="soft_nms", postprocessor_kwargs={"sigma": 0.5})
"""

from .base import BasePostProcessor, PostProcessorConfig
from .nms import StandardNMS, BatchedNMS, nms_kernel
from .soft_nms import SoftNMS, LinearSoftNMS, GaussianSoftNMS
from .diou_nms import DIoUNMS, CIoUNMS
from .box_utils import (
    box_iou,
    box_iou_pairwise,
    box_diou,
    box_ciou,
    box_giou,
    rescale_boxes,
    filter_valid_boxes,
    xyxy_to_xywh,
    xywh_to_xyxy,
)


# Registry of available post-processors
# Follows the same pattern as CAM_METHODS in common/cam/__init__.py
POSTPROCESSORS = {
    # Standard methods
    "nms": StandardNMS,
    "batched_nms": BatchedNMS,

    # Soft-NMS variants
    "soft_nms": SoftNMS,
    "linear_soft_nms": LinearSoftNMS,
    "gaussian_soft_nms": GaussianSoftNMS,

    # Distance-aware methods
    "diou_nms": DIoUNMS,
    "ciou_nms": CIoUNMS,
}


def get_postprocessor(
    name: str = "nms",
    **kwargs
) -> BasePostProcessor:
    """
    Factory function to create a post-processor by name.

    This is the recommended way to instantiate post-processors, as it
    handles validation and provides a clean API.

    Args:
        name: Name of the post-processor (see POSTPROCESSORS for options)
        **kwargs: Arguments passed to the post-processor constructor.
            Common kwargs:
                - conf_thres: Confidence threshold (default: 0.25)
                - iou_thres: IoU threshold (default: 0.45)
                - max_det: Maximum detections (default: 300)
                - agnostic: Class-agnostic NMS (default: False)
            Method-specific kwargs:
                - sigma: Gaussian decay for soft_nms (default: 0.5)
                - method: "gaussian" or "linear" for soft_nms

    Returns:
        Initialized post-processor instance

    Raises:
        ValueError: If the post-processor name is not found

    Example:
        >>> processor = get_postprocessor("soft_nms", sigma=0.5, iou_thres=0.5)
        >>> boxes, scores, classes = processor(boxes, scores, classes)
    """
    if name not in POSTPROCESSORS:
        available = ", ".join(sorted(POSTPROCESSORS.keys()))
        raise ValueError(
            f"Unknown postprocessor '{name}'. Available: {available}"
        )

    return POSTPROCESSORS[name](**kwargs)


def list_postprocessors() -> dict:
    """
    List all available post-processors with their metadata.

    Returns:
        Dictionary mapping post-processor names to their metadata:
            - description: Human-readable description
            - supports_batched: Whether batched processing is supported

    Example:
        >>> for name, info in list_postprocessors().items():
        ...     print(f"{name}: {info['description']}")
    """
    return {
        name: {
            "description": cls.description,
            "supports_batched": cls.supports_batched,
        }
        for name, cls in POSTPROCESSORS.items()
    }


def register_postprocessor(name: str, cls: type):
    """
    Register a custom post-processor.

    This allows users to add their own post-processing methods to the registry.
    The class must inherit from BasePostProcessor.

    Args:
        name: Unique name for the post-processor
        cls: Post-processor class (must inherit from BasePostProcessor)

    Raises:
        TypeError: If cls doesn't inherit from BasePostProcessor
        ValueError: If name is already registered

    Example:
        >>> class MyCustomNMS(BasePostProcessor):
        ...     name = "my_nms"
        ...     description = "My custom NMS"
        ...     def __call__(self, boxes, scores, class_ids, **kwargs):
        ...         # Custom logic
        ...         return boxes, scores, class_ids
        ...
        >>> register_postprocessor("my_nms", MyCustomNMS)
    """
    if not issubclass(cls, BasePostProcessor):
        raise TypeError(
            f"Post-processor class must inherit from BasePostProcessor, "
            f"got {cls.__name__}"
        )

    if name in POSTPROCESSORS:
        raise ValueError(
            f"Post-processor '{name}' is already registered. "
            f"Use a different name or unregister the existing one first."
        )

    POSTPROCESSORS[name] = cls


def unregister_postprocessor(name: str):
    """
    Remove a post-processor from the registry.

    Args:
        name: Name of the post-processor to remove

    Raises:
        KeyError: If the post-processor is not found
    """
    if name not in POSTPROCESSORS:
        raise KeyError(f"Post-processor '{name}' not found in registry")

    del POSTPROCESSORS[name]


# Public API
__all__ = [
    # Base classes
    "BasePostProcessor",
    "PostProcessorConfig",

    # Post-processor classes
    "StandardNMS",
    "BatchedNMS",
    "SoftNMS",
    "LinearSoftNMS",
    "GaussianSoftNMS",
    "DIoUNMS",
    "CIoUNMS",

    # Registry and factory
    "POSTPROCESSORS",
    "get_postprocessor",
    "list_postprocessors",
    "register_postprocessor",
    "unregister_postprocessor",

    # Utilities
    "box_iou",
    "box_iou_pairwise",
    "box_diou",
    "box_ciou",
    "box_giou",
    "rescale_boxes",
    "filter_valid_boxes",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "nms_kernel",
]
