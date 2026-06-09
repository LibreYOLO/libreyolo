"""Compatibility re-exports; the RF-DETR recipe lives in ``libreyolo.data.augment.rfdetr``."""

from ...data.augment.rfdetr import (
    RFDETRDetTransform,
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
    compute_multi_scale_scales,
)

__all__ = [
    "RFDETRDetTransform",
    "RFDETRSegPassThroughDataset",
    "RFDETRSegTransform",
    "compute_multi_scale_scales",
]
