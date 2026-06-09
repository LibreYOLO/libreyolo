"""Compatibility re-exports; the RF-DETR recipe lives in ``libreyolo.data.augment.rfdetr``."""

from ...data.augment.rfdetr import (
    RFDETRDetTransform,
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
    compute_multi_scale_scales,
)
from ...data.obb import normalize_obb_angle, scale_xywhr

__all__ = [
    "RFDETRDetTransform",
    "RFDETRSegPassThroughDataset",
    "RFDETRSegTransform",
    "compute_multi_scale_scales",
    "normalize_obb_angle",
    "scale_xywhr",
]
