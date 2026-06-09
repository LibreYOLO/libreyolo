"""Compatibility re-exports; the RF-DETR pose recipe lives in ``libreyolo.data.augment.rfdetr``."""

from ...data.augment.rfdetr import RFDETRPoseTransform, compute_multi_scale_scales

__all__ = ["RFDETRPoseTransform", "compute_multi_scale_scales"]
