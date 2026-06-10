"""Standalone, model-free torch ops shared across LibreYOLO.

This package is the home of postprocessing primitives that are useful on
their own, without loading any model. Today it holds the detection-fusion
ops behind :class:`libreyolo.LibreEnsemble`; shared NMS-style helpers used
elsewhere in the library are expected to consolidate here over time.
"""

from .fusion import (
    FUSIONS,
    nms_fusion,
    wbf_seeded,
    weighted_boxes_fusion,
)

__all__ = [
    "FUSIONS",
    "nms_fusion",
    "wbf_seeded",
    "weighted_boxes_fusion",
]
