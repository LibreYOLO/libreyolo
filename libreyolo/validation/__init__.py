"""
Validation module for LibreYOLO.

Provides validation infrastructure for computing detection metrics
including mAP, precision, and recall.

Example:
    >>> from libreyolo import LIBREYOLO
    >>> model = LIBREYOLO("weights/libreyoloXs.pt")
    >>> results = model.val(data="coco8.yaml", batch=16)
    >>> print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")

    Or using the validator directly:
    >>> from libreyolo.validation import DetectionValidator, ValidationConfig
    >>> config = ValidationConfig(data="coco8.yaml", batch_size=16)
    >>> validator = DetectionValidator(model=model, config=config)
    >>> results = validator()
"""

from .base import BaseValidator
from .config import ValidationConfig
from .detection_validator import DetectionValidator
from .metrics import DetMetrics
from .coco_evaluator import COCOEvaluator
from .preprocessors import (
    BaseValPreprocessor,
    StandardValPreprocessor,
    YOLOXValPreprocessor,
    RFDETRValPreprocessor,
)
from .utils import (
    clip_boxes,
    match_predictions_to_gt,
    process_batch,
    scale_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
)

__all__ = [
    # Config
    "ValidationConfig",
    # Validators
    "BaseValidator",
    "DetectionValidator",
    # Preprocessors
    "BaseValPreprocessor",
    "StandardValPreprocessor",
    "YOLOXValPreprocessor",
    "RFDETRValPreprocessor",
    # Metrics
    "DetMetrics",
    "COCOEvaluator",
    # Utilities
    "match_predictions_to_gt",
    "process_batch",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "scale_boxes",
    "clip_boxes",
]
