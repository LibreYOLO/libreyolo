"""
YOLOv9 module for LibreYOLO.

This module provides YOLOv9 model implementations with support for
t (tiny), s (small), m (medium), and c (compact/largest) variants.

Now includes training support with:
- YOLO9Trainer: Training loop with v9-specific features
- YOLO9Loss: Task-aligned loss function
"""

from .model import LibreYOLO9
from .nn import LibreYOLO9Model
from .trainer import YOLO9Trainer
from .loss import YOLO9Loss, BoxMatcher, Vec2Box
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset

__all__ = [
    "LibreYOLO9",
    "LibreYOLO9Model",
    "YOLO9Trainer",
    "YOLO9Loss",
    "BoxMatcher",
    "Vec2Box",
    "YOLO9TrainTransform",
    "YOLO9MosaicMixupDataset",
]
