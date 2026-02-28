"""
YOLOv9 module for LibreYOLO.

This module provides YOLOv9 model implementations with support for
t (tiny), s (small), m (medium), and c (compact/largest) variants.

Now includes training support with:
- V9Trainer: Training loop with v9-specific features
- V9TrainConfig: Configuration dataclass for training
- YOLOv9Loss: Task-aligned loss function
"""

from .model import LIBREYOLO9
from .nn import LibreYOLO9Model
from .config import V9TrainConfig
from .trainer import V9Trainer
from .loss import YOLOv9Loss, BoxMatcher, Vec2Box
from .transforms import V9TrainTransform, V9MosaicMixupDataset

__all__ = [
    "LIBREYOLO9",
    "LibreYOLO9Model",
    "V9TrainConfig",
    "V9Trainer",
    "YOLOv9Loss",
    "BoxMatcher",
    "Vec2Box",
    "V9TrainTransform",
    "V9MosaicMixupDataset",
]
