"""
YOLOX implementation for LibreYOLO.

This module provides:
- YOLOXModel: The complete YOLOX neural network architecture
- LIBREYOLOX: High-level wrapper with LibreYOLO-compatible API
- YOLOXLoss: Loss function with SimOTA assigner
- YOLOXTrainer: Training loop
- YOLOXTrainConfig: Training configuration
"""

from .model import LIBREYOLOX
from .nn import YOLOXModel
from .loss import YOLOXLoss
from .trainer import YOLOXTrainer
from .config import YOLOXTrainConfig

__all__ = [
    # Model
    "LIBREYOLOX",
    "YOLOXModel",
    # Loss
    "YOLOXLoss",
    # Training
    "YOLOXTrainer",
    "YOLOXTrainConfig",
]
