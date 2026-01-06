"""
RT-DETR implementation for LibreYOLO.

RT-DETR (Real-Time DEtection TRansformer) is a transformer-based object detector
that achieves real-time performance while maintaining high accuracy.

This module provides:
- RTDETRModel: The complete RT-DETR neural network architecture
- LIBREYOLORTDETR: High-level wrapper with LibreYOLO-compatible API
- RTDETRLoss: Loss function with Hungarian matcher
- RTDETRTrainer: Training loop
- RTDETRTrainConfig: Training configuration
"""

from .model import LIBREYOLORTDETR
from .nn import RTDETRModel
from .loss import RTDETRLoss, HungarianMatcher
from .trainer import RTDETRTrainer
from .config import RTDETRTrainConfig

__all__ = [
    # Model
    "LIBREYOLORTDETR",
    "RTDETRModel",
    # Loss
    "RTDETRLoss",
    "HungarianMatcher",
    # Training
    "RTDETRTrainer",
    "RTDETRTrainConfig",
]
