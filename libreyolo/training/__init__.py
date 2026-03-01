"""
Training module for LibreYOLO.

Shared training infrastructure (EMA, schedulers, augmentation primitives).
Model-specific trainers live in their respective models/ subdirectories.
"""

# Shared components
from .augment import (
    TrainTransform,
    ValTransform,
    MosaicMixupDataset,
    augment_hsv,
    random_affine,
    preproc,
)
from .scheduler import LRScheduler, LinearLRScheduler, CosineAnnealingScheduler
from .ema import ModelEMA

# Dataset re-export for backward compatibility
from ..data.dataset import YOLODataset, COCODataset, create_dataloader, load_data_config

__all__ = [
    # Dataset
    "YOLODataset",
    "COCODataset",
    "create_dataloader",
    "load_data_config",
    # Augmentation
    "TrainTransform",
    "ValTransform",
    "MosaicMixupDataset",
    "augment_hsv",
    "random_affine",
    "preproc",
    # Scheduler
    "LRScheduler",
    "LinearLRScheduler",
    "CosineAnnealingScheduler",
    # EMA
    "ModelEMA",
]
