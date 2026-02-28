"""
Training module for LibreYOLO.

Shared training infrastructure (EMA, schedulers, augmentation primitives).
Model-specific trainers live in their respective models/ subdirectories.
"""

# YOLOX trainer/config — re-exported for backward compatibility
from ..models.yolox.config import YOLOXTrainConfig, get_config, YOLOX_CONFIGS
from ..models.yolox.trainer import YOLOXTrainer

# Shared components
from .augment import (
    TrainTransform,
    ValTransform,
    MosaicMixupDataset,
    augment_hsv,
    random_affine,
    preproc,
)
from .scheduler import LRScheduler
from .ema import ModelEMA

# Dataset re-export for backward compatibility
from ..data.dataset import YOLODataset, COCODataset, create_dataloader, load_data_config

__all__ = [
    # YOLOX Config
    "YOLOXTrainConfig",
    "get_config",
    "YOLOX_CONFIGS",
    # YOLOX Trainer
    "YOLOXTrainer",
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
    # EMA
    "ModelEMA",
]
