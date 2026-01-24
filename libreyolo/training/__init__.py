"""
Training module for LibreYOLO.

Provides training support for YOLOX models.
"""

from .config import YOLOXTrainConfig, get_config, YOLOX_CONFIGS
from .trainer import YOLOXTrainer
from .dataset import YOLODataset, COCODataset, create_dataloader, load_data_config
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
