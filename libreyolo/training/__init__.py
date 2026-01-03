"""
Training module for LibreYOLO.

Provides training support for YOLOX and RT-DETR models.
"""

from .config import YOLOXTrainConfig, get_config, YOLOX_CONFIGS
from .trainer import YOLOXTrainer

# RT-DETR training
from .rtdetr import RTDETRTrainConfig, RTDETRTrainer, RTDETRCriterion, HungarianMatcher
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
    # RT-DETR Training
    "RTDETRTrainConfig",
    "RTDETRTrainer",
    "RTDETRCriterion",
    "HungarianMatcher",
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
