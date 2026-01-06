"""
Training module for LibreYOLO.

Provides SHARED training infrastructure (dataset, augment, ema, scheduler).
Model-specific training code (loss, trainer, config) lives in each model folder:
- libreyolo.yolox (YOLOXTrainer, YOLOXTrainConfig, YOLOXLoss)
- libreyolo.rtdetr (RTDETRTrainer, RTDETRTrainConfig, RTDETRLoss)
"""

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
    # Dataset (shared)
    "YOLODataset",
    "COCODataset",
    "create_dataloader",
    "load_data_config",
    # Augmentation (shared)
    "TrainTransform",
    "ValTransform",
    "MosaicMixupDataset",
    "augment_hsv",
    "random_affine",
    "preproc",
    # Scheduler (shared)
    "LRScheduler",
    # EMA (shared)
    "ModelEMA",
]
