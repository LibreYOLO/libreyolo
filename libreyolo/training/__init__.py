"""Shared training infrastructure (EMA, schedulers, augmentation, config)."""

from .callbacks import (
    TrainCallback as TrainCallback,
    TrainCallbackList as TrainCallbackList,
    TrainCallbacks as TrainCallbacks,
    TrainEpochEvent as TrainEpochEvent,
)
from .config import (
    TrainConfig as TrainConfig,
    YOLOXConfig as YOLOXConfig,
    YOLO9Config as YOLO9Config,
)
