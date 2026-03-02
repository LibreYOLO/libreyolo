"""
Training module for LibreYOLO.

Shared training infrastructure (EMA, schedulers, augmentation primitives).
Model-specific trainers live in their respective models/ subdirectories.
"""

from .trainer import BaseTrainer
from .scheduler import BaseScheduler
