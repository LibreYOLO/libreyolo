"""Training configuration for RT-DETR.

Provides a dataclass-based configuration with RT-DETR-specific defaults.
"""

from dataclasses import dataclass
from typing import Tuple

from libreyolo.training.config import TrainConfig


@dataclass(kw_only=True)
class RTDETRConfig(TrainConfig):
    """Training configuration for RT-DETR models."""
    
    # RT-DETR specific optimizer settings
    lr_backbone: float = 0.00001  # Separate backbone learning rate (10x lower than base)
    betas: Tuple[float, float] = (0.9, 0.999)  # AdamW betas
    
    # Default name for RTDETR experiments
    name: str = "rtdetr_exp"
