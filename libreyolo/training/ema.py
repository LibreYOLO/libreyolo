"""
Exponential Moving Average (EMA) for model weights.

Adapted from the official YOLOX repository.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn


def is_parallel(model):
    """Check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average.

    Keeps a moving average of model parameters and buffers.
    This smoothed version of weights helps training stability.

    From https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate
            updates: Initial update counter
        """
        # Create EMA model (FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # Decay exponential ramp (helps early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA parameters."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict() if is_parallel(model) else model.state_dict()
            )
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()
