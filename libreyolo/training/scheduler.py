"""
Learning rate schedulers for YOLOX training.

Adapted from the official YOLOX repository.
"""

import math
from functools import partial


class LRScheduler:
    """Learning rate scheduler with multiple strategies."""

    def __init__(self, name, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Initialize LR scheduler.

        Supported schedulers: [cos, warmcos, yoloxwarmcos, multistep]

        Args:
            name: Scheduler name
            lr: Base learning rate
            iters_per_epoch: Number of iterations per epoch
            total_epochs: Total number of training epochs
            kwargs: Additional scheduler-specific arguments
                - cos: None
                - warmcos: warmup_epochs, warmup_lr_start (default 1e-6)
                - yoloxwarmcos: warmup_epochs, no_aug_epochs, warmup_lr_start, min_lr_ratio
                - multistep: milestones (epochs), gamma (default 0.1)
        """
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

        self.__dict__.update(kwargs)
        self.lr_func = self._get_lr_func(name)

    def update_lr(self, iters):
        """Get learning rate for given iteration."""
        return self.lr_func(iters)

    def _get_lr_func(self, name):
        if name == "cos":
            lr_func = partial(cos_lr, self.lr, self.total_iters)
        elif name == "warmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
            lr_func = partial(
                warm_cos_lr,
                self.lr,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
            )
        elif name == "yoloxwarmcos":
            warmup_total_iters = self.iters_per_epoch * self.warmup_epochs
            no_aug_iters = self.iters_per_epoch * self.no_aug_epochs
            warmup_lr_start = getattr(self, "warmup_lr_start", 0)
            min_lr_ratio = getattr(self, "min_lr_ratio", 0.2)
            lr_func = partial(
                yolox_warm_cos_lr,
                self.lr,
                min_lr_ratio,
                self.total_iters,
                warmup_total_iters,
                warmup_lr_start,
                no_aug_iters,
            )
        elif name == "multistep":
            milestones = [
                int(self.total_iters * milestone / self.total_epochs)
                for milestone in self.milestones
            ]
            gamma = getattr(self, "gamma", 0.1)
            lr_func = partial(multistep_lr, self.lr, milestones, gamma)
        else:
            raise ValueError(f"Scheduler '{name}' not supported.")
        return lr_func


def cos_lr(lr, total_iters, iters):
    """Cosine learning rate."""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def warm_cos_lr(lr, total_iters, warmup_total_iters, warmup_lr_start, iters):
    """Cosine learning rate with warmup."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters)
            )
        )
    return lr


def yolox_warm_cos_lr(
    lr,
    min_lr_ratio,
    total_iters,
    warmup_total_iters,
    warmup_lr_start,
    no_aug_iter,
    iters,
):
    """YOLOX-style cosine learning rate with warmup and no-aug period."""
    min_lr = lr * min_lr_ratio
    if iters <= warmup_total_iters:
        # Quadratic warmup
        lr = (lr - warmup_lr_start) * pow(
            iters / float(warmup_total_iters), 2
        ) + warmup_lr_start
    elif iters >= total_iters - no_aug_iter:
        # Constant min LR during no-aug period
        lr = min_lr
    else:
        # Cosine annealing
        lr = min_lr + 0.5 * (lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
        )
    return lr


def multistep_lr(lr, milestones, gamma, iters):
    """MultiStep learning rate."""
    for milestone in milestones:
        lr *= gamma if iters >= milestone else 1.0
    return lr
