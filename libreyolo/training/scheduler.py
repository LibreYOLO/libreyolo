"""Learning rate schedulers."""

import math
from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """Base class for all LR schedulers."""

    def __init__(self, lr, iters_per_epoch, total_epochs):
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_epochs = total_epochs
        self.total_iters = iters_per_epoch * total_epochs

    @abstractmethod
    def update_lr(self, iters: int) -> float:
        """Return the learning rate for the given iteration."""


class WarmupCosineScheduler(BaseScheduler):
    """
    YOLOX-style cosine scheduler with quadratic warmup and end plateau.

    LR schedule:
    - Warmup: quadratic increase from warmup_lr_start to lr over warmup_epochs
    - Main: cosine annealing from lr to min_lr
    - Plateau: constant min_lr for the final plateau_epochs
    """

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        warmup_lr_start: float = 0.0,
        plateau_epochs: int = 15,
        min_lr_ratio: float = 0.05,
    ):
        super().__init__(lr, iters_per_epoch, total_epochs)
        self.warmup_iters = iters_per_epoch * warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.plateau_iters = iters_per_epoch * plateau_epochs
        self.min_lr = lr * min_lr_ratio

    def update_lr(self, iters: int) -> float:
        if iters <= self.warmup_iters:
            # Quadratic warmup
            lr = (self.lr - self.warmup_lr_start) * pow(
                iters / float(self.warmup_iters), 2
            ) + self.warmup_lr_start
        elif iters >= self.total_iters - self.plateau_iters:
            # Constant min LR during plateau period
            lr = self.min_lr
        else:
            # Cosine annealing
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - self.warmup_iters)
                    / (self.total_iters - self.warmup_iters - self.plateau_iters)
                )
            )
        return lr


class LinearLRScheduler(BaseScheduler):
    """
    Linear learning rate scheduler with warmup.

    LR schedule:
    - Warmup: linear increase from warmup_lr_start to lr over warmup_iters
    - Main: linear decrease from lr to lr * min_lr_ratio over remaining iterations
    """

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_epochs: int = 3,
        warmup_lr_start: float = 0.0001,
        min_lr_ratio: float = 0.01,
    ):
        super().__init__(lr, iters_per_epoch, total_epochs)
        self.warmup_iters = iters_per_epoch * warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.min_lr = lr * min_lr_ratio

    def update_lr(self, iters: int) -> float:
        if iters <= self.warmup_iters:
            # Linear warmup
            if self.warmup_iters > 0:
                lr = (
                    self.lr - self.warmup_lr_start
                ) * iters / self.warmup_iters + self.warmup_lr_start
            else:
                lr = self.lr
        else:
            # Linear decay
            progress = (iters - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            lr = self.lr - (self.lr - self.min_lr) * progress
        return lr


class FlatCosineScheduler(BaseScheduler):
    """Flat learning rate after warmup, with cosine decay over the final phase.

    Schedule:
    - Warmup: linear increase from ``warmup_lr_start`` to ``lr`` over
      ``warmup_epochs``.
    - Flat: constant ``lr`` until the last ``no_aug_epochs`` epochs.
    - Cosine: cosine decay from ``lr`` to ``lr * min_lr_ratio`` over the final
      ``no_aug_epochs`` epochs.

    Matches D-FINE's official recipe (warmup → flat) with an extra cosine tail
    to stabilize convergence as augmentation winds down.
    """

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_epochs: int = 2,
        warmup_lr_start: float = 1e-6,
        no_aug_epochs: int = 4,
        min_lr_ratio: float = 0.05,
    ):
        super().__init__(lr, iters_per_epoch, total_epochs)
        self.warmup_iters = iters_per_epoch * warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.cosine_iters = iters_per_epoch * no_aug_epochs
        self.min_lr = lr * min_lr_ratio

    def update_lr(self, iters: int) -> float:
        if iters <= self.warmup_iters:
            if self.warmup_iters > 0:
                return (
                    (self.lr - self.warmup_lr_start) * iters / self.warmup_iters
                    + self.warmup_lr_start
                )
            return self.lr
        cosine_start = self.total_iters - self.cosine_iters
        if iters < cosine_start:
            return self.lr
        progress = min(1.0, (iters - cosine_start) / max(1, self.cosine_iters))
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )


class CosineAnnealingScheduler(BaseScheduler):
    """
    Cosine annealing scheduler with warmup.
    """

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_epochs: int = 3,
        warmup_lr_start: float = 0.0001,
        min_lr_ratio: float = 0.01,
    ):
        super().__init__(lr, iters_per_epoch, total_epochs)
        self.warmup_iters = iters_per_epoch * warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.min_lr = lr * min_lr_ratio

    def update_lr(self, iters: int) -> float:
        if iters <= self.warmup_iters:
            # Linear warmup
            if self.warmup_iters > 0:
                lr = (
                    self.lr - self.warmup_lr_start
                ) * iters / self.warmup_iters + self.warmup_lr_start
            else:
                lr = self.lr
        else:
            # Cosine annealing
            progress = (iters - self.warmup_iters) / max(
                1, self.total_iters - self.warmup_iters
            )
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        return lr
