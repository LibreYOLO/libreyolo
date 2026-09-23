"""GTR quadratic warmup, flat phase, cosine decay and final plateau.

Adapted from Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, engine/optim/lr_scheduler.py.
Changes: the shared trainer's scalar update_lr interface and short-run bounds.
"""

import math


class GTRScheduler:
    def __init__(self, lr, iters_per_epoch, config):
        self.lr = lr
        self.min_lr = lr * config.min_lr_ratio
        self.total_iters = config.epochs * iters_per_epoch
        requested_warmup = (
            min(config.warmup_iters, 3 * iters_per_epoch)
            if config.warmup_epochs is None
            else round(config.warmup_epochs * iters_per_epoch)
        )
        # Bound only phases that cannot fit the run. In particular, do not
        # cap valid caller overrides to arbitrary fractions of the budget.
        self.warmup_iters = min(requested_warmup, max(0, self.total_iters - 1))
        self.tail_iters = min(
            config.no_aug_epochs * iters_per_epoch,
            max(0, self.total_iters - self.warmup_iters - 1),
        )
        self.flat_iters = min(
            max(config.flat_epochs * iters_per_epoch, self.warmup_iters),
            max(self.warmup_iters, self.total_iters - self.tail_iters - 1),
        )
        self.warmup_lr_start = min(config.warmup_lr_start, lr)

    def update_lr(self, iters):
        if self.warmup_iters and iters <= self.warmup_iters:
            return (
                self.warmup_lr_start
                + (self.lr - self.warmup_lr_start) * (iters / self.warmup_iters) ** 2
            )
        if iters <= self.flat_iters:
            return self.lr
        if iters >= self.total_iters - self.tail_iters:
            return self.min_lr
        progress = (iters - self.flat_iters) / max(
            1, self.total_iters - self.flat_iters - self.tail_iters
        )
        return self.min_lr + (self.lr - self.min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
