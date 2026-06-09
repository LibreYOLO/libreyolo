"""Weights & Biases logger built on the public training hooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, epoch_metrics, run_name_for


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "WandbLogger requires the 'wandb' package. "
            "Install it with: pip install libreyolo[wandb]"
        ) from exc
    return wandb


class WandbLogger(BaseLogger):
    """Log training to Weights & Biases.

    Args:
        project: W&B project. Defaults to the ``WANDB_PROJECT`` environment
            variable, then ``"libreyolo"``.
        name: Run name. Defaults to ``<family><size>-<task>``.
        entity: W&B entity (team/user). Defaults to the logged-in account.
        log_checkpoints: Upload ``weights/best.pt`` as a W&B model artifact
            when training ends.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_wandb()
        self.project = project
        self.name = name
        self.entity = entity
        self.log_checkpoints = log_checkpoints
        self._run = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        wandb = _import_wandb()
        self._run = wandb.init(
            project=self.project or os.environ.get("WANDB_PROJECT", "libreyolo"),
            name=self.name or run_name_for(event),
            entity=self.entity,
            config=dict(event.config),
        )

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._run is None:
            return
        self._run.log(epoch_metrics(event), step=event.epoch)

    def _handle_end(self, event: TrainEndEvent) -> None:
        if self._run is None:
            return
        wandb = _import_wandb()
        if event.best_metric is not None:
            self._run.summary["best_metric"] = event.best_metric
            self._run.summary["best_epoch"] = event.best_epoch
        if self.log_checkpoints:
            best = Path(event.save_dir) / "weights" / "best.pt"
            if best.is_file():
                artifact = wandb.Artifact(
                    name=f"{run_name_for(event)}-best", type="model"
                )
                artifact.add_file(str(best))
                self._run.log_artifact(artifact)
        self._run.finish()
        self._run = None

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self._run is None:
            return
        self._run.finish(exit_code=1)
        self._run = None
