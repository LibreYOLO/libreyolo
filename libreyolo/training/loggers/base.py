"""Shared scaffolding for built-in experiment loggers.

Built-in loggers are ordinary :class:`~libreyolo.training.callbacks.TrainCallback`
objects layered on top of the public hook system. Two rules apply to all of
them:

- A missing backend package fails loudly at construction time (the user
  explicitly asked for the logger, so a silent no-op would hide a bug).
- A backend failure at runtime (server down, auth expired, ...) is logged
  once and the logger disables itself — it must never kill a training run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)

logger = logging.getLogger("libreyolo")


def epoch_metrics(event: TrainEpochEvent) -> Dict[str, float]:
    """Flatten a :class:`TrainEpochEvent` into the canonical metric schema.

    The same names are used by every built-in logger so dashboards look
    identical across backends: ``train/loss``, ``train/loss/<component>``,
    ``lr/<group>``, ``val/<metric>`` and ``time/epoch_seconds``.
    """
    metrics: Dict[str, float] = {"train/loss": event.train_loss}
    for name, value in event.train_loss_items.items():
        metrics[f"train/loss/{name}"] = value
    for name, value in event.lr.items():
        metrics[f"lr/{name}"] = value
    for name, value in event.val_metrics.items():
        metrics[f"val/{name.removeprefix('metrics/')}"] = value
    metrics["time/epoch_seconds"] = event.epoch_seconds
    return metrics


def run_name_for(event: Any) -> str:
    """Default run name derived from the model identity, e.g. ``yolo9s-detect``.

    Works with any train event (they all carry model_family/model_size/task).
    """
    family = event.model_family or "model"
    size = event.model_size or ""
    return f"{family}{size}-{event.task}"


class BaseLogger:
    """Base class for built-in loggers: guarded dispatch around the protocol.

    Subclasses implement ``_handle_start`` / ``_handle_epoch_end`` /
    ``_handle_end`` / ``_handle_exception``. Any exception raised by a
    handler disables the logger for the rest of the run instead of
    propagating into the training loop.
    """

    def __init__(self):
        self._disabled = False

    def on_train_start(self, event: TrainStartEvent) -> None:
        self._guarded(self._handle_start, event)

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        self._guarded(self._handle_epoch_end, event)

    def on_train_end(self, event: TrainEndEvent) -> None:
        self._guarded(self._handle_end, event)

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        self._guarded(self._handle_exception, event)

    def _guarded(self, handler, event: Any) -> None:
        if self._disabled:
            return
        try:
            handler(event)
        except Exception:
            self._disabled = True
            logger.exception(
                "%s failed; disabling it for the rest of this run "
                "(training continues)",
                type(self).__name__,
            )
            # Best-effort teardown so an already-opened backend run/writer
            # is not left dangling (e.g. an MLflow run stuck in RUNNING).
            try:
                self._teardown()
            except Exception:
                logger.exception(
                    "%s teardown after failure also failed", type(self).__name__
                )

    def _handle_start(self, event: TrainStartEvent) -> None:
        """Open the backend run/writer."""

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        """Log per-epoch metrics."""

    def _handle_end(self, event: TrainEndEvent) -> None:
        """Upload artifacts and close the run as successful."""

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        """Close the run as failed."""

    def _teardown(self) -> None:
        """Release backend resources after a handler failure (close the
        run/writer as failed). Called once when the logger disables itself."""
