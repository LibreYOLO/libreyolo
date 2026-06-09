"""TensorBoard logger built on the public training hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, epoch_metrics


def _import_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoardLogger requires the 'tensorboard' package. "
            "Install it with: pip install libreyolo[tensorboard]"
        ) from exc
    return SummaryWriter


class TensorBoardLogger(BaseLogger):
    """Log training to TensorBoard event files.

    Args:
        log_dir: Directory for event files. Defaults to
            ``<save_dir>/tensorboard`` of the training run.
    """

    def __init__(self, log_dir: Optional[str] = None):
        super().__init__()
        _import_summary_writer()
        self.log_dir = log_dir
        self._writer = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        writer_cls = _import_summary_writer()
        log_dir = self.log_dir or str(Path(event.save_dir) / "tensorboard")
        self._writer = writer_cls(log_dir=log_dir)
        if event.config:
            lines = [f"    {key}: {value}" for key, value in event.config.items()]
            self._writer.add_text("train_config", "\n".join(lines))

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._writer is None:
            return
        for name, value in epoch_metrics(event).items():
            self._writer.add_scalar(name, value, global_step=event.epoch)
        self._writer.flush()

    def _handle_end(self, event: TrainEndEvent) -> None:
        self._close()

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        self._close()

    def _teardown(self) -> None:
        self._close()

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
