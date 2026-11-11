"""Comet logger built on the public training hooks."""

from __future__ import annotations

import os

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, artifact_paths, epoch_metrics, run_name_for


def _import_comet():
    try:
        import comet_ml
    except ImportError as exc:
        raise ImportError(
            "CometLogger requires the 'comet-ml' package. "
            'Install it with: pip install "libreyolo[comet]"'
        ) from exc
    return comet_ml


class CometLogger(BaseLogger):
    """Log training to Comet.

    Args:
        project_name: Comet project. Defaults to ``COMET_PROJECT_NAME``, then
            ``"libreyolo"``.
        workspace: Comet workspace. Defaults to Comet's configured workspace.
        name: Experiment name. Defaults to ``<family><size>-<task>``.
        api_key: Comet API key. Prefer ``COMET_API_KEY`` or ``comet login``.
        online: Set ``False`` for Comet's offline experiment mode.
        log_artifacts: Upload the standard training result files at train end.
        log_checkpoints: Also upload ``weights/best.pt``.
    """

    def __init__(
        self,
        project_name: str | None = None,
        workspace: str | None = None,
        name: str | None = None,
        api_key: str | None = None,
        online: bool | None = None,
        log_artifacts: bool = True,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_comet()
        self.project_name = project_name
        self.workspace = workspace
        self.name = name
        self.api_key = api_key
        self.online = online
        self.log_artifacts = log_artifacts
        self.log_checkpoints = log_checkpoints
        self._experiment = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        comet = _import_comet()
        kwargs = {
            "project_name": self.project_name
            or os.environ.get("COMET_PROJECT_NAME", "libreyolo"),
            "mode": "create",
        }
        for key, value in (
            ("workspace", self.workspace),
            ("api_key", self.api_key),
            ("online", self.online),
        ):
            if value is not None:
                kwargs[key] = value
        self._experiment = comet.start(**kwargs)
        self._experiment.set_name(self.name or run_name_for(event))
        if event.config:
            self._experiment.log_parameters(dict(event.config))

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._experiment is None:
            return
        self._experiment.log_metrics(
            epoch_metrics(event), step=event.epoch, epoch=event.epoch
        )

    def _handle_end(self, event: TrainEndEvent) -> None:
        if self._experiment is None:
            return
        summary = {"status": "finished"}
        if event.best_metric is not None:
            summary.update(
                best_metric=event.best_metric,
                best_epoch=event.best_epoch,
            )
        self._experiment.log_others(summary)
        if self.log_artifacts:
            for path in artifact_paths(event, log_checkpoints=self.log_checkpoints):
                self._experiment.log_asset(str(path), file_name=path.name)
        self._finish()

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        if self._experiment is not None:
            self._experiment.log_others(
                {
                    "status": "failed",
                    "exception_type": event.exception_type,
                    "exception_message": event.exception_message,
                }
            )
        self._finish()

    def _teardown(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._experiment is None:
            return
        self._experiment.end()
        self._experiment = None
