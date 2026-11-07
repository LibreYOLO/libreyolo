"""ClearML logger built on the public training hooks."""

from __future__ import annotations

from collections.abc import Sequence

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, artifact_paths, epoch_metrics, run_name_for


def _import_clearml_task():
    try:
        from clearml import Task
    except ImportError as exc:
        raise ImportError(
            "ClearMLLogger requires the 'clearml' package. "
            'Install it with: pip install "libreyolo[clearml]"'
        ) from exc
    return Task


def _metric_parts(name: str) -> tuple[str, str]:
    title, separator, series = name.partition("/")
    return (title, series) if separator else ("metrics", title)


class ClearMLLogger(BaseLogger):
    """Log training to ClearML.

    Args:
        project_name: ClearML project name. Defaults to ``"LibreYOLO"``.
        task_name: ClearML task name. Defaults to ``<family><size>-<task>``.
        tags: Optional ClearML task tags.
        output_uri: Optional ClearML output storage URI.
        log_artifacts: Upload the standard training result files at train end.
        log_checkpoints: Also upload ``weights/best.pt``.
    """

    def __init__(
        self,
        project_name: str = "LibreYOLO",
        task_name: str | None = None,
        tags: Sequence[str] | None = None,
        output_uri: str | None = None,
        log_artifacts: bool = True,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_clearml_task()
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tuple(tags or ())
        self.output_uri = output_uri
        self.log_artifacts = log_artifacts
        self.log_checkpoints = log_checkpoints
        self._task = None
        self._clearml_logger = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        task_cls = _import_clearml_task()
        self._task = task_cls.init(
            project_name=self.project_name,
            task_name=self.task_name or run_name_for(event),
            tags=list(self.tags) or None,
            reuse_last_task_id=False,
            output_uri=self.output_uri,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
        )
        self._clearml_logger = self._task.get_logger()
        if event.config:
            self._task.connect_configuration(
                dict(event.config),
                name="TrainConfig",
                ignore_remote_overrides=True,
            )

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._clearml_logger is None:
            return
        for name, value in epoch_metrics(event).items():
            title, series = _metric_parts(name)
            self._clearml_logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=event.epoch,
            )

    def _handle_end(self, event: TrainEndEvent) -> None:
        if self._task is None:
            return
        if event.best_metric is not None and self._clearml_logger is not None:
            self._clearml_logger.report_single_value("best_metric", event.best_metric)
            if event.best_epoch is not None:
                self._clearml_logger.report_single_value("best_epoch", event.best_epoch)
        if self.log_artifacts:
            for path in artifact_paths(event, log_checkpoints=self.log_checkpoints):
                self._task.upload_artifact(
                    name=path.name,
                    artifact_object=str(path),
                    wait_on_upload=True,
                )
        self._task.close()
        self._task = None
        self._clearml_logger = None

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        self._fail(event.exception_type, event.exception_message)

    def _teardown(self) -> None:
        self._fail(
            "logger backend failure",
            "LibreYOLO disabled ClearML after a logging error",
        )

    def _fail(self, reason: str, message: str) -> None:
        if self._task is None:
            return
        self._task.mark_failed(
            ignore_errors=False,
            status_reason=reason,
            status_message=message,
        )
        self._task = None
        self._clearml_logger = None
