"""Neptune logger built on the public training hooks."""

from __future__ import annotations

from collections.abc import Sequence

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, artifact_paths, epoch_metrics, run_name_for


def _import_neptune_run():
    try:
        from neptune_scale import Run
    except ImportError as exc:
        raise ImportError(
            "NeptuneLogger requires the 'neptune-scale' package. "
            'Install it with: pip install "libreyolo[neptune]"'
        ) from exc
    return Run


class NeptuneLogger(BaseLogger):
    """Log training with Neptune's current ``neptune-scale`` client.

    Args:
        project: Neptune project path. Defaults to ``NEPTUNE_PROJECT``.
        api_token: Neptune API token. Prefer ``NEPTUNE_API_TOKEN``.
        name: Experiment name. Defaults to ``<family><size>-<task>``.
        run_id: Optional custom ID, also used by Neptune to resume a run.
        tags: Optional Neptune run tags.
        mode: Neptune mode: ``"async"``, ``"offline"`` or ``"disabled"``.
        capture_console: Capture stdout/stderr into Neptune.
        log_artifacts: Upload the standard training result files at train end.
        log_checkpoints: Also upload ``weights/best.pt``.
    """

    def __init__(
        self,
        project: str | None = None,
        api_token: str | None = None,
        name: str | None = None,
        run_id: str | None = None,
        tags: Sequence[str] | None = None,
        mode: str | None = None,
        capture_console: bool = False,
        log_artifacts: bool = True,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_neptune_run()
        self.project = project
        self.api_token = api_token
        self.name = name
        self.run_id = run_id
        self.tags = tuple(tags or ())
        self.mode = mode
        self.capture_console = capture_console
        self.log_artifacts = log_artifacts
        self.log_checkpoints = log_checkpoints
        self._run = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        run_cls = _import_neptune_run()
        self._run = run_cls(
            project=self.project,
            api_token=self.api_token,
            experiment_name=self.name or run_name_for(event),
            run_id=self.run_id,
            mode=self.mode,
            enable_console_log_capture=self.capture_console,
        )
        if event.config:
            self._run.log_configs(
                data={"config": dict(event.config)},
                flatten=True,
                cast_unsupported=True,
            )
        if self.tags:
            self._run.add_tags(tags=self.tags)

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._run is None:
            return
        self._run.log_metrics(data=epoch_metrics(event), step=event.epoch)

    def _handle_end(self, event: TrainEndEvent) -> None:
        if self._run is None:
            return
        summary = {"run/status": "finished"}
        if event.best_metric is not None:
            summary.update(
                {
                    "run/best_metric": event.best_metric,
                    "run/best_epoch": event.best_epoch,
                }
            )
        self._run.log_configs(data=summary, cast_unsupported=True)
        if self.log_artifacts:
            files = {
                f"artifacts/{path.name}": path
                for path in artifact_paths(event, log_checkpoints=self.log_checkpoints)
            }
            if files:
                self._run.assign_files(files=files)
        self._close()

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        if self._run is not None:
            self._run.log_configs(
                data={
                    "run/status": "failed",
                    "run/exception_type": event.exception_type,
                    "run/exception_message": event.exception_message,
                }
            )
        self._close()

    def _teardown(self) -> None:
        if self._run is None:
            return
        self._run.terminate()
        self._run = None

    def _close(self) -> None:
        if self._run is None:
            return
        self._run.close()
        self._run = None
