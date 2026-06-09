"""MLflow logger built on the public training hooks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, epoch_metrics, run_name_for

# MLflow accepts alphanumerics, underscores, dashes, periods, spaces and
# slashes in metric/param keys; anything else becomes an underscore.
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")
_MAX_PARAM_VALUE_LENGTH = 500


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflowLogger requires the 'mlflow' package. "
            "Install it with: pip install libreyolo[mlflow]"
        ) from exc
    return mlflow


def _sanitize_key(name: str) -> str:
    return _INVALID_KEY_CHARS.sub("_", name)


class MLflowLogger(BaseLogger):
    """Log training to an MLflow tracking server (or local ``mlruns/``).

    The tracking URI falls back to the standard ``MLFLOW_TRACKING_URI``
    environment variable when not given, matching plain MLflow behaviour.

    Args:
        tracking_uri: MLflow tracking URI. Defaults to MLflow's own
            resolution (env var or local ``mlruns/`` directory).
        experiment_name: Experiment to log under. Defaults to the active
            MLflow experiment.
        run_name: Run name. Defaults to ``<family><size>-<task>``.
        log_artifacts: Upload ``results.csv``, ``train_config.yaml`` and
            ``summary.json`` from the run's save_dir when training ends.
        log_checkpoints: Also upload ``weights/best.pt`` (can be large).
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_artifacts: bool = True,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_mlflow()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_artifacts = log_artifacts
        self.log_checkpoints = log_checkpoints
        self._run_active = False

    def _handle_start(self, event: TrainStartEvent) -> None:
        mlflow = _import_mlflow()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name or run_name_for(event))
        self._run_active = True
        if event.config:
            params = {
                _sanitize_key(key): str(value)[:_MAX_PARAM_VALUE_LENGTH]
                for key, value in event.config.items()
            }
            mlflow.log_params(params)

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if not self._run_active:
            return
        mlflow = _import_mlflow()
        metrics = {
            _sanitize_key(name): value
            for name, value in epoch_metrics(event).items()
        }
        mlflow.log_metrics(metrics, step=event.epoch)

    def _handle_end(self, event: TrainEndEvent) -> None:
        if not self._run_active:
            return
        mlflow = _import_mlflow()
        if self.log_artifacts:
            save_dir = Path(event.save_dir)
            candidates = [
                save_dir / "results.csv",
                save_dir / "train_config.yaml",
                save_dir / "summary.json",
            ]
            if self.log_checkpoints:
                candidates.append(save_dir / "weights" / "best.pt")
            for path in candidates:
                if path.is_file():
                    mlflow.log_artifact(str(path))
        mlflow.end_run(status="FINISHED")
        self._run_active = False

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if not self._run_active:
            return
        mlflow = _import_mlflow()
        mlflow.end_run(status="FAILED")
        self._run_active = False
