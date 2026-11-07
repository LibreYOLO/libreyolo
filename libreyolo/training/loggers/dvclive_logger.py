"""DVCLive logger built on the public training hooks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)
from .base import BaseLogger, epoch_metrics, run_name_for


def _import_dvclive():
    try:
        from dvclive import Live
    except ImportError as exc:
        raise ImportError(
            "DVCLiveLogger requires the 'dvclive' package. "
            'Install it with: pip install "libreyolo[dvclive]"'
        ) from exc
    return Live


def _safe_param(value: Any) -> Any:
    """Convert resolved config values to DVCLive's supported param types."""
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_param(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_param(item) for item in value]
    return str(value)


def _metric_name(name: str, all_names: set[str]) -> str:
    """Avoid DVCLive's scalar-versus-subtree summary collision.

    DVCLive treats ``/`` as nesting, so it cannot hold both ``train/loss``
    as a float and ``train/loss/box`` as a child. Keep the parent canonical
    and encode the conflicting child separator as a dot: ``train/loss.box``.
    """
    parts = name.split("/")
    for index in range(len(parts) - 1):
        parent = "/".join(parts[: index + 1])
        if parent in all_names:
            suffix = ".".join(parts[index + 1 :])
            return f"{parent}.{suffix}"
    return name


class DVCLiveLogger(BaseLogger):
    """Write training metrics and parameters with DVCLive.

    The safe defaults do not save a DVC experiment, write ``dvc.yaml`` or
    cache files. Enable those DVCLive behaviours explicitly if desired.

    Args:
        log_dir: Output directory. Defaults to ``<save_dir>/dvclive``.
        resume: Resume existing DVCLive history. Defaults to whether the
            LibreYOLO training run itself is resumed.
        report: Optional DVCLive report format (``"html"``, ``"md"`` or
            ``"notebook"``).
        save_dvc_exp: Save a DVC experiment when the run ends. Default false.
        dvcyaml: Optional path to a ``dvc.yaml`` file, or a boolean accepted
            by DVCLive. Default ``None`` disables pipeline-file updates.
        monitor_system: Ask DVCLive to record system metrics.
        log_checkpoints: Register ``weights/best.pt`` as a model artifact.
    """

    def __init__(
        self,
        log_dir: str | None = None,
        resume: bool | None = None,
        report: str | None = None,
        save_dvc_exp: bool = False,
        dvcyaml: str | Path | bool | None = None,
        monitor_system: bool = False,
        log_checkpoints: bool = False,
    ):
        super().__init__()
        _import_dvclive()
        self.log_dir = log_dir
        self.resume = resume
        self.report = report
        self.save_dvc_exp = save_dvc_exp
        self.dvcyaml = dvcyaml
        self.monitor_system = monitor_system
        self.log_checkpoints = log_checkpoints
        self._live = None

    def _handle_start(self, event: TrainStartEvent) -> None:
        live_cls = _import_dvclive()
        self._live = live_cls(
            dir=self.log_dir or str(Path(event.save_dir) / "dvclive"),
            resume=(event.start_epoch > 1 if self.resume is None else self.resume),
            report=self.report,
            save_dvc_exp=self.save_dvc_exp,
            dvcyaml=self.dvcyaml,
            exp_name=run_name_for(event),
            monitor_system=self.monitor_system,
        )
        if event.config:
            self._live.log_params(_safe_param(dict(event.config)))

    def _handle_epoch_end(self, event: TrainEpochEvent) -> None:
        if self._live is None:
            return
        # LibreYOLO events are 1-based. Setting the step explicitly also keeps
        # resumed runs aligned with the trainer rather than a local counter.
        self._live.step = event.epoch
        metrics = epoch_metrics(event)
        names = set(metrics)
        for name, value in metrics.items():
            self._live.log_metric(_metric_name(name, names), value)
        self._live.make_summary()

    def _handle_end(self, event: TrainEndEvent) -> None:
        if self._live is None:
            return
        if event.best_metric is not None:
            self._live.summary["best_metric"] = event.best_metric
            self._live.summary["best_epoch"] = event.best_epoch
        if self.log_checkpoints:
            best = Path(event.save_dir) / "weights" / "best.pt"
            if best.is_file():
                self._live.log_artifact(
                    best,
                    type="model",
                    name=f"{run_name_for(event)}-best",
                    cache=False,
                )
        self._finish()

    def _handle_exception(self, event: TrainExceptionEvent) -> None:
        self._finish()

    def _teardown(self) -> None:
        self._finish()

    def _finish(self) -> None:
        if self._live is None:
            return
        self._live.end()
        self._live = None
