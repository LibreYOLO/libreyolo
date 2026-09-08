"""Built-in training artifact writers."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)

logger = logging.getLogger("libreyolo")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` to ``path`` atomically (tmp file + ``os.replace``).

    A reader (the monitor UI or a polling agent) therefore never observes a
    half-written file; it sees either the previous version or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                _json_safe(value), f, allow_nan=False, indent=2, sort_keys=True
            )
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class TrainingArtifactsCallback:
    """Write durable training artifacts from normalized trainer events."""

    def __init__(
        self,
        *,
        enabled_families: Iterable[str] | None = None,
        results_name: str = "results.csv",
        summary_name: str = "summary.json",
    ):
        self.enabled_families = (
            {family.lower() for family in enabled_families}
            if enabled_families is not None
            else None
        )
        self.results_name = results_name
        self.summary_name = summary_name

    def on_train_start(self, event: TrainStartEvent) -> None:
        if not self._enabled(event):
            return

        save_dir = self._save_dir(event)
        if event.start_epoch <= 1:
            for filename in (self.results_name, self.summary_name):
                path = save_dir / filename
                if path.exists():
                    path.unlink()
        else:
            self._trim_csv_before_epoch(
                save_dir / self.results_name,
                start_epoch=event.start_epoch,
            )

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        if not self._enabled(event):
            return

        self._append_csv_row(
            self._save_dir(event) / self.results_name,
            self._epoch_row(event),
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        if not self._enabled(event):
            return

        save_dir = self._save_dir(event)
        results_path = save_dir / self.results_name
        logged_epochs = self._read_logged_epochs(results_path)
        summary = {
            "total_epochs": event.total_epochs,
            "completed_epochs": max(event.completed_epochs, len(logged_epochs)),
            "invocation_completed_epochs": event.completed_epochs,
            "logged_epochs": logged_epochs,
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "final_loss": event.final_loss,
            "best_metric": event.best_metric,
            "best_epoch": event.best_epoch,
            "total_seconds": event.total_seconds,
            "checkpoints": {
                "best": event.results.get("best_checkpoint"),
                "last": event.results.get("last_checkpoint"),
            },
            "results_scope": "current_invocation",
            "results": dict(event.results),
        }
        self._write_json(save_dir / self.summary_name, summary)

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        return None

    def _enabled(
        self,
        event: TrainStartEvent | TrainEpochEvent | TrainEndEvent | TrainExceptionEvent,
    ) -> bool:
        if self.enabled_families is None:
            return True
        return event.model_family.lower() in self.enabled_families

    @staticmethod
    def _save_dir(
        event: TrainStartEvent | TrainEpochEvent | TrainEndEvent | TrainExceptionEvent,
    ) -> Path:
        save_dir = Path(event.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    @classmethod
    def _epoch_row(cls, event: TrainEpochEvent) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": event.epoch,
            "time": event.epoch_seconds,
            "train/loss": event.train_loss,
            "validated": event.validated,
            "is_best": event.is_best,
            "current_metric": event.current_metric,
            "current_metric_name": event.current_metric_name,
            "best_metric": event.best_metric,
            "best_metric_name": event.best_metric_name,
            "best_epoch": event.best_epoch,
        }

        for name, value in event.train_loss_items.items():
            row[cls._train_loss_column(name)] = value
        for name, value in event.val_metrics.items():
            row[cls._metric_column(name)] = value
        for name, value in event.lr.items():
            row[f"lr/{name}"] = value

        return row

    @staticmethod
    def _train_loss_column(name: str) -> str:
        normalized = name.strip().replace(" ", "_")
        if normalized.startswith("train/"):
            return normalized
        if normalized.endswith("_loss"):
            return f"train/{normalized}"
        return f"train/{normalized}_loss"

    @staticmethod
    def _metric_column(name: str) -> str:
        normalized = name.strip().replace(" ", "_")
        if "/" in normalized:
            return normalized
        return f"metrics/{normalized}"

    @classmethod
    def _append_csv_row(cls, path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized_row = {key: cls._csv_value(value) for key, value in row.items()}

        if not path.exists() or path.stat().st_size == 0:
            cls._write_csv(path, list(normalized_row), [normalized_row])
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        new_columns = [key for key in normalized_row if key not in fieldnames]
        if new_columns:
            fieldnames.extend(new_columns)
            rows.append(normalized_row)
            cls._write_csv(path, fieldnames, rows)
            return

        cls._append_csv(path, fieldnames, normalized_row)

    @staticmethod
    def _write_csv(
        path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _append_csv(
        path: Path,
        fieldnames: list[str],
        row: Mapping[str, Any],
    ) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    @classmethod
    def _write_json(cls, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    cls._json_value(value),
                    f,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _read_logged_epochs(path: Path) -> list[int]:
        if not path.exists() or path.stat().st_size == 0:
            return []

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            epochs = []
            for row in reader:
                try:
                    epochs.append(int(row.get("epoch", "")))
                except (TypeError, ValueError):
                    continue
            return epochs

    @classmethod
    def _trim_csv_before_epoch(cls, path: Path, *, start_epoch: int) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if "epoch" not in fieldnames:
                return

            rows = []
            for row in reader:
                try:
                    epoch = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                if epoch < start_epoch:
                    rows.append(row)

        cls._write_csv(path, fieldnames, rows)

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): cls._json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        if isinstance(value, bool):
            return int(value)
        return value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingStatusCallback:
    """Write a live, machine-readable ``status.json`` for every training run.

    Unlike :class:`TrainingArtifactsCallback` (which is family-gated and writes
    the per-epoch ``results.csv``), this callback is always on for every model
    family. It exists to serve two starved consumers of an agent-launched run:

    - an **agent** polling ``status.json`` answers "what epoch / is it alive /
      best metric so far / did it crash" in a few tokens instead of tailing a
      log, and
    - the ``libreyolo monitor`` **web UI**, which relays the same file to a
      browser so a human gets live feedback without the agent in the loop.

    The file is rewritten atomically on every epoch and carries ``state``
    (``running`` / ``completed`` / ``failed``), progress, an ETA derived from
    mean epoch time, the latest and best metrics, and, on failure, the
    exception message. It also tees the ``libreyolo`` console log to
    ``train.log`` in the run directory so the monitor can show the terminal.
    """

    def __init__(
        self,
        *,
        status_name: str = "status.json",
        metrics_name: str = "metrics.jsonl",
        log_name: str = "train.log",
        write_log: bool = True,
    ):
        self.status_name = status_name
        self.metrics_name = metrics_name
        self.log_name = log_name
        self.write_log = write_log
        self._start_time: float | None = None
        self._epoch_time_sum = 0.0
        self._epoch_time_count = 0
        self._base: dict[str, Any] = {}
        self._log_handler: logging.Handler | None = None
        self._log_path: Path | None = None
        self._prev_log_level: int | None = None

    # -- events ------------------------------------------------------------

    def on_train_start(self, event: TrainStartEvent) -> None:
        save_dir = self._save_dir(event)
        self._start_time = time.time()
        self._epoch_time_sum = 0.0
        self._epoch_time_count = 0
        self._base = {
            "schema_version": 1,
            "pid": os.getpid(),
            "model_family": event.model_family,
            "model_size": event.model_size,
            "task": event.task,
            "save_dir": event.save_dir,
            "total_epochs": event.total_epochs,
            "start_epoch": event.start_epoch,
            "started_at": _utcnow_iso(),
        }
        # A fresh run (not a resume) starts the universal metric history clean.
        if event.start_epoch <= 1:
            metrics_path = save_dir / self.metrics_name
            if metrics_path.exists():
                try:
                    metrics_path.unlink()
                except OSError:
                    logger.debug("Could not reset %s", self.metrics_name, exc_info=True)
        self._open_log(save_dir, fresh=event.start_epoch <= 1)
        self._write(
            save_dir,
            state="running",
            completed_epochs=max(event.start_epoch - 1, 0),
            current_epoch=None,
        )

    def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
        save_dir = self._save_dir(event)
        self._epoch_time_sum += event.epoch_seconds
        self._epoch_time_count += 1
        mean_epoch = self._epoch_time_sum / max(self._epoch_time_count, 1)
        # ``event.epoch`` is the one-based number of the epoch that just
        # finished, so it is also the count of completed epochs.
        completed = event.epoch
        remaining = max(event.total_epochs - completed, 0)
        metrics = {
            name.removeprefix("metrics/"): value
            for name, value in event.val_metrics.items()
        }
        # Append the full epoch row to a universal, chart-ready history. Reuses
        # the exact schema of the family-gated results.csv so the monitor can
        # chart every family, gated or not, from a single append-only file.
        self._append_metrics(save_dir, TrainingArtifactsCallback._epoch_row(event))
        self._write(
            save_dir,
            state="running",
            current_epoch=event.epoch,
            completed_epochs=completed,
            epoch_seconds=event.epoch_seconds,
            mean_epoch_seconds=mean_epoch,
            eta_seconds=mean_epoch * remaining,
            train_loss=event.train_loss,
            metrics=metrics,
            validated=event.validated,
            current_metric=event.current_metric,
            current_metric_name=event.current_metric_name,
            best_metric=event.best_metric,
            best_metric_name=event.best_metric_name,
            best_epoch=event.best_epoch,
        )

    def on_train_end(self, event: TrainEndEvent) -> None:
        save_dir = self._save_dir(event)
        self._write(
            save_dir,
            state="completed",
            completed_epochs=event.completed_epochs,
            total_seconds=event.total_seconds,
            train_loss=event.final_loss,
            best_metric=event.best_metric,
            best_epoch=event.best_epoch,
            checkpoints={
                "best": event.results.get("best_checkpoint"),
                "last": event.results.get("last_checkpoint"),
            },
        )
        self._close_log()

    def on_train_exception(self, event: TrainExceptionEvent) -> None:
        save_dir = self._save_dir(event)
        self._write(
            save_dir,
            state="failed",
            current_epoch=event.epoch,
            elapsed_seconds=event.elapsed_seconds,
            error={
                "type": event.exception_type,
                "message": event.exception_message,
            },
        )
        self._close_log()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _save_dir(event: Any) -> Path:
        save_dir = Path(event.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _write(self, save_dir: Path, **fields: Any) -> None:
        payload = dict(self._base)
        payload.update(fields)
        if self._start_time is not None:
            payload.setdefault(
                "elapsed_seconds", round(time.time() - self._start_time, 3)
            )
        total = payload.get("total_epochs") or 0
        completed = payload.get("completed_epochs") or 0
        payload["progress"] = (completed / total) if total else 0.0
        payload["updated_at"] = _utcnow_iso()
        try:
            _atomic_write_json(save_dir / self.status_name, payload)
        except Exception:
            # Status is best-effort telemetry: never let it break a run.
            logger.debug("Failed to write %s", self.status_name, exc_info=True)

    def _append_metrics(self, save_dir: Path, row: Mapping[str, Any]) -> None:
        try:
            line = json.dumps(_json_safe(row), allow_nan=False)
            with open(save_dir / self.metrics_name, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.debug("Failed to append %s", self.metrics_name, exc_info=True)

    def _open_log(self, save_dir: Path, *, fresh: bool) -> None:
        if not self.write_log:
            return
        try:
            self._log_path = save_dir / self.log_name
            if fresh and self._log_path.exists():
                self._log_path.unlink()
            handler = logging.FileHandler(self._log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            lib_logger = logging.getLogger("libreyolo")
            # The handler only sees records the logger admits, so lift the
            # logger to INFO if it is quieter (restored in _close_log).
            if lib_logger.level == logging.NOTSET or lib_logger.level > logging.INFO:
                self._prev_log_level = lib_logger.level
                lib_logger.setLevel(logging.INFO)
            lib_logger.addHandler(handler)
            self._log_handler = handler
        except Exception:
            logger.debug("Failed to open %s", self.log_name, exc_info=True)
            self._log_handler = None

    def _close_log(self) -> None:
        handler = self._log_handler
        if handler is None:
            return
        self._log_handler = None
        try:
            lib_logger = logging.getLogger("libreyolo")
            lib_logger.removeHandler(handler)
            handler.close()
            if self._prev_log_level is not None:
                lib_logger.setLevel(self._prev_log_level)
                self._prev_log_level = None
        except Exception:
            logger.debug("Failed to close %s", self.log_name, exc_info=True)
