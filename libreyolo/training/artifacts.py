"""Built-in training artifact writers."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .callbacks import (
    TrainEndEvent,
    TrainEpochEvent,
    TrainExceptionEvent,
    TrainStartEvent,
)


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
