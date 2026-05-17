"""Unit tests for built-in training artifacts."""

from __future__ import annotations

import csv
import json
import math

import pytest
import torch
from torch import nn

from libreyolo.training.artifacts import TrainingArtifactsCallback
from libreyolo.training.callbacks import TrainEndEvent, TrainEpochEvent, TrainStartEvent
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


def _start_event(save_dir, *, family="yolo9", start_epoch=1):
    return TrainStartEvent(
        start_epoch=start_epoch,
        total_epochs=2,
        model_family=family,
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
    )


def _epoch_event(
    save_dir,
    *,
    family="yolo9",
    epoch=1,
    train_loss_items=None,
    val_metrics=None,
):
    return TrainEpochEvent(
        epoch=epoch,
        total_epochs=2,
        model_family=family,
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        train_loss=1.5 + epoch,
        train_loss_items=train_loss_items or {"box": 0.2, "cls": 0.3},
        lr={"group0": 0.01},
        val_metrics=val_metrics or {"metrics/mAP50": 0.6},
        validated=True,
        is_best=epoch == 1,
        current_metric=0.6,
        current_metric_name="metrics/mAP50",
        best_metric=0.6,
        best_metric_name="metrics/mAP50",
        best_epoch=1,
        epoch_seconds=0.5,
    )


def _end_event(save_dir, *, family="yolo9"):
    return TrainEndEvent(
        total_epochs=2,
        completed_epochs=2,
        model_family=family,
        model_size="s",
        task="detect",
        save_dir=str(save_dir),
        final_loss=2.5,
        best_metric=0.6,
        best_epoch=1,
        total_seconds=4.0,
        results={
            "final_loss": 2.5,
            "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        },
    )


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class ArtifactTrainer(BaseTrainer):
    artifact_model_families = ("yolo9",)

    def get_model_family(self) -> str:
        return "yolo9"

    def get_model_tag(self) -> str:
        return "YOLOv9-s"

    def create_transforms(self):
        raise NotImplementedError

    def create_scheduler(self, iters_per_epoch: int):
        raise NotImplementedError

    def get_loss_components(self, outputs):
        return {}

    def setup(self):
        self.save_dir = self._test_save_dir
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._is_setup = True

    def _train_epoch(self, epoch: int):
        return (
            1.0,
            {
                "mAP50": 0.5,
                "mAP50_95": 0.4,
                "best_metric": 0.4,
                "best_metric_key": "metrics/mAP50-95",
                "metrics": {"metrics/mAP50": 0.5, "metrics/mAP50-95": 0.4},
            },
            {"box": 0.2},
            {"group0": 0.01},
        )

    def _save_checkpoint(
        self, epoch: int, loss: float, val_metrics=None, is_best=None
    ):
        return None


class UnsupportedArtifactTrainer(ArtifactTrainer):
    artifact_model_families = ()

    def get_model_family(self) -> str:
        return "unsupported"


def test_base_trainer_writes_artifacts_for_yolo9(tmp_path):
    trainer = ArtifactTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    rows = _read_csv(tmp_path / "results.csv")
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train/box_loss"] == "0.2"
    assert rows[0]["metrics/mAP50-95"] == "0.4"
    assert (tmp_path / "summary.json").exists()


def test_base_trainer_default_remains_artifact_free(tmp_path):
    trainer = UnsupportedArtifactTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "summary.json").exists()


def test_yolo9_trainer_artifacts_cover_e2e_family():
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.models.yolo9_e2e.trainer import YOLO9E2ETrainer

    assert "yolo9" in YOLO9Trainer.artifact_model_families
    assert "yolo9_e2e" in YOLO9Trainer.artifact_model_families
    assert "yolo9_e2e" in YOLO9E2ETrainer.artifact_model_families


def test_yolonas_trainer_opts_into_artifacts():
    from libreyolo.models.yolonas.pose_trainer import YOLONASPoseTrainer
    from libreyolo.models.yolonas.trainer import YOLONASTrainer

    assert YOLONASTrainer.artifact_model_families == ("yolonas",)
    assert YOLONASPoseTrainer.artifact_model_families == ("yolonas",)


def test_yolonas_pose_trainer_writes_artifacts_through_base_path(tmp_path):
    from libreyolo.models.yolonas.pose_trainer import YOLONASPoseTrainer

    class Wrapper:
        task = "pose"
        names = {0: "object"}

    class StubYOLONASPoseTrainer(YOLONASPoseTrainer):
        def setup(self):
            self.save_dir = self._test_save_dir
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self._is_setup = True

        def _train_epoch(self, epoch: int):
            val_metrics = {
                "best_metric": -0.5,
                "best_metric_key": "loss/val",
                "mAP50": None,
                "mAP50_95": None,
                "metrics": {"loss/val": 0.5},
            }
            return (
                1.0,
                val_metrics,
                {"cls": 0.2, "pose_reg": 0.3},
                {"group0": 0.01},
            )

        def _save_checkpoint(self, epoch, loss, val_metrics=None, is_best=None):
            return None

    trainer = StubYOLONASPoseTrainer(
        model=nn.Linear(1, 1),
        wrapper_model=Wrapper(),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
        num_keypoints=4,
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    rows = _read_csv(tmp_path / "results.csv")
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train/cls_loss"] == "0.2"
    assert rows[0]["train/pose_reg_loss"] == "0.3"
    assert rows[0]["loss/val"] == "0.5"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model_family"] == "yolonas"
    assert summary["task"] == "pose"


def test_yolonas_trainer_writes_artifacts_through_base_path(tmp_path):
    from libreyolo.models.yolonas.trainer import YOLONASTrainer

    class StubYOLONASTrainer(YOLONASTrainer):
        def setup(self):
            self.save_dir = self._test_save_dir
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self._is_setup = True

        def _train_epoch(self, epoch: int):
            return 1.0, None, {"cls": 0.2}, {"group0": 0.01}

        def _save_checkpoint(self, epoch, loss, val_metrics=None, is_best=None):
            return None

    trainer = StubYOLONASTrainer(
        model=nn.Linear(1, 1),
        data=None,
        device="cpu",
        ema=False,
        epochs=1,
    )
    trainer._test_save_dir = tmp_path

    trainer.train()

    rows = _read_csv(tmp_path / "results.csv")
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train/cls_loss"] == "0.2"
    assert (tmp_path / "summary.json").exists()


def test_training_artifacts_write_yolo9_results_csv_and_summary(tmp_path):
    callback = TrainingArtifactsCallback()

    callback.on_train_start(_start_event(tmp_path, family="yolo9"))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9"))
    callback.on_train_end(_end_event(tmp_path, family="yolo9"))

    rows = _read_csv(tmp_path / "results.csv")
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["train/loss"] == "2.5"
    assert rows[0]["train/box_loss"] == "0.2"
    assert rows[0]["train/cls_loss"] == "0.3"
    assert rows[0]["metrics/mAP50"] == "0.6"
    assert rows[0]["lr/group0"] == "0.01"
    assert rows[0]["validated"] == "1"
    assert rows[0]["is_best"] == "1"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model_family"] == "yolo9"
    assert summary["completed_epochs"] == 2
    assert summary["final_loss"] == pytest.approx(2.5)
    assert summary["results"]["best_checkpoint"].endswith("weights/best.pt")


def test_training_artifacts_support_yolonas_and_growing_csv_columns(tmp_path):
    callback = TrainingArtifactsCallback()

    callback.on_train_start(_start_event(tmp_path, family="yolonas"))
    callback.on_train_epoch_end(
        _epoch_event(
            tmp_path,
            family="yolonas",
            epoch=1,
            train_loss_items={"cls": 0.2, "iou": 0.3},
            val_metrics={"mAP50": 0.6},
        )
    )
    callback.on_train_epoch_end(
        _epoch_event(
            tmp_path,
            family="yolonas",
            epoch=2,
            train_loss_items={"cls": 0.4, "iou": 0.5, "dfl": 0.6},
            val_metrics={"mAP50": 0.7, "mAP50_95": 0.55},
        )
    )

    rows = _read_csv(tmp_path / "results.csv")
    assert len(rows) == 2
    assert rows[0]["train/cls_loss"] == "0.2"
    assert rows[0]["train/iou_loss"] == "0.3"
    assert rows[0]["train/dfl_loss"] == ""
    assert rows[0]["metrics/mAP50"] == "0.6"
    assert rows[0]["metrics/mAP50_95"] == ""
    assert rows[1]["train/dfl_loss"] == "0.6"
    assert rows[1]["metrics/mAP50_95"] == "0.55"


def test_training_artifacts_trim_csv_rows_on_resume(tmp_path):
    callback = TrainingArtifactsCallback()

    callback.on_train_start(_start_event(tmp_path, family="yolo9", start_epoch=1))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=1))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=2))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=2))

    callback.on_train_start(_start_event(tmp_path, family="yolo9", start_epoch=2))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=2))

    rows = _read_csv(tmp_path / "results.csv")
    assert [row["epoch"] for row in rows] == ["1", "2"]


def test_training_artifacts_summary_reports_cumulative_resume_history(tmp_path):
    callback = TrainingArtifactsCallback()

    callback.on_train_start(_start_event(tmp_path, family="yolo9", start_epoch=1))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=1))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=2))
    callback.on_train_start(_start_event(tmp_path, family="yolo9", start_epoch=3))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="yolo9", epoch=3))
    callback.on_train_end(
        TrainEndEvent(
            total_epochs=3,
            completed_epochs=1,
            model_family="yolo9",
            model_size="s",
            task="detect",
            save_dir=str(tmp_path),
            final_loss=1.0,
            best_metric=0.6,
            best_epoch=1,
            total_seconds=1.0,
            results={"epoch_metrics": [{"epoch": 3}]},
        )
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_epochs"] == 3
    assert summary["invocation_completed_epochs"] == 1
    assert summary["logged_epochs"] == [1, 2, 3]
    assert summary["results_scope"] == "current_invocation"


def test_training_artifacts_summary_uses_standard_json_for_nonfinite_values(tmp_path):
    callback = TrainingArtifactsCallback()
    event = TrainEndEvent(
        total_epochs=1,
        completed_epochs=1,
        model_family="yolo9",
        model_size="s",
        task="detect",
        save_dir=str(tmp_path),
        final_loss=math.nan,
        best_metric=math.inf,
        best_epoch=1,
        total_seconds=1.0,
        results={"final_loss": math.nan, "best_metric": math.inf},
    )

    callback.on_train_end(event)

    raw = (tmp_path / "summary.json").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw
    summary = json.loads(raw)
    assert summary["final_loss"] is None
    assert summary["best_metric"] is None
    assert summary["results"]["final_loss"] is None


def test_training_artifacts_csv_omits_nonfinite_values(tmp_path):
    callback = TrainingArtifactsCallback()

    callback.on_train_start(_start_event(tmp_path, family="yolo9"))
    callback.on_train_epoch_end(
        TrainEpochEvent(
            epoch=1,
            total_epochs=1,
            model_family="yolo9",
            model_size="s",
            task="detect",
            save_dir=str(tmp_path),
            train_loss=math.nan,
            train_loss_items={"box": math.inf},
            lr={"group0": 0.01},
            val_metrics={"metrics/mAP50": -math.inf},
            validated=True,
            is_best=False,
            current_metric=math.nan,
            current_metric_name="metrics/mAP50",
            best_metric=None,
            best_metric_name=None,
            best_epoch=None,
            epoch_seconds=0.5,
        )
    )

    row = _read_csv(tmp_path / "results.csv")[0]
    assert row["train/loss"] == ""
    assert row["train/box_loss"] == ""
    assert row["metrics/mAP50"] == ""
    assert row["current_metric"] == ""


def test_training_artifacts_ignore_families_outside_allowlist(tmp_path):
    callback = TrainingArtifactsCallback(enabled_families=("yolo9", "yolonas"))

    callback.on_train_start(_start_event(tmp_path, family="dummy"))
    callback.on_train_epoch_end(_epoch_event(tmp_path, family="dummy"))
    callback.on_train_end(_end_event(tmp_path, family="dummy"))

    assert not (tmp_path / "results.csv").exists()
    assert not (tmp_path / "summary.json").exists()
