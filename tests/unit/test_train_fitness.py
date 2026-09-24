"""Custom fitness controls selection, patience and metadata without changing metrics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from libreyolo.training.callbacks import TrainCallbackList
from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.weight_averaging import MetricGatedAverager
from libreyolo.utils.serialization import (
    CheckpointMetadataError,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

pytestmark = pytest.mark.unit


class LossFitness:
    def __init__(self):
        self.calls = []

    def fitness(self, metrics):
        assert not dist.is_initialized() or dist.get_rank() == 0
        with pytest.raises(TypeError):
            metrics["metrics/loss"] = 0
        self.calls.append(dict(metrics))
        return -metrics["metrics/loss"]


class FitnessTrainer(BaseTrainer):
    """Run the real orchestration and checkpoint writer with tiny epoch work."""

    losses = (3.0, 1.0, 2.0, 1.0, 0.5)
    skipped = ()

    def get_model_family(self):
        return "yolo9"

    def get_model_tag(self):
        return "tiny"

    def create_transforms(self):
        raise NotImplementedError

    def create_scheduler(self, iters_per_epoch):
        raise NotImplementedError

    def get_loss_components(self, outputs):
        return {}

    def setup(self):
        self.save_dir = self._test_save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self._is_setup = True

    def _train_epoch(self, epoch):
        if self.is_distributed:
            # A peer proceeding after rank-zero failure/stop would hang here.
            dist.all_reduce(torch.ones(1))
        with torch.no_grad():
            self.model.weight.fill_(epoch + 1)
        metrics = None if epoch in self.skipped else self._metrics(epoch)
        if self.is_distributed and dist.get_rank() != 0:
            metrics = None
        return 1.0, metrics

    def _metrics(self, epoch):
        return {
            "mAP50": 0.1 * (epoch + 1),
            "mAP50_95": 0.05 * (epoch + 1),
            "best_metric": 0.05 * (epoch + 1),
            "best_metric_key": "metrics/mAP50-95",
            "metrics": {
                "metrics/mAP50-95": 0.05 * (epoch + 1),
                "metrics/loss": self.losses[epoch],
            },
        }

    def _run_validation(self, epoch, *, save_plots=None):
        return self._metrics(epoch)


def make_trainer(tmp_path, **kwargs):
    trainer = FitnessTrainer(
        nn.Linear(1, 1, bias=False),
        data=None,
        device="cpu",
        ema=False,
        epochs=5,
        patience=2,
        no_aug_epochs=0,
        **kwargs,
    )
    trainer._test_save_dir = tmp_path
    return trainer


def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    validate_checkpoint_metadata(checkpoint, strict=True)
    return checkpoint


def test_custom_fitness_selects_checkpoint_and_patience_before_observers(tmp_path):
    scorer = LossFitness()
    observed = []

    def observer(event):
        observed.append(
            load_checkpoint(Path(event.save_dir) / "weights/best.pt")["epoch"]
        )

    trainer = make_trainer(tmp_path, callbacks=[scorer, observer])
    result = trainer.train()
    best = load_checkpoint(result["best_checkpoint"])
    last = load_checkpoint(result["last_checkpoint"])

    assert len(scorer.calls) == len(result["epoch_metrics"]) == 4
    assert observed == [0, 1, 1, 1]
    assert result["best_epoch"] == 2
    assert [e["is_best"] for e in result["epoch_metrics"]] == [True, True, False, False]
    assert [e["current_metric"] for e in result["epoch_metrics"]] == [-3, -1, -2, -1]
    assert all(
        e["current_metric_name"] == "fitness/custom" for e in result["epoch_metrics"]
    )
    assert best["epoch"] == 1 and last["epoch"] == 3
    assert best["model"]["weight"].item() == 2
    assert best["best_metric_value"] == -1
    assert best["fitness_source"] == "callback"
    assert "callbacks" not in best["config"]
    assert result["val_metrics"] == scorer.calls
    assert result["val_metrics"][-1]["metrics/mAP50-95"] == pytest.approx(0.2)


def test_no_scorer_keeps_default_selection_and_checkpoint_contract(tmp_path):
    trainer = make_trainer(tmp_path)
    result = trainer.train()
    assert result["best_epoch"] == 5
    assert all(event["is_best"] for event in result["epoch_metrics"])
    best = load_checkpoint(result["best_checkpoint"])
    assert "fitness_source" not in best
    assert best["best_metric_key"] == "metrics/mAP50-95"
    assert best["best_metric_value"] == 0.25


def test_skipped_validation_keeps_score_name_and_counts_patience_in_epochs(tmp_path):
    scorer = LossFitness()
    trainer = make_trainer(tmp_path, callbacks=scorer)
    trainer.skipped = (2, 3)
    result = trainer.train()
    assert len(scorer.calls) == 2
    assert len(result["epoch_metrics"]) == 4
    assert result["epoch_metrics"][-1]["current_metric"] is None
    assert result["epoch_metrics"][-1]["best_metric_name"] == "fitness/custom"
    assert result["epoch_metrics"][-1]["val_metrics"] == {}
    last = load_checkpoint(result["last_checkpoint"])
    assert last["best_metric_key"] == "fitness/custom"
    assert last["best_metric_value"] == -1


@pytest.mark.parametrize(
    "score",
    [
        None,
        True,
        "0.5",
        [],
        complex(1, 0),
        float("nan"),
        float("inf"),
        -float("inf"),
        torch.ones(1),
        torch.tensor(True),
        torch.tensor(1j),
    ],
)
def test_invalid_score_does_not_update_best_or_write_checkpoint(tmp_path, score):
    trainer = make_trainer(
        tmp_path, callbacks=SimpleNamespace(fitness=lambda metrics: score)
    )
    with pytest.raises((TypeError, ValueError), match="finite real scalar"):
        trainer.train()
    assert trainer.best_epoch == 0
    assert not (tmp_path / "weights/last.pt").exists()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("score", [0, -2.5, np.float32(0.5), torch.tensor(-3.0)])
def test_real_scalars_are_accepted_without_changing_raw_metrics(tmp_path, score):
    trainer = make_trainer(
        tmp_path, callbacks=SimpleNamespace(fitness=lambda metrics: score)
    )
    original = trainer._metrics(0)
    scored = trainer._apply_fitness(original)
    assert scored["best_metric"] == float(score)
    assert original["best_metric"] == 0.05
    assert scored["metrics"] == original["metrics"]


def test_missing_metric_error_propagates_without_checkpoint(tmp_path):
    callback = SimpleNamespace(fitness=lambda metrics: metrics["missing"])
    trainer = make_trainer(tmp_path, callbacks=callback)
    with pytest.raises(KeyError, match="missing"):
        trainer.train()
    assert trainer.best_epoch == 0
    assert not (tmp_path / "weights/last.pt").exists()


def test_fitness_only_objects_and_multiple_scorer_rejection():
    callback = LossFitness()
    callbacks = TrainCallbackList(callback)
    callbacks.on_train_start(None)
    callbacks.on_train_epoch_end(None)
    assert callbacks.fitness == callback.fitness
    with pytest.raises(ValueError, match="At most one"):
        TrainCallbackList([callback, LossFitness()])
    with pytest.raises(ValueError, match="At most one"):
        callbacks.append(LossFitness())
    assert len(callbacks) == 1
    with pytest.raises(TypeError, match="fitness must be callable"):
        TrainCallbackList(SimpleNamespace(fitness=None))


def test_separate_vlm_and_vla_trainers_reject_scorers():
    from libreyolo.models.vla.training.trainer import VLATrainer
    from libreyolo.models.vlm.training.trainer import VLMDetectionTrainer

    for trainer, wrapper in (
        (VLATrainer, SimpleNamespace()),
        (VLMDetectionTrainer, SimpleNamespace(FAMILY="qwen3vl")),
    ):
        with pytest.raises(
            NotImplementedError, match="does not support custom fitness"
        ):
            trainer(wrapper, data="unused.yaml", callbacks=LossFitness())


def test_custom_resume_rejected_before_loading_weights_and_default_resume_works(
    tmp_path,
):
    original = make_trainer(tmp_path / "original", callbacks=LossFitness())
    result = original.train()
    resumed = make_trainer(tmp_path / "resumed")
    before = resumed.model.weight.detach().clone()
    with pytest.raises(ValueError, match="Cannot resume a custom-fitness"):
        resumed.resume(result["last_checkpoint"])
    assert torch.equal(before, resumed.model.weight)
    with pytest.raises(ValueError, match="Custom fitness does not support resume"):
        original.resume(result["last_checkpoint"])

    default = make_trainer(tmp_path / "default")
    default_result = default.train()
    with pytest.raises(ValueError, match="Custom fitness does not support resume"):
        original.resume(default_result["last_checkpoint"])
    resumed.resume(default_result["last_checkpoint"])
    assert resumed.start_epoch == 5
    assert resumed.best_epoch == 5
    assert resumed.best_mAP50_95 == 0.25


def test_average_pool_and_average_checkpoint_use_custom_score(tmp_path):
    scorer = LossFitness()
    trainer = make_trainer(tmp_path, callbacks=scorer)
    trainer._weight_averager = MetricGatedAverager(2)
    result = trainer.train()
    average = load_checkpoint(result["average_checkpoint"])
    assert sorted(trainer._weight_averager.metrics()) == [-1, -1]
    assert average["model"]["weight"].item() == 3  # epochs 2 and 4
    assert (
        average["best_metric_key"] == average["average_metric_key"] == "fitness/custom"
    )
    assert average["fitness_source"] == "callback"
    assert average["average_metric_value"] == -1
    assert len(scorer.calls) == 5  # four epochs plus averaged-weight validation


def test_precise_bn_refresh_can_reverse_stop_decision(tmp_path, monkeypatch):
    scorer = LossFitness()
    trainer = make_trainer(tmp_path, callbacks=scorer)
    monkeypatch.setattr(
        trainer, "_maybe_precise_bn", lambda epoch, force=False: epoch == 3
    )
    refreshed = trainer._metrics(3)
    refreshed["metrics"]["metrics/loss"] = 0.1
    monkeypatch.setattr(trainer, "_validate_epoch", lambda *args, **kwargs: refreshed)
    result = trainer.train()
    assert len(result["epoch_metrics"]) == 5
    assert result["best_epoch"] == 4
    assert result["epoch_metrics"][3]["current_metric"] == -0.1
    assert len(scorer.calls) == 6


@pytest.mark.parametrize(
    "metadata", [{"fitness_source": "unknown"}, {"fitness_source": "callback"}]
)
def test_checkpoint_helper_rejects_invalid_fitness_metadata(metadata):
    with pytest.raises(CheckpointMetadataError, match="fitness"):
        wrap_libreyolo_checkpoint(
            {},
            model_family="yolo9",
            size="t",
            task="detect",
            nc=1,
            imgsz=32,
            **metadata,
        )


def _distributed_worker(rank, init_file, root, failure):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        scorer = LossFitness()
        if failure == "invalid":
            scorer = SimpleNamespace(fitness=lambda metrics: float("nan"))
        elif failure == "exit":

            def exit_fitness(metrics):
                raise SystemExit("fitness interrupted")

            scorer = SimpleNamespace(fitness=exit_fitness)
        trainer = make_trainer(Path(root) / "run", callbacks=scorer)
        try:
            result = trainer.train()
            outcome = {
                "epochs": len(result["epoch_metrics"]),
                "calls": len(scorer.calls),
            }
        except (ValueError, RuntimeError, SystemExit) as exc:
            outcome = {"error": str(exc), "type": type(exc).__name__}
        Path(root, f"rank{rank}.json").write_text(json.dumps(outcome))
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.timeout(90)
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo"
)
@pytest.mark.parametrize("failure", [None, "invalid", "exit"])
def test_distributed_fitness_rank_zero_stop_and_failure(tmp_path, failure):
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "init"), str(tmp_path), failure),
        nprocs=2,
        join=True,
    )
    ranks = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    if failure:
        assert [r["type"] for r in ranks] == [
            "ValueError" if failure == "invalid" else "SystemExit",
            "RuntimeError",
        ]
        message = (
            "finite real scalar" if failure == "invalid" else "fitness interrupted"
        )
        assert all(message in r["error"] for r in ranks)
        assert not (tmp_path / "run/weights/last.pt").exists()
    else:
        assert [r["epochs"] for r in ranks] == [4, 4]
        assert [r["calls"] for r in ranks] == [4, 0]
        assert load_checkpoint(tmp_path / "run/weights/best.pt")["best_epoch"] == 2
