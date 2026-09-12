"""Offline unit tests for LibreVLA training and validation.

The upstream ``lerobot`` package is replaced by small fakes through the
trainer's single import seam, so the loop, checkpointing, contract file,
callbacks, split logic and the offline validator all run on CPU in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from libreyolo.models.vla.base import LibreVLAModel
from libreyolo.models.vla.checkpoint import CONTRACT_FILENAME, read_contract
from libreyolo.models.vla.training import trainer as trainer_mod
from libreyolo.models.vla.training.data import (
    camera_names,
    camera_rename_map,
    resolve_data_source,
    split_episodes,
)
from libreyolo.models.vla.training.trainer import VLATrainer, VLAValidator
from libreyolo.training.callbacks import TrainEndEvent, TrainEpochEvent, TrainStartEvent

pytestmark = [pytest.mark.unit, pytest.mark.vla]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestSplitEpisodes:
    def test_default_holds_out_tail(self):
        assert split_episodes(10) == (list(range(9)), [9])
        assert split_episodes(20, val_split=0.25) == (
            list(range(15)),
            [15, 16, 17, 18, 19],
        )
        assert split_episodes(1) == ([0], [])
        assert split_episodes(2) == ([0], [1])
        assert split_episodes(5, val_split=0.0) == ([0, 1, 2, 3, 4], [])

    def test_explicit_lists(self):
        assert split_episodes(5, val_episodes=[1, 3]) == ([0, 2, 4], [1, 3])
        assert split_episodes(5, train_episodes=[0], val_episodes=[4]) == ([0], [4])
        with pytest.raises(ValueError, match="overlap"):
            split_episodes(5, train_episodes=[0, 1], val_episodes=[1])
        with pytest.raises(ValueError, match="outside"):
            split_episodes(5, val_episodes=[7])
        with pytest.raises(ValueError):
            split_episodes(0)
        with pytest.raises(ValueError):
            split_episodes(5, val_split=1.0)


def test_camera_rename_map_and_names():
    keys = [
        "observation.images.up",
        "observation.images.side",
        "observation.images.wrist",
    ]
    slots = ["camera1", "camera2"]
    assert camera_rename_map(keys, slots) == {
        "observation.images.up": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }
    assert camera_names(keys, slots) == ["up", "side"]
    assert camera_rename_map(["observation.images.camera1"], ["camera1"]) == {}


def test_resolve_data_source(tmp_path):
    src = resolve_data_source("lerobot/svla_so101_pickplace")
    assert src.repo_id == "lerobot/svla_so101_pickplace" and src.root is None
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    local = resolve_data_source(tmp_path)
    assert local.root == tmp_path.resolve() and local.repo_id == tmp_path.name
    assert local.label == str(tmp_path.resolve())
    with pytest.raises(ValueError, match="owner/name"):
        resolve_data_source("nodataset")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="meta/info.json"):
        resolve_data_source(empty)
    with pytest.raises(ValueError):
        resolve_data_source("")


# ---------------------------------------------------------------------------
# Fake upstream
# ---------------------------------------------------------------------------

T, D, S = 4, 3, 2


class FakeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        return {
            "observation.images.up": torch.rand(3, 8, 8, generator=g),
            "observation.state": torch.rand(1, S, generator=g),
            "action": torch.rand(T, D, generator=g),
            "action_is_pad": torch.tensor([False, False, False, idx % 2 == 1]),
            "task": "fake task",
        }


class FakeMeta:
    total_episodes = 5
    fps = 30
    camera_keys = ["observation.images.up"]
    features = {
        "action": {"names": ["j1", "j2", "j3"]},
        "observation.state": {"names": ["s1", "s2"]},
    }
    stats = {"action": {"mean": torch.zeros(D), "std": torch.ones(D)}}


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(2.0))
        self.saved = []

    def forward(self, batch):
        loss = (self.w * batch["action"].mean()) ** 2
        return loss, {}

    def get_optim_params(self):
        return self.parameters()

    def predict_action_chunk(self, batch):
        return batch["action"] * 0 + self.w

    def reset(self):
        pass

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(b"fake")
        self.saved.append(directory)


class FakeProcessor:
    def __init__(self, name):
        self.name = name
        self.saved = []

    def __call__(self, batch):
        return batch

    def save_pretrained(self, directory):
        self.saved.append(Path(directory))
        (Path(directory) / f"{self.name}.json").write_text("{}")


class FakeConfig:
    def __init__(self):
        self.input_features = {}
        self.output_features = {}
        self.normalization_mapping = {}
        self.chunk_size = T
        self.pretrained_path = None
        self.device = None
        self.saved = []

    def get_optimizer_preset(self):
        return SimpleNamespace(
            lr=1e-1,
            grad_clip_norm=1.0,
            build=lambda params: torch.optim.SGD(params, lr=1e-1),
        )

    def get_scheduler_preset(self):
        return None

    def save_pretrained(self, directory):
        self.saved.append(Path(directory))


class FakeVLA(LibreVLAModel):
    FAMILY = "smolvla"
    FILENAME_PREFIX = "LibreFakeVLA"
    HF_REPOS = {"base": "fake/base"}
    HF_REVISIONS = {"base": "a" * 40}

    def __init__(self, **kwargs):
        super().__init__("base", **kwargs)
        self._config = None
        self.fake_policy = FakePolicy()

    def _ensure_weights(self):
        return "fake-base-dir"

    def _pretrained_config(self, snapshot_dir):
        return FakeConfig()

    def _load_policy(self, snapshot_dir):
        self._config = FakeConfig()
        self._policy = self.fake_policy
        self._preprocessor = FakeProcessor("pre")
        self._postprocessor = FakeProcessor("post")

    @property
    def config(self):
        if self._config is None:
            self._ensure_loaded()
        return self._config

    @property
    def camera_slots(self):
        return ["camera1", "camera2", "camera3"]

    @property
    def state_dim(self):
        return S

    @property
    def action_dim(self):
        return D

    @property
    def chunk_size(self):
        return T

    def _predict_chunk(self, observation):
        return torch.zeros(T, D)


@pytest.fixture
def fake_lerobot(monkeypatch):
    made = {}

    def make_policy(config, ds_meta=None, rename_map=None):
        made["config"] = config
        made["rename_map"] = rename_map
        made["policy"] = FakePolicy()
        return made["policy"]

    def make_pre_post_processors(config, pretrained_path=None, **kwargs):
        made["processor_kwargs"] = kwargs
        return FakeProcessor("pre"), FakeProcessor("post")

    def fake_bundle():
        return (
            lambda repo_id, root=None, episodes=None, delta_timestamps=None: (
                FakeDataset(
                    3
                    * (
                        len(episodes)
                        if episodes is not None
                        else FakeMeta.total_episodes
                    )
                )
            ),
            lambda repo_id, root=None: FakeMeta(),
            lambda config, meta: {"action": [0.0]},
            make_policy,
            make_pre_post_processors,
            lambda stats, rename_map: stats,
        )

    monkeypatch.setattr(trainer_mod, "_lerobot", fake_bundle)
    return made


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Recorder:
    """Object-style callback: plain callables only receive epoch events."""

    def __init__(self):
        self.events = []

    def on_train_start(self, event):
        self.events.append(event)

    def on_train_epoch_end(self, event):
        self.events.append(event)

    def on_train_end(self, event):
        self.events.append(event)

    def on_train_exception(self, event):
        self.events.append(event)


def test_trainer_runs_saves_contract_and_emits_callbacks(fake_lerobot, tmp_path):
    recorder = Recorder()
    events = recorder.events
    model = FakeVLA(device="cpu")
    results = model.train(
        data="fake/dataset",
        epochs=2,
        batch=4,
        output_dir=str(tmp_path / "run"),
        callbacks=[recorder],
        loggers=None,
        seed=1,
    )
    assert results["epochs"] == 2 and results["metric_name"] == "val/loss"
    assert results["train_episodes"] == [0, 1, 2, 3] and results["val_episodes"] == [4]
    assert results["best_epoch"] in (1, 2)
    best, last = Path(results["best"]), Path(results["last"])
    for directory in (best, last):
        assert (directory / CONTRACT_FILENAME).is_file()
        assert (directory / "model.safetensors").is_file()
        assert (directory / "pre.json").is_file() and (
            directory / "post.json"
        ).is_file()
    contract = read_contract(best)
    assert contract["family"] == "smolvla" and contract["size"] == "base"
    assert (
        contract["base_repo"] == "fake/base" and contract["base_revision"] == "a" * 40
    )
    assert contract["cameras"] == ["up"] and contract["fps"] == 30.0
    assert contract["action_names"] == ["j1", "j2", "j3"] and contract[
        "state_names"
    ] == ["s1", "s2"]
    assert contract["chunk_size"] == T and contract["data"] == "fake/dataset"

    # The dataset camera was renamed onto the first policy slot.
    assert fake_lerobot["rename_map"] == {
        "observation.images.up": "observation.images.camera1"
    }
    assert fake_lerobot["config"].pretrained_path == "fake-base-dir"
    overrides = fake_lerobot["processor_kwargs"]["preprocessor_overrides"]
    assert (
        overrides["rename_observations_processor"]["rename_map"]
        == fake_lerobot["rename_map"]
    )
    assert "normalizer_processor" in overrides

    # Loss decreased: SGD on (w * mean)^2 drives w towards zero.
    assert float(fake_lerobot["policy"].w.detach()) < 2.0

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "TrainStartEvent" and kinds[-1] == "TrainEndEvent"
    assert kinds.count("TrainEpochEvent") == 2
    start = next(e for e in events if isinstance(e, TrainStartEvent))
    assert start.task == "act" and start.config["family"] == "smolvla"
    epoch = next(e for e in events if isinstance(e, TrainEpochEvent))
    assert epoch.validated and "val/loss" in epoch.val_metrics
    end = next(e for e in events if isinstance(e, TrainEndEvent))
    assert end.results["best"] == results["best"]


def test_trainer_increments_run_dir_and_honours_max_steps(fake_lerobot, tmp_path):
    model = FakeVLA(device="cpu")
    first = model.train(
        data="fake/dataset",
        epochs=1,
        batch=2,
        output_dir=str(tmp_path / "exp"),
        max_steps=1,
    )
    second = model.train(
        data="fake/dataset",
        epochs=1,
        batch=2,
        output_dir=str(tmp_path / "exp"),
        max_steps=1,
    )
    assert (
        Path(first["save_dir"]).name == "exp"
        and Path(second["save_dir"]).name == "exp2"
    )


def test_trainer_rejects_bad_config(fake_lerobot):
    model = FakeVLA(device="cpu")
    with pytest.raises(ValueError, match="epochs"):
        VLATrainer(model, data="fake/dataset", epochs=0)
    with pytest.raises(ValueError, match="batch"):
        VLATrainer(model, data="fake/dataset", batch=0)
    with pytest.raises(ValueError, match="overlap"):
        model.train(data="fake/dataset", epochs=1, train_episodes=[0], val_episodes=[0])


def test_train_and_val_without_lerobot_raise_install_hint(monkeypatch):
    def missing():
        raise ImportError("LibreVLA models require the 'vla' extra")

    monkeypatch.setattr(trainer_mod, "_lerobot", missing)
    model = FakeVLA(device="cpu")
    with pytest.raises(ImportError, match="vla"):
        model.train(data="fake/dataset", epochs=1)
    with pytest.raises(ImportError, match="vla"):
        model.val(data="fake/dataset")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_reports_action_error_on_held_out_episodes(fake_lerobot):
    model = FakeVLA(device="cpu")
    metrics = model.val(data="fake/dataset", batch=3)
    assert set(metrics) >= {
        "val/action_l1",
        "val/action_mse",
        "val/action_l1_first",
        "val/action_l1_dims",
        "val/steps",
        "episodes",
    }
    assert metrics["episodes"] == [4]
    assert list(metrics["val/action_l1_dims"]) == ["j1", "j2", "j3"]
    assert metrics["val/action_l1"] > 0  # the fake policy predicts a constant
    # Padded steps are excluded: 3 frames x 4 steps minus one pad on odd frames.
    assert metrics["val/steps"] == 3 * T - 1

    all_metrics = VLAValidator(
        model, data="fake/dataset", split="all", max_batches=1
    ).run()
    assert all_metrics["episodes"] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="split"):
        VLAValidator(model, data="fake/dataset", split="test")


def test_val_defaults_to_contract_dataset(fake_lerobot, tmp_path):
    from libreyolo.models.vla.checkpoint import write_contract

    write_contract(
        tmp_path,
        family="smolvla",
        size="base",
        base_repo="fake/base",
        base_revision=None,
        data="fake/dataset",
        fps=30,
        cameras=["up"],
        action_names=["j1", "j2", "j3"],
        state_names=None,
        chunk_size=T,
    )
    model = FakeVLA(device="cpu", checkpoint_dir=str(tmp_path))
    assert model.val()["episodes"] == [4]
    assert (
        json.loads((tmp_path / CONTRACT_FILENAME).read_text())["data"] == "fake/dataset"
    )
