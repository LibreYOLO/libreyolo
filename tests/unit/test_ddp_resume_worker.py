"""A DDP worker rebuilds the parent's model and resumes the parent's run.

Each worker runs ``cls(weights, **_build_init_kw(parent))`` and then
``train(**train_kw)`` with the arguments the parent captured. These tests do
the same in one CPU process, without starting a process group.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import libreyolo.training.ddp_spawn as ddp_spawn
from libreyolo.training.trainer import BaseTrainer
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture
def detect_yaml(tmp_path):
    root = tmp_path / "data"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(root / "images" / split / "a.jpg")
        (root / "labels" / split / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "a", 1: "b"},
            }
        )
    )
    return str(path)


@pytest.fixture
def captured(monkeypatch):
    """Stop BaseTrainer before any real work and record what train() asked for."""
    runs = []
    real_init = BaseTrainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        runs.append({"config": self.config})

    def resume(self, checkpoint_path):
        runs[-1]["resumed_from"] = checkpoint_path

    monkeypatch.setattr(BaseTrainer, "__init__", init)
    monkeypatch.setattr(BaseTrainer, "setup", lambda self: None)
    monkeypatch.setattr(BaseTrainer, "resume", resume)
    monkeypatch.setattr(
        BaseTrainer,
        "train",
        lambda self: {"save_dir": str(Path(self.config.project) / self.config.name)},
    )
    return runs


def _parent_spawn_args(monkeypatch, model, **train_kw):
    """What a multi-GPU ``model.train(**train_kw)`` hands to the workers."""
    spawned = {}

    def spawn_for_model(model_instance, kw, nprocs, *, devices=None, batch_key="batch"):
        spawned.update(model=model_instance, train_kw=dict(kw))
        return {}

    with monkeypatch.context() as patched:
        patched.setattr(ddp_spawn, "spawn_for_model", spawn_for_model)
        patched.setattr(torch.cuda, "is_available", lambda: True)
        model.train(device=[0, 1], **train_kw)
    return spawned["model"], spawned["train_kw"]


def _worker(parent, train_kw):
    """Rebuild the parent's model as ``_libreyolo_ddp_worker`` does, on CPU."""
    init_kw = ddp_spawn._build_init_kw(parent)
    for key in ("_module", "_class", "_class_object"):
        init_kw.pop(key, None)
    init_kw["device"] = "cpu"
    weights = str(parent.model_path) if train_kw.get("resume") else None
    return type(parent)(weights, **init_kw), dict(train_kw, device="cpu")


@pytest.mark.parametrize(
    "module,class_name",
    [
        ("yolo9", "LibreYOLO9"),
        ("yolo9_e2e", "LibreYOLO9E2E"),
        ("yolo9_p2", "LibreYOLO9P2"),
    ],
)
def test_yolo9_worker_loads_a_fine_tune_with_coco_width_class_towers(
    tmp_path, detect_yaml, captured, monkeypatch, module, class_name
):
    """Fine-tuning a COCO checkpoint on 2 classes keeps its 80-wide class
    towers. A fresh 2-class YOLO9-t build has 64-wide towers, so the worker's
    ``cls(last.pt, nb_classes=2)`` failed with a size mismatch."""
    from libreyolo import LibreYOLO

    cls = getattr(importlib.import_module(f"libreyolo.models.{module}.model"), class_name)
    cls(None, size="t", device="cpu").train(data=detect_yaml, epochs=3, device="cpu")
    config = captured[-1]["config"].to_dict()
    config.update(project=str(tmp_path / "runs"), name="dr", exist_ok=True)

    tuned = cls(None, size="t", device="cpu")
    tuned._rebuild_for_new_classes(2)
    last = tmp_path / "runs" / "dr" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    checkpoint = wrap_libreyolo_checkpoint(
        tuned.model.state_dict(),
        model_family=tuned._get_model_name(),
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    torch.save({**checkpoint, "epoch": 0, "config": config}, last)

    parent, train_kw = _parent_spawn_args(
        monkeypatch, LibreYOLO(str(last), device="cpu"), data=detect_yaml, resume=True
    )
    worker, train_kw = _worker(parent, train_kw)

    expected = parent.model.state_dict()
    loaded = worker.model.state_dict()
    assert loaded.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(loaded[key], value), key

    worker.train(**train_kw)
    run = captured[-1]
    assert run["resumed_from"] == str(last)
    assert (Path(run["config"].project), run["config"].name) == (tmp_path / "runs", "dr")
