"""``train(resume=...)`` continues the saved run: its arguments, directory and file."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit

# Families whose train() resumes through BaseTrainer.resume().
FAMILIES = [
    ("yolo9", "LibreYOLO9", "t"),
    ("yolo9_e2e", "LibreYOLO9E2E", "t"),
    ("yolo9_p2", "LibreYOLO9P2", "t"),
    ("yolox", "LibreYOLOX", "n"),
    ("yolo7", "LibreYOLO7", "b"),
    ("dfine", "LibreDFINE", "n"),
    ("rtdetrv4", "LibreRTDETRv4", "s"),
    ("deim", "LibreDEIM", "n"),
    ("deimv2", "LibreDEIMv2", "atto"),
    ("rtdetr", "LibreRTDETR", "r18"),
    ("rtdetrv2", "LibreRTDETRv2", "r18"),
    ("ec", "LibreEC", "s"),
    ("yolonas", "LibreYOLONAS", "s"),
    ("ppyoloe", "LibrePPYOLOE", "s"),
    ("picodet", "LibrePICODET", "s"),
    ("rtmdet", "LibreRTMDet", "t"),
    ("domedetr", "LibreDOMEDETR", "s"),
    ("tinyformer", "LibreTinyFormer", "s"),
    ("fomo", "LibreFOMO", "s"),
    ("convnext", "LibreConvNeXt", "t"),
    ("convnextv2", "LibreConvNeXtV2", "atto"),
    ("resnet", "LibreResNet", "18"),
    ("mobilenetv4", "LibreMobileNetV4", "s"),
    ("efficientnetv2", "LibreEfficientNetV2", "b0"),
    ("unet", "LibreUNet", "s"),
    ("nafnet", "LibreNAFNet", "s"),
]


def _dataset_yaml(root: Path, label: str, **extra) -> str:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(root / "images" / split / "a.jpg")
        (root / "labels" / split / "a.txt").write_text(label)
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", **extra}
        )
    )
    return str(path)


@pytest.fixture
def detect_yaml(tmp_path):
    return _dataset_yaml(tmp_path / "data", "0 0.5 0.5 0.2 0.2\n", names={0: "a"})


@pytest.fixture
def pose_yaml(tmp_path):
    keypoints = " ".join(["0.5 0.5 2"] * 17)
    return _dataset_yaml(
        tmp_path / "pose",
        f"0 0.5 0.5 0.2 0.2 {keypoints}\n",
        names={0: "person"},
        kpt_shape=[17, 3],
        flip_idx=list(range(17)),
    )


@pytest.fixture
def captured(monkeypatch):
    """Stop BaseTrainer before any real work and record what train() asked for."""
    runs = []

    def setup(self):
        self._is_setup = True

    def resume(self, checkpoint_path):
        runs[-1]["resumed_from"] = checkpoint_path

    def train(self):
        return {"save_dir": str(Path(self.config.project) / self.config.name)}

    real_init = BaseTrainer.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        runs.append({"config": self.config})

    monkeypatch.setattr(BaseTrainer, "__init__", init)
    monkeypatch.setattr(BaseTrainer, "setup", setup)
    monkeypatch.setattr(BaseTrainer, "resume", resume)
    monkeypatch.setattr(BaseTrainer, "train", train)
    return runs


def _save_run_checkpoint(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True)
    torch.save({"epoch": 2, "config": config}, path)
    return path


@pytest.mark.parametrize("module,class_name,size", FAMILIES)
def test_resume_restores_saved_arguments_run_dir_and_checkpoint(
    tmp_path, detect_yaml, captured, module, class_name, size
):
    cls = getattr(importlib.import_module(f"libreyolo.models.{module}.model"), class_name)
    cls(None, size=size, device="cpu").train(data=detect_yaml, epochs=3, device="cpu")
    saved = captured[-1]["config"].to_dict()
    saved.update(epochs=7, batch=3, lr0=0.0123, workers=0, project="elsewhere", name="exp")
    last = _save_run_checkpoint(tmp_path / "runs" / "exp2" / "weights" / "last.pt", saved)

    # resume=<path> from a model that was not loaded from that run.
    cls(None, size=size, device="cpu").train(resume=str(last), device="cpu")
    run = captured[-1]
    assert run["resumed_from"] == str(last)
    assert (run["config"].epochs, run["config"].batch) == (7, 3)
    assert run["config"].lr0 == pytest.approx(0.0123)
    assert run["config"].data == saved["data"]
    # The run keeps writing into the directory the checkpoint came from.
    assert Path(run["config"].project) == tmp_path / "runs"
    assert (run["config"].name, run["config"].exist_ok) == ("exp2", True)

    # Explicit arguments override the saved ones.
    cls(None, size=size, device="cpu").train(
        resume=str(last), epochs=9, project=str(tmp_path / "new"), device="cpu"
    )
    run = captured[-1]
    assert (run["config"].epochs, run["config"].batch) == (9, 3)
    assert Path(run["config"].project) == tmp_path / "new"


@pytest.mark.parametrize(
    "module,class_name,size", [("ec", "LibreEC", "s"), ("yolonas", "LibreYOLONAS", "s")]
)
def test_pose_resume_takes_the_keypoint_layout_from_the_dataset(
    tmp_path, pose_yaml, captured, module, class_name, size
):
    """Pose trainers receive num_keypoints/keypoint_dim from the dataset, so
    restoring the saved copies would pass them twice."""
    cls = getattr(importlib.import_module(f"libreyolo.models.{module}.model"), class_name)
    cls(None, size=size, device="cpu", task="pose").train(data=pose_yaml, device="cpu")
    saved = captured[-1]["config"].to_dict()
    saved.update(epochs=7)
    last = _save_run_checkpoint(tmp_path / "exp" / "weights" / "last.pt", saved)

    cls(None, size=size, device="cpu", task="pose").train(resume=str(last), device="cpu")

    run = captured[-1]
    assert (run["resumed_from"], run["config"].epochs) == (str(last), 7)
    assert (run["config"].num_keypoints, run["config"].keypoint_dim) == (17, 3)


def test_resume_true_continues_the_loaded_run(tmp_path, detect_yaml, captured):
    from libreyolo import LibreYOLO9

    LibreYOLO9(None, size="t", device="cpu").train(data=detect_yaml, epochs=3, device="cpu")
    saved = captured[-1]["config"].to_dict()
    saved.update(epochs=5, imgsz=160, batch=2)
    last = _save_run_checkpoint(tmp_path / "exp" / "weights" / "last.pt", saved)

    model = LibreYOLO9(None, size="t", device="cpu")
    model.model_path = str(last)
    model.train(resume=True, device="cpu")

    run = captured[-1]
    assert run["resumed_from"] == str(last)
    assert (run["config"].epochs, run["config"].imgsz, run["config"].batch) == (5, 160, 2)
    assert (Path(run["config"].project), run["config"].name) == (tmp_path, "exp")


def test_resume_without_a_checkpoint_is_rejected(detect_yaml, captured):
    from libreyolo import LibreYOLO9

    with pytest.raises(ValueError, match="requires a checkpoint"):
        LibreYOLO9(None, size="t", device="cpu").train(data=detect_yaml, resume=True)
    assert captured == []


def test_resume_of_released_weights_fails_before_training(tmp_path, detect_yaml, captured):
    """Released weights carry no epoch or saved arguments; this was a bare
    KeyError('epoch') after the new run directory and loaders were built."""
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", device="cpu")
    released = tmp_path / "weights" / "LibreYOLO9t.pt"
    released.parent.mkdir()
    torch.save({"model": model.model.state_dict(), "nc": 80}, released)
    model.model_path = str(released)

    with pytest.raises(ValueError, match="no training state"):
        model.train(data=detect_yaml, resume=True)
    assert captured == []


class _TinyTrainer(BaseTrainer):
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


def _tiny_trainer(epochs):
    return _TinyTrainer(nn.Linear(1, 1), data=None, device="cpu", ema=False, epochs=epochs)


def test_resume_rejects_weights_without_training_state(tmp_path):
    path = tmp_path / "LibreYOLO9t.pt"
    torch.save({"model": nn.Linear(1, 1).state_dict(), "nc": 80}, path)
    trainer = _tiny_trainer(epochs=3)
    before = trainer.model.weight.detach().clone()

    with pytest.raises(ValueError, match="no training state"):
        trainer.resume(str(path))
    assert torch.equal(before, trainer.model.weight)


def test_resume_rejects_a_finished_run(tmp_path):
    path = tmp_path / "last.pt"
    torch.save({"model": nn.Linear(1, 1).state_dict(), "epoch": 2, "config": {}}, path)

    with pytest.raises(ValueError, match="already trained 3/3 epochs"):
        _tiny_trainer(epochs=3).resume(str(path))
