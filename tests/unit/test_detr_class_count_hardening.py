"""DETR-family class-count synchronization tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit


class _FakeDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.decoder = SimpleNamespace(num_classes=num_classes, reg_max=32)


class _FakeWrapper:
    task = "detect"

    def __init__(self, num_classes: int):
        self.nb_classes = num_classes
        self.names = {i: f"class_{i}" for i in range(num_classes)}
        self.device = torch.device("cpu")
        self.model = _FakeDetector(num_classes)

    def _rebuild_for_new_classes(self, num_classes: int):
        self.nb_classes = num_classes
        self.names = {i: f"class_{i}" for i in range(num_classes)}
        self.model = _FakeDetector(num_classes)


def _write_data_yaml(tmp_path, names=("red", "green", "blue")):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "images" / "train" / "one.jpg").touch()
    (tmp_path / "images" / "val" / "one.jpg").touch()
    data_yaml = tmp_path / "data.yaml"
    quoted_names = ", ".join(names)
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                f"names: [{quoted_names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


@pytest.mark.parametrize(
    ("trainer_import", "size"),
    [
        ("libreyolo.models.rtdetr.trainer:RTDETRTrainer", "r18"),
        ("libreyolo.models.dfine.trainer:DFINETrainer", "n"),
        ("libreyolo.models.deim.trainer:DEIMTrainer", "n"),
        ("libreyolo.models.deimv2.trainer:DEIMv2Trainer", "s"),
    ],
)
def test_detr_trainers_sync_dataset_nc_before_criterion(
    tmp_path, trainer_import, size
):
    module_name, class_name = trainer_import.split(":")
    module = __import__(module_name, fromlist=[class_name])
    trainer_cls = getattr(module, class_name)

    data_yaml = _write_data_yaml(tmp_path)
    wrapper = _FakeWrapper(num_classes=80)
    trainer = trainer_cls(
        model=wrapper.model,
        wrapper_model=wrapper,
        size=size,
        num_classes=80,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        workers=0,
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )

    trainer.on_num_classes_resolved()
    trainer.on_setup()

    assert trainer.num_classes == 3
    assert trainer.config.num_classes == 3
    assert wrapper.nb_classes == 3
    assert trainer.model.decoder.num_classes == 3
    assert trainer.criterion.num_classes == 3
    if hasattr(trainer.criterion, "empty_weight"):
        assert trainer.criterion.empty_weight.shape[0] == 4


def test_detr_trainer_without_rebuild_path_fails_before_loss(tmp_path):
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    data_yaml = _write_data_yaml(tmp_path, names=("one", "two"))
    trainer = RTDETRTrainer(
        model=_FakeDetector(num_classes=80),
        wrapper_model=None,
        size="r18",
        num_classes=80,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        workers=0,
        amp=False,
        ema=False,
        eval_interval=-1,
    )

    with pytest.raises(RuntimeError, match="Pass a wrapper_model"):
        trainer.on_num_classes_resolved()
