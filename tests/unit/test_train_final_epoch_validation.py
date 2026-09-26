"""The final epoch validates, so runs shorter than eval_interval report metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from libreyolo.training.config import TrainConfig
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


def _schedule(eval_interval: int, epochs: int, **config) -> list[int]:
    trainer = SimpleNamespace(
        config=TrainConfig(eval_interval=eval_interval, epochs=epochs, **config)
    )
    trainer._is_final_epoch = lambda epoch: BaseTrainer._is_final_epoch(trainer, epoch)
    return [
        epoch + 1
        for epoch in range(epochs)
        if BaseTrainer._should_validate_epoch(trainer, epoch)
    ]


@pytest.mark.parametrize(
    "eval_interval,epochs,validated",
    [
        (10, 3, [3]),  # shorter than the interval: the final epoch only
        (10, 25, [10, 20, 25]),  # interval kept, final epoch added
        (2, 4, [2, 4]),  # final epoch already scheduled: validated once
        (1, 3, [1, 2, 3]),
        (0, 3, []),  # validation off
        (-1, 3, []),
    ],
)
def test_validation_schedule_includes_the_final_epoch(eval_interval, epochs, validated):
    assert _schedule(eval_interval, epochs) == validated


@pytest.mark.parametrize("config", [{"save_plots": True}, {"precise_bn": 2}])
def test_validation_off_also_skips_final_plots_and_precise_bn_metrics(config):
    """val=False means no validation; final plots and precise BN used to force
    one anyway."""
    assert _schedule(0, 3, **config) == []
    assert _schedule(10, 3, **config) == [3]


def test_val_false_turns_validation_off():
    assert TrainConfig.from_kwargs(val=False, eval_interval=5).eval_interval == 0
    assert TrainConfig.from_kwargs(val=True, eval_interval=5).eval_interval == 5


def test_detr_training_loop_validates_the_final_epoch():
    """D-FINE and DEIM run their own epoch loop; it must follow the same schedule."""
    from libreyolo.models.deim.trainer import DEIMTrainer

    class OneBatchLoader:
        dataset = SimpleNamespace()
        collate_fn = None

        def __iter__(self):
            yield (torch.zeros(1, 3, 16, 16), torch.zeros(1, 2, 5), ((16, 16),), (0,))

        def __len__(self):
            return 1

    param = torch.nn.Parameter(torch.tensor(1.0))
    trainer = DEIMTrainer.__new__(DEIMTrainer)
    trainer.train_loader = OneBatchLoader()
    trainer.config = TrainConfig(epochs=1, eval_interval=10, amp=False)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.device = torch.device("cpu")
    trainer.scaler = None
    trainer.optimizer = torch.optim.SGD([param], lr=0.1)
    trainer.ema_model = None
    trainer.lr_scheduler = SimpleNamespace(update_lr=lambda _: 0.1)
    trainer.get_loss_components = lambda outputs: {}
    trainer.on_forward = lambda *args, **kwargs: {"total_loss": param.sum()}
    trainer._validate_epoch = lambda epoch: {"validated_epoch": epoch}

    _, val_metrics, _, _ = DEIMTrainer._train_epoch(trainer, 0)

    assert val_metrics == {"validated_epoch": 0}


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / "data"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (64, 64), (40 * index, 90, 160)).save(
                root / "images" / split / f"{index}.jpg"
            )
            (root / "labels" / split / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "a"}}
        )
    )
    return str(path)


@pytest.mark.parametrize(
    "val,extra", [(True, {}), (False, {}), (False, {"save_plots": True, "precise_bn": 2})]
)
def test_yolo9_short_run_validates_unless_val_is_off(tmp_path, tiny_dataset, val, extra):
    from libreyolo import LibreYOLO9

    results = LibreYOLO9(None, size="t", device="cpu").train(
        data=tiny_dataset,
        epochs=1,
        batch=2,
        imgsz=64,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="short",
        val=val,
        **extra,
    )

    assert results["epoch_metrics"][-1]["validated"] is val
    assert (tmp_path / "runs" / "short" / "weights" / "best.pt").exists() is val
