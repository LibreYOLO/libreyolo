"""Regression coverage: num_classes must be threaded through for non-OBB
YOLO training and for the plain YOLO-directory-format validation path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator

pytestmark = pytest.mark.unit


def _write_train_dataset(tmp_path, bad_class_id=5, nc=1):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color="white").save(img_dir / "sample.jpg")
    (lbl_dir / "sample.txt").write_text(f"{bad_class_id} 0.5 0.5 0.2 0.2\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "nc": nc,
                "names": [f"c{i}" for i in range(nc)],
            }
        )
    )
    return data_yaml


@pytest.mark.parametrize(
    ("trainer_import", "size"),
    [
        # Uses the shared BaseTrainer._setup_data (libreyolo/training/trainer.py).
        ("libreyolo.models.rtdetr.trainer:RTDETRTrainer", "r18"),
        # DFINE/DEIM mirror BaseTrainer._setup_data by hand (their own
        # _setup_data override, "duplicated from the parent for clarity") to
        # wire a family-specific collate_fn -- same bug, independent copy.
        ("libreyolo.models.dfine.trainer:DFINETrainer", "n"),
        ("libreyolo.models.deim.trainer:DEIMTrainer", "n"),
    ],
)
def test_trainer_setup_data_warns_and_drops_out_of_range_class_non_obb(
    tmp_path, trainer_import, size
):
    """A class id >= nc in a plain detect dataset must be skipped with a
    warning at data-loading time, not silently kept for the loss to choke
    on later."""
    module_name, class_name = trainer_import.split(":")
    module = __import__(module_name, fromlist=[class_name])
    trainer_cls = getattr(module, class_name)

    data_yaml = _write_train_dataset(tmp_path, bad_class_id=5, nc=1)
    trainer = trainer_cls(
        model=torch.nn.Identity(),
        size=size,
        num_classes=1,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    with pytest.warns(UserWarning, match="out of range"):
        trainer._setup_data()

    raw_dataset = trainer.train_loader.dataset.dataset
    assert len(raw_dataset.annotations[0][0]) == 0


def _write_val_dataset(tmp_path, bad_class_id=5):
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color="white").save(img_dir / "sample.jpg")
    (lbl_dir / "sample.txt").write_text(f"{bad_class_id} 0.5 0.5 0.2 0.2\n")


def test_validator_warns_on_plain_yolo_directory_format(tmp_path):
    """``data_dir=`` skips ``load_data_config()``'s file-list pre-resolution
    entirely and lands in the plain "YOLO directory format" branch, which
    used to omit num_classes altogether (unlike the file-list-mode and
    COCO-JSON branches, which always passed it)."""
    _write_val_dataset(tmp_path, bad_class_id=5)

    model = SimpleNamespace(
        nb_classes=1,
        _checkpoint_train_config=lambda: {},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data_dir=str(tmp_path),
        split="val",
        batch_size=1,
        num_workers=0,
        device="cpu",
    )
    validator = DetectionValidator(model, config)

    with pytest.warns(UserWarning, match="out of range"):
        dataloader = validator._setup_dataloader()

    assert len(dataloader.dataset.annotations[0][0]) == 0
