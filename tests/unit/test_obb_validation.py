"""Unit tests for OBB validation metrics."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.base.model import BaseModel
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.obb_validator import OBBValidator
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor

pytestmark = pytest.mark.unit


class _Core(torch.nn.Module):
    def forward(self, images):
        return images


class _DummyOBBModel:
    nb_classes = 1
    names = {0: "rect"}
    size = "t"
    task = "obb"
    device = torch.device("cpu")
    model = _Core()

    def _get_input_size(self):
        return 64

    def _get_model_name(self):
        return "DummyOBB"

    def _get_val_preprocessor(self, img_size=None):
        img_size = img_size or 64
        return YOLO9ValPreprocessor(img_size=(img_size, img_size))

    def _forward(self, images):
        return images

    def _postprocess(self, *_args, **_kwargs):
        return {
            "boxes": [[22.0, 27.0, 42.0, 37.0]],
            "scores": [0.99],
            "classes": [0],
            "obb": [[32.0, 32.0, 20.0, 10.0, 0.0, 0.99, 0.0]],
            "num_detections": 1,
        }


class _WideDummyOBBModel(_DummyOBBModel):
    nb_classes = 80


class _RectangularDummyOBBModel(_DummyOBBModel):
    def __init__(self):
        self.postprocess_calls = []

    def _postprocess(self, *_args, **kwargs):
        self.postprocess_calls.append(kwargs)
        return {
            "boxes": [[25.0, 15.0, 55.0, 25.0]],
            "scores": [0.99],
            "classes": [0],
            "obb": [[40.0, 20.0, 30.0, 10.0, 0.0, 0.99, 0.0]],
            "num_detections": 1,
        }


def _write_obb_dataset(root: Path) -> Path:
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.34375 0.421875 0.65625 0.421875 "
        "0.65625 0.578125 0.34375 0.578125\n",
        encoding="utf-8",
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(root).replace("\\", "/") + "\n"
        "val: images/val\n"
        "names:\n"
        "  0: rect\n",
        encoding="utf-8",
    )
    return data_yaml


def _write_rectangular_obb_dataset(root: Path) -> Path:
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (80, 40), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.3125 0.375 0.6875 0.375 0.6875 0.625 0.3125 0.625\n",
        encoding="utf-8",
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(root).replace("\\", "/") + "\n"
        "val: images/val\n"
        "names:\n"
        "  0: rect\n",
        encoding="utf-8",
    )
    return data_yaml


def test_obb_validator_metrics_are_perfect_for_exact_prediction():
    validator = OBBValidator.__new__(OBBValidator)
    validator.nc = 1
    validator.config = ValidationConfig(data="unused.yaml", iou_thresholds=(0.5, 0.75))
    validator.iou_thresholds = (0.5, 0.75)
    validator._gt_by_class = {0: {0: [np.array([32.0, 32.0, 20.0, 10.0, 0.0])]}}
    validator._num_gt_by_class = {0: 1}
    validator._predictions_by_class = {
        0: [
            {
                "image_id": 0,
                "score": 0.99,
                "xywhr": np.array([32.0, 32.0, 20.0, 10.0, 0.0]),
            }
        ]
    }

    metrics = validator._compute_metrics()

    assert metrics["metrics/precision"] == pytest.approx(1.0)
    assert metrics["metrics/recall"] == pytest.approx(1.0)
    assert metrics["metrics/mAP50"] == pytest.approx(1.0)
    assert metrics["metrics/mAP50-95"] == pytest.approx(1.0)


def test_obb_validation_rejects_augmented_validation_before_base_tta():
    with pytest.raises(ValueError, match="oriented boxes"):
        BaseModel.val(_DummyOBBModel(), data="unused.yaml", imgsz=64, augment=True)


def test_obb_validator_runs_on_yolo_obb_dataset(tmp_path):
    data_yaml = _write_obb_dataset(tmp_path)
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        iou_thresholds=(0.5, 0.75),
        save_dir=str(tmp_path / "val_run"),
    )

    metrics = OBBValidator(_DummyOBBModel(), config).run()

    assert metrics["metrics/mAP50"] == pytest.approx(1.0)
    assert metrics["metrics/mAP75"] == pytest.approx(1.0)
    assert metrics["metrics/mAP50-95"] == pytest.approx(1.0)
    assert metrics["speed/images_seen"] == 1


def test_obb_validator_uses_original_geometry_for_rectangular_images(tmp_path):
    data_yaml = _write_rectangular_obb_dataset(tmp_path)
    model = _RectangularDummyOBBModel()
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        iou_thresholds=(0.5, 0.75),
        save_dir=str(tmp_path / "val_run"),
    )

    metrics = OBBValidator(model, config).run()

    assert metrics["metrics/mAP50"] == pytest.approx(1.0)
    assert metrics["metrics/mAP75"] == pytest.approx(1.0)
    assert model.postprocess_calls
    assert model.postprocess_calls[0]["original_size"] == (80, 40)
    assert model.postprocess_calls[0]["input_size"] == 64
    assert model.postprocess_calls[0]["letterbox"] is True


def test_obb_validator_infers_nc_from_names_when_nc_is_missing(tmp_path):
    data_yaml = _write_obb_dataset(tmp_path)
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        iou_thresholds=(0.5,),
        save_dir=str(tmp_path / "val_run"),
    )
    validator = OBBValidator(_WideDummyOBBModel(), config)

    metrics = validator.run()

    assert validator.nc == 1
    assert validator.class_names == ["rect"]
    assert metrics["metrics/mAP50"] == pytest.approx(1.0)


def test_obb_validator_pads_short_names_to_nc(tmp_path):
    data_yaml = _write_obb_dataset(tmp_path)
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "val: images/val\n"
        "nc: 2\n"
        "names:\n"
        "  - rect\n",
        encoding="utf-8",
    )
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        iou_thresholds=(0.5,),
        save_dir=str(tmp_path / "val_run"),
    )
    validator = OBBValidator(_WideDummyOBBModel(), config)

    metrics = validator.run()

    assert validator.nc == 2
    assert validator.class_names == ["rect", "class_1"]
    assert metrics["metrics/mAP50"] == pytest.approx(1.0)


def test_obb_validator_skips_invalid_ground_truth_rows(tmp_path, caplog):
    data_yaml = _write_obb_dataset(tmp_path)
    label_file = tmp_path / "labels" / "val" / "sample.txt"
    label_file.write_text(
        label_file.read_text(encoding="utf-8")
        + "0 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=64,
        num_workers=0,
        verbose=False,
        iou_thresholds=(0.5,),
        save_dir=str(tmp_path / "val_run"),
    )

    with caplog.at_level(logging.WARNING):
        metrics = OBBValidator(_DummyOBBModel(), config).run()

    assert metrics["metrics/mAP50"] == pytest.approx(1.0)
    assert "Skipped 1 invalid YOLO OBB label rows" in caplog.text
