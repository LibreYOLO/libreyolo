"""Single-class detection training and validation contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.data.dataset import COCODataset, YOLODataset
from libreyolo.data.utils import load_data_config
from libreyolo.data.yolo_coco_api import YOLOCocoAPI
from libreyolo.models.base.model import _wrap_train_with_cfg
from libreyolo.models.base.model import BaseModel
from libreyolo.models.registry import families_in
from libreyolo.training.config import TrainConfig
from libreyolo.training.trainer import BaseTrainer
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import (
    DetectionValidator,
    _collapse_coco_ground_truth,
)

pytestmark = pytest.mark.unit


def _rng_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _write_yolo_sample(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    Image.new("RGB", (64, 48), color="white").save(image_path)
    label_path.write_text(
        "0 0.25 0.25 0.2 0.2\n2 0.75 0.75 0.2 0.2\n",
        encoding="utf-8",
    )
    return [image_path], [label_path]


def _write_coco_sample(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (64, 48), color="white").save(image_dir / "sample.jpg")
    (annotation_dir / "train.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "sample.jpg",
                        "width": 64,
                        "height": 48,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 3,
                        "bbox": [4, 4, 12, 12],
                        "area": 144,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [32, 24, 12, 12],
                        "area": 144,
                        "iscrowd": 0,
                    },
                ],
                "categories": [
                    {"id": 3, "name": "cat"},
                    {"id": 7, "name": "dog"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_yolo_data_yaml(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    image_dir = dataset_root / "images" / "val"
    label_dir = dataset_root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.25 0.25 0.2 0.2\n2 0.75 0.75 0.2 0.2\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/val",
                "val": "images/val",
                "nc": 3,
                "names": ["cat", "dog", "bird"],
            }
        ),
        encoding="utf-8",
    )
    return yaml_path


def test_single_cls_defaults_off_and_serializes_when_enabled():
    assert TrainConfig().single_cls is False
    assert ValidationConfig(data="unused.yaml").single_cls is False
    assert TrainConfig(single_cls=True).to_dict()["single_cls"] is True


def test_load_data_config_single_cls_is_an_opt_in_view(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "dataset"),
                "train": "images/train",
                "val": "images/val",
                "nc": 3,
                "names": ["cat", "dog", "bird"],
            }
        ),
        encoding="utf-8",
    )

    omitted = load_data_config(str(yaml_path), autodownload=False)
    explicit_false = load_data_config(
        str(yaml_path), autodownload=False, single_cls=False
    )
    collapsed = load_data_config(str(yaml_path), autodownload=False, single_cls=True)

    assert omitted == explicit_false
    assert "_original_names" not in omitted
    assert collapsed["nc"] == 1
    assert collapsed["names"] == {0: "object"}
    assert collapsed["_original_nc"] == 3
    assert collapsed["_original_names"] == ["cat", "dog", "bird"]


def test_base_trainer_resolves_one_class_and_stamps_object_name(tmp_path):
    class _Trainer(BaseTrainer):
        def get_model_family(self):
            return "test"

        def get_model_tag(self):
            return "test"

        def create_transforms(self):
            return None, None

        def create_scheduler(self, iters_per_epoch):
            return None

        def get_loss_components(self, outputs):
            return {}

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "dataset"),
                "train": "images/train",
                "val": "images/val",
                "nc": 3,
                "names": ["cat", "dog", "bird"],
            }
        ),
        encoding="utf-8",
    )
    trainer = _Trainer.__new__(_Trainer)
    trainer.config = TrainConfig(
        data=str(yaml_path),
        num_classes=80,
        single_cls=True,
    )
    trainer.model = SimpleNamespace(num_classes=1)
    trainer.wrapper_model = SimpleNamespace(
        nb_classes=1,
        names={0: "class_0"},
    )

    resolved = trainer._resolve_num_classes_from_data_config()
    trainer._sync_wrapped_model_num_classes(resolved)

    assert resolved == trainer.num_classes == trainer.config.num_classes == 1
    assert trainer.wrapper_model.names == {0: "object"}


def test_yolo_dataset_default_is_bit_identical_and_rng_neutral(tmp_path):
    image_files, label_files = _write_yolo_sample(tmp_path)

    np.random.seed(810)
    rng_before = np.random.get_state()
    omitted = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
    )
    rng_after_omitted = np.random.get_state()

    np.random.seed(810)
    explicit_false = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        single_cls=False,
    )
    rng_after_false = np.random.get_state()

    assert _rng_state_equal(rng_before, rng_after_omitted)
    assert _rng_state_equal(rng_after_omitted, rng_after_false)
    np.testing.assert_array_equal(
        omitted.annotations[0][0], explicit_false.annotations[0][0]
    )


def test_yolo_dataset_single_cls_remaps_every_positive_class(tmp_path):
    image_files, label_files = _write_yolo_sample(tmp_path)

    dataset = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        single_cls=True,
    )

    labels = dataset.annotations[0][0]
    assert labels.shape == (2, 5)
    np.testing.assert_array_equal(labels[:, 4], np.zeros(2))


def _write_yolo_obb_sample(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    image_path = tmp_path / "sample_obb.jpg"
    label_path = tmp_path / "sample_obb.txt"
    Image.new("RGB", (64, 48), color="white").save(image_path)
    label_path.write_text(
        "0 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30\n"
        "2 0.60 0.60 0.80 0.60 0.80 0.80 0.60 0.80\n",
        encoding="utf-8",
    )
    return [image_path], [label_path]


def test_obb_single_cls_remaps_every_positive_class_within_the_bound(tmp_path):
    """single_cls collapses to one class, so num_classes=1 must not drop class 2.

    The YOLO box path remaps before the range check and keeps the row. The OBB
    path validated the source id against num_classes first, so with the class
    count single_cls implies it dropped every row whose source id was not 0.
    """
    image_files, label_files = _write_yolo_obb_sample(tmp_path)

    dataset = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        load_obb=True,
        num_classes=1,
        single_cls=True,
    )

    labels = dataset.annotations[0][0]
    assert labels.shape == (2, 6)
    np.testing.assert_array_equal(labels[:, 4], np.zeros(2))


def test_obb_class_bound_still_applies_without_single_cls(tmp_path):
    image_files, label_files = _write_yolo_obb_sample(tmp_path)

    dataset = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        load_obb=True,
        num_classes=1,
    )

    labels = dataset.annotations[0][0]
    assert labels.shape == (1, 6)
    np.testing.assert_array_equal(labels[:, 4], np.zeros(1))


def test_obb_single_cls_still_rejects_a_negative_class(tmp_path):
    image_path = tmp_path / "negative.jpg"
    label_path = tmp_path / "negative.txt"
    Image.new("RGB", (64, 48), color="white").save(image_path)
    label_path.write_text(
        "-1 0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30\n",
        encoding="utf-8",
    )

    dataset = YOLODataset(
        img_files=[image_path],
        label_files=[label_path],
        img_size=(64, 64),
        load_obb=True,
        num_classes=1,
        single_cls=True,
    )

    assert dataset.annotations[0][0].shape[0] == 0


def test_coco_dataset_single_cls_keeps_original_mapping_then_collapses(tmp_path):
    pytest.importorskip("pycocotools")
    _write_coco_sample(tmp_path)

    omitted = COCODataset(
        data_dir=str(tmp_path),
        json_file="annotations/train.json",
        name="images/train",
        img_size=(64, 64),
        num_classes=2,
        names={0: "cat", 1: "dog"},
    )
    explicit_false = COCODataset(
        data_dir=str(tmp_path),
        json_file="annotations/train.json",
        name="images/train",
        img_size=(64, 64),
        num_classes=2,
        names={0: "cat", 1: "dog"},
        single_cls=False,
    )
    collapsed = COCODataset(
        data_dir=str(tmp_path),
        json_file="annotations/train.json",
        name="images/train",
        img_size=(64, 64),
        num_classes=1,
        names={0: "cat", 1: "dog"},
        single_cls=True,
    )

    np.testing.assert_array_equal(
        omitted.annotations[0][0], explicit_false.annotations[0][0]
    )
    assert collapsed.category_id_to_label == {3: 0, 7: 1}
    assert collapsed.label_to_category_id == {0: 3}
    assert collapsed._classes == ("object",)
    np.testing.assert_array_equal(collapsed.annotations[0][0][:, 4], np.zeros(2))


def test_yolo_coco_api_single_cls_keeps_nonzero_ground_truth(tmp_path):
    image_files, label_files = _write_yolo_sample(tmp_path)

    api = YOLOCocoAPI(
        None,
        None,
        ["object"],
        image_files=image_files,
        label_files=label_files,
        single_cls=True,
    )

    annotations = api.loadAnns()
    assert len(annotations) == 2
    assert {annotation["category_id"] for annotation in annotations} == {0}


def test_native_coco_ground_truth_collapse_is_independent(tmp_path):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    _write_coco_sample(tmp_path)
    original = COCO(str(tmp_path / "annotations" / "train.json"))

    collapsed = _collapse_coco_ground_truth(original, category_id=3)

    assert set(original.getCatIds()) == {3, 7}
    assert set(collapsed.getCatIds()) == {3}
    assert {ann["category_id"] for ann in collapsed.anns.values()} == {3}


def test_single_cls_native_coco_validation_scores_all_collapsed_gt(tmp_path):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    from libreyolo.validation import COCOEvaluator

    _write_coco_sample(tmp_path)
    original = COCO(str(tmp_path / "annotations" / "train.json"))
    collapsed = _collapse_coco_ground_truth(original, category_id=3)
    evaluator = COCOEvaluator(collapsed, label_to_category_id={0: 3})
    evaluator.update(
        {
            "boxes": [[4, 4, 16, 16], [32, 24, 44, 36]],
            "scores": [0.99, 0.98],
            "classes": [0, 0],
        },
        image_id=1,
    )

    metrics = evaluator.compute()

    assert metrics["mAP"] == pytest.approx(1.0)


def test_validator_auto_enables_single_cls_from_checkpoint_config():
    model = SimpleNamespace(
        nb_classes=1,
        _checkpoint_train_config=lambda: {"single_cls": True},
    )
    config = ValidationConfig(data="unused.yaml", device="cpu")

    validator = DetectionValidator(model, config)

    assert validator.config.single_cls is True
    assert config.single_cls is False
    assert validator._checkpoint_single_cls is True


def test_validator_warns_but_continues_on_generic_nc_mismatch(tmp_path, caplog):
    yaml_path = _write_yolo_data_yaml(tmp_path)
    model = SimpleNamespace(
        nb_classes=80,
        _checkpoint_train_config=lambda: {},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data=str(yaml_path),
        batch_size=1,
        num_workers=0,
        device="cpu",
    )
    validator = DetectionValidator(model, config)

    with caplog.at_level(logging.WARNING):
        dataloader = validator._setup_dataloader()

    assert validator.nc == 3
    assert len(dataloader.dataset) == 1
    assert "class count (80) differs from dataset class count (3)" in caplog.text


def test_checkpoint_single_cls_collapses_validator_dataset(tmp_path):
    yaml_path = _write_yolo_data_yaml(tmp_path)
    model = SimpleNamespace(
        nb_classes=1,
        _checkpoint_train_config=lambda: {"single_cls": True},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data=str(yaml_path),
        batch_size=1,
        num_workers=0,
        device="cpu",
    )
    validator = DetectionValidator(model, config)

    dataloader = validator._setup_dataloader()

    assert validator.nc == 1
    assert validator.class_names == ["object"]
    np.testing.assert_array_equal(
        dataloader.dataset.annotations[0][0][:, 4], np.zeros(2)
    )


def test_validator_warns_once_before_single_cls_clip(caplog):
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(single_cls=True)

    with caplog.at_level(logging.WARNING):
        validator._warn_single_cls_nonzero_gt(np.array([0, 2]))
        validator._warn_single_cls_nonzero_gt(np.array([3]))

    assert caplog.text.count("non-zero ground-truth class ids") == 1


def test_python_gate_accepts_every_g0_g1_detection_family():
    def train(self, data, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    supported = families_in("g0") + families_in("g1")
    assert len(supported) == 13

    for family in supported:
        wrapper = SimpleNamespace(FAMILY=family, task="detect")
        assert wrapped(wrapper, "data.yaml", single_cls=True)["single_cls"] is True


@pytest.mark.parametrize(
    ("family", "task"),
    [("yolox", "detect"), ("dfine", "segment")],
)
def test_python_gate_rejects_unsupported_family_or_task(family, task):
    def train(self, data, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    wrapper = SimpleNamespace(FAMILY=family, task=task)

    with pytest.raises(ValueError, match="G0/G1 detection"):
        wrapped(wrapper, "data.yaml", single_cls=True)


def test_resume_inherits_single_cls_from_checkpoint_probe():
    def train(self, data, *, resume=False, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    wrapper = SimpleNamespace(
        FAMILY="yolo9",
        task="detect",
        model_path="last.pt",
        _checkpoint_train_config=lambda source=None: {"single_cls": source is None},
    )

    result = wrapped(wrapper, "data.yaml", resume=True)

    assert result["single_cls"] is True


def test_checkpoint_config_probe_reads_saved_single_cls(tmp_path):
    class _Probe:
        _cache_checkpoint_train_config = BaseModel._cache_checkpoint_train_config

    checkpoint_path = tmp_path / "last.pt"
    torch.save({"model": {}, "config": {"single_cls": True}}, checkpoint_path)

    config = BaseModel._checkpoint_train_config(_Probe(), checkpoint_path)

    assert config["single_cls"] is True
