"""Training on a subset of a dataset's classes, without editing annotation
files (see repo-local research_class.md for the design writeup).

Generalizes the existing single_cls remap-at-parse-time mechanism: classes=
picks which original dataset class ids survive; every other class's boxes
are dropped as if never annotated, source annotation files stay untouched.
Kept ids are NOT compacted to a contiguous range -- they keep their original
numbering, so predictions and checkpoint metadata stay directly comparable
to the full dataset (an external COCO-style evaluator, an exported model
read by raw class index, a person who knows "id 7 = train") without any
translation layer. The cost is that the model head still covers every index
up to the highest kept id; unrequested classes below it simply never
receive positive supervision, the same as a class with zero occurrences in
an ordinary dataset. single_cls stays a genuine remap (collapses every kept
id to 0) since that is an intentional merge, not an exclusion.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from libreyolo.data.dataset import COCODataset, YOLODataset
from libreyolo.data.obb import parse_yolo_obb_label_line
from libreyolo.data.utils import build_class_remap, load_data_config
from libreyolo.data.yolo_coco_api import parse_yolo_label_line
from libreyolo.models.base.model import _wrap_train_with_cfg
from libreyolo.training.config import TrainConfig
from libreyolo.validation.config import ValidationConfig
from libreyolo.validation.detection_validator import DetectionValidator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TrainConfig validation
# ---------------------------------------------------------------------------


def test_train_config_classes_defaults_off():
    assert TrainConfig().classes is None


def test_train_config_classes_normalizes_to_int_list():
    assert TrainConfig(classes=[0, "1", 3]).classes == [0, 1, 3]


@pytest.mark.parametrize(
    "bad",
    [[], [-1, 0], [0, 0, 1]],
)
def test_train_config_classes_rejects_invalid(bad):
    with pytest.raises(ValueError):
        TrainConfig(classes=bad)


# ---------------------------------------------------------------------------
# build_class_remap / load_data_config
# ---------------------------------------------------------------------------


def test_build_class_remap_none_when_no_classes():
    assert build_class_remap(None) is None


def test_build_class_remap_keeps_original_ids():
    assert build_class_remap([3, 0, 1]) == {0: 0, 1: 1, 3: 3}


def test_build_class_remap_single_cls_collapses_to_zero():
    assert build_class_remap([3, 0, 1], single_cls=True) == {0: 0, 1: 0, 3: 0}


NAMES10 = (
    "car",
    "bicycle",
    "person",
    "dog",
    "truck",
    "cat",
    "bus",
    "train",
    "boat",
    "bird",
)


def _write_data_yaml(tmp_path: Path, nc=4, names=("car", "bicycle", "dog", "cat")):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/train",
                "nc": nc,
                "names": list(names),
            }
        )
    )
    return data_yaml


def test_load_data_config_classes_leaves_nc_and_names_untouched(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    cfg = load_data_config(str(data_yaml), autodownload=False, classes=[0, 1, 3])

    assert cfg["nc"] == 4
    assert cfg["names"] == ["car", "bicycle", "dog", "cat"]
    assert cfg["_class_remap"] == {0: 0, 1: 1, 3: 3}
    assert "_original_nc" not in cfg
    assert "_original_names" not in cfg


def test_load_data_config_classes_and_single_cls_collapse_kept_to_one(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    cfg = load_data_config(
        str(data_yaml), autodownload=False, classes=[0, 1, 3], single_cls=True
    )

    assert cfg["nc"] == 1
    assert cfg["names"] == {0: "object"}
    assert cfg["_class_remap"] == {0: 0, 1: 0, 3: 0}
    assert cfg["_original_nc"] == 4
    assert cfg["_original_names"] == ["car", "bicycle", "dog", "cat"]


def test_load_data_config_classes_out_of_range_fails_fast(tmp_path):
    data_yaml = _write_data_yaml(tmp_path)

    with pytest.raises(ValueError, match="outside this dataset's declared nc"):
        load_data_config(str(data_yaml), autodownload=False, classes=[0, 99])


# ---------------------------------------------------------------------------
# Label parsers
# ---------------------------------------------------------------------------


def test_parse_yolo_label_line_excluded_class_is_silently_dropped():
    remap = {0: 0, 1: 1, 3: 3}
    line = "2 0.5 0.5 0.2 0.2"  # dog, excluded

    assert parse_yolo_label_line(line, 100, 100, None, class_remap=remap) is None


def test_parse_yolo_label_line_kept_class_keeps_its_original_id():
    remap = {0: 0, 1: 1, 3: 3}
    line = "3 0.5 0.5 0.2 0.2"  # cat, id 3, kept as 3

    result = parse_yolo_label_line(line, 100, 100, None, class_remap=remap)

    assert result[0] == 3


def test_parse_yolo_obb_label_line_excluded_class_raises():
    remap = {0: 0, 1: 1, 3: 3}
    row = "2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"

    with pytest.raises(ValueError, match="excluded by classes"):
        parse_yolo_obb_label_line(row, class_remap=remap)


def test_parse_yolo_obb_label_line_kept_class_keeps_its_original_id():
    remap = {0: 0, 1: 1, 3: 3}
    row = "3 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"

    cls_id, _ = parse_yolo_obb_label_line(row, class_remap=remap)

    assert cls_id == 3


# ---------------------------------------------------------------------------
# YOLODataset (.txt)
# ---------------------------------------------------------------------------


def _write_yolo_sample(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    label_path = tmp_path / "sample.txt"
    Image.new("RGB", (64, 48), color="white").save(image_path)
    label_path.write_text(
        "0 0.1 0.1 0.1 0.1\n1 0.3 0.3 0.1 0.1\n2 0.5 0.5 0.1 0.1\n3 0.7 0.7 0.1 0.1\n",
        encoding="utf-8",
    )
    return [image_path], [label_path]


def test_yolo_dataset_class_remap_drops_excluded_keeps_original_ids(tmp_path):
    image_files, label_files = _write_yolo_sample(tmp_path)
    remap = {0: 0, 1: 1, 3: 3}  # drop original id 2 (dog); keep the rest as-is

    dataset = YOLODataset(
        img_files=image_files,
        label_files=label_files,
        img_size=(64, 64),
        class_remap=remap,
    )

    labels = dataset.annotations[0][0]
    assert labels.shape[0] == 3
    assert sorted(labels[:, 4].tolist()) == [0.0, 1.0, 3.0]


# ---------------------------------------------------------------------------
# COCODataset (JSON)
# ---------------------------------------------------------------------------


def _write_coco_sample(tmp_path: Path):
    image_dir = tmp_path / "images" / "train"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (64, 48), color="white").save(image_dir / "sample.jpg")
    (annotation_dir / "train.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 64, "height": 48}
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": 1,
                        "category_id": cat_id,
                        "bbox": [4, 4, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                    for i, cat_id in enumerate((1, 2, 3, 4), start=1)
                ],
                "categories": [
                    {"id": 1, "name": "car"},
                    {"id": 2, "name": "bicycle"},
                    {"id": 3, "name": "dog"},
                    {"id": 4, "name": "cat"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_coco_dataset_classes_drops_excluded_keeps_original_labels(tmp_path):
    pytest.importorskip("pycocotools")
    _write_coco_sample(tmp_path)

    dataset = COCODataset(
        data_dir=str(tmp_path),
        json_file="annotations/train.json",
        name="images/train",
        img_size=(64, 64),
        num_classes=4,
        names={0: "car", 1: "bicycle", 2: "dog", 3: "cat"},
        classes=[0, 1, 3],
    )

    # category 3 (dog, label 2) is excluded; the rest keep their original label.
    assert dataset.category_id_to_label == {1: 0, 2: 1, 4: 3}
    # names cover the full original set, including the excluded class --
    # nothing is hidden, it just never receives positive supervision.
    assert dataset._classes == ("car", "bicycle", "dog", "cat")
    labels = dataset.annotations[0][0]
    assert labels.shape[0] == 3
    assert sorted(labels[:, 4].tolist()) == [0.0, 1.0, 3.0]


# ---------------------------------------------------------------------------
# End-to-end trainer
# ---------------------------------------------------------------------------


def _write_train_dataset(tmp_path, nc=4, names=("car", "bicycle", "dog", "cat")):
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color="white").save(img_dir / "sample.jpg")
    (lbl_dir / "sample.txt").write_text(
        "0 0.1 0.1 0.1 0.1\n1 0.3 0.3 0.1 0.1\n2 0.5 0.5 0.1 0.1\n3 0.7 0.7 0.1 0.1\n"
    )
    return _write_data_yaml(tmp_path, nc=nc, names=names)


@pytest.mark.parametrize(
    ("trainer_import", "size"),
    [
        # Uses the shared BaseTrainer._setup_data.
        ("libreyolo.models.rtdetr.trainer:RTDETRTrainer", "r18"),
        # DFINE/DEIM mirror BaseTrainer._setup_data by hand -- independent
        # copy of the same class-filter wiring.
        ("libreyolo.models.dfine.trainer:DFINETrainer", "n"),
        ("libreyolo.models.deim.trainer:DEIMTrainer", "n"),
    ],
)
def test_trainer_setup_data_trains_only_the_requested_class_subset(
    tmp_path, trainer_import, size
):
    module_name, class_name = trainer_import.split(":")
    module = __import__(module_name, fromlist=[class_name])
    trainer_cls = getattr(module, class_name)

    data_yaml = _write_train_dataset(tmp_path)
    trainer = trainer_cls(
        model=torch.nn.Identity(),
        size=size,
        num_classes=4,
        data=str(data_yaml),
        classes=[0, 1, 3],
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    trainer._setup_data()

    # nc stays the full declared count -- classes= only filters the loss.
    assert trainer.num_classes == 4
    raw_dataset = trainer.train_loader.dataset.dataset
    labels = raw_dataset.annotations[0][0]
    assert labels.shape[0] == 3
    assert sorted(labels[:, 4].tolist()) == [0.0, 1.0, 3.0]


def test_trainer_run_validation_threads_classes_into_epoch_validation(
    tmp_path, monkeypatch
):
    """BaseTrainer._run_validation builds its own ValidationConfig for the
    per-epoch mAP printed during training. Unlike a standalone model.val()
    call, no checkpoint has been saved yet for it to auto-inherit classes=
    from, so it must be threaded through explicitly -- otherwise per-epoch
    validation silently scores against the full, unfiltered dataset while
    the model was only ever supervised on the requested subset, tanking the
    reported mAP for reasons that have nothing to do with prediction quality.
    """
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    data_yaml = _write_train_dataset(tmp_path)
    trainer = RTDETRTrainer(
        model=torch.nn.Identity(),
        wrapper_model=SimpleNamespace(task="detect", model=torch.nn.Identity()),
        size="r18",
        num_classes=4,
        data=str(data_yaml),
        classes=[0, 1, 3],
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    captured = {}

    class _StopEarly(Exception):
        pass

    class _SpyValidationConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _StopEarly

    monkeypatch.setattr(
        "libreyolo.validation.ValidationConfig", _SpyValidationConfig
    )

    result = trainer._run_validation(0)

    assert result is None  # the spy's exception is caught and logged
    assert captured.get("classes") == [0, 1, 3]


# ---------------------------------------------------------------------------
# Validator auto-inherit
# ---------------------------------------------------------------------------


def test_validator_auto_inherits_classes_from_checkpoint(tmp_path):
    data_yaml = _write_train_dataset(tmp_path)
    model = SimpleNamespace(
        nb_classes=4,
        _checkpoint_train_config=lambda: {"classes": [0, 1, 3]},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data=str(data_yaml), batch_size=1, num_workers=0, device="cpu"
    )

    validator = DetectionValidator(model, config)

    assert validator.config.classes == [0, 1, 3]
    dataloader = validator._setup_dataloader()
    assert validator.nc == 4
    assert validator.class_names == ["car", "bicycle", "dog", "cat"]
    labels = dataloader.dataset.annotations[0][0]
    assert labels.shape[0] == 3


def test_validator_explicit_classes_not_overridden_by_checkpoint(tmp_path):
    data_yaml = _write_train_dataset(tmp_path)
    model = SimpleNamespace(
        nb_classes=4,
        _checkpoint_train_config=lambda: {"classes": [0, 1, 3]},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        num_workers=0,
        device="cpu",
        classes=[0, 2],
    )

    validator = DetectionValidator(model, config)

    assert validator.config.classes == [0, 2]


def _minimal_val_model(nb_classes=4, checkpoint_config=None):
    return SimpleNamespace(
        nb_classes=nb_classes,
        _checkpoint_train_config=lambda: checkpoint_config or {},
        _get_val_preprocessor=lambda img_size: None,
    )


def test_validator_init_metrics_filters_yolo_ground_truth_by_classes(tmp_path):
    """COCO ground truth built from YOLO-format labels (YOLOCocoAPI) must
    drop the excluded class, or mAP is silently scored against the full
    dataset instead of the requested subset (#Greptile finding on
    detection_validator.py:312).
    """
    data_yaml = _write_train_dataset(tmp_path)  # one image, labels 0/1/2/3
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        num_workers=0,
        device="cpu",
        classes=[0, 1, 3],  # excludes label 2 ("dog")
    )
    validator = DetectionValidator(_minimal_val_model(), config)
    validator._setup_dataloader()
    validator._init_metrics()

    coco_gt = validator.coco_evaluator.coco_gt
    assert set(coco_gt.getCatIds()) == {0, 1, 3}
    assert {a["category_id"] for a in coco_gt.anns.values()} == {0, 1, 3}
    assert len(coco_gt.anns) == 3


def test_validator_init_metrics_filters_coco_json_ground_truth_by_classes(tmp_path):
    """Same guarantee for native COCO-JSON ground truth: dropping the
    excluded class's category and annotations, not just filtering the
    dataloader used to train/score predictions.
    """
    pytest.importorskip("pycocotools")
    image_dir = tmp_path / "images" / "val"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (64, 48), color="white").save(image_dir / "sample.jpg")
    (annotation_dir / "instances_val.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "sample.jpg", "width": 64, "height": 48}
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": 1,
                        "category_id": cat_id,
                        "bbox": [4, 4, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                    for i, cat_id in enumerate((1, 2, 3, 4), start=1)
                ],
                "categories": [
                    {"id": 1, "name": "car"},
                    {"id": 2, "name": "bicycle"},
                    {"id": 3, "name": "dog"},
                    {"id": 4, "name": "cat"},
                ],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = _write_data_yaml(tmp_path)  # nc=4, names=car/bicycle/dog/cat

    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        num_workers=0,
        device="cpu",
        classes=[0, 1, 3],  # excludes label 2 ("dog" -> category_id 3)
    )
    validator = DetectionValidator(_minimal_val_model(), config)
    validator._setup_dataloader()
    validator._init_metrics()

    coco_gt = validator.coco_evaluator.coco_gt
    assert set(coco_gt.getCatIds()) == {1, 2, 4}
    assert len(coco_gt.getAnnIds()) == 3


# ---------------------------------------------------------------------------
# _wrap_train_with_cfg gate
# ---------------------------------------------------------------------------


def test_python_gate_accepts_classes_for_g0_g1_detection():
    def train(self, data, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    wrapper = SimpleNamespace(FAMILY="rtdetr", task="detect")

    result = wrapped(wrapper, "data.yaml", classes=[0, 1, 3])

    assert result["classes"] == [0, 1, 3]


def test_python_gate_rejects_classes_for_unsupported_family_or_task():
    def train(self, data, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    wrapper = SimpleNamespace(FAMILY="yolox", task="detect")

    with pytest.raises(ValueError, match="G0/G1 detection"):
        wrapped(wrapper, "data.yaml", classes=[0, 1])


def test_resume_inherits_classes_from_checkpoint():
    def train(self, data, *, resume=False, **kwargs):
        return kwargs

    wrapped = _wrap_train_with_cfg(train)
    wrapper = SimpleNamespace(
        FAMILY="yolo9",
        task="detect",
        model_path="last.pt",
        _checkpoint_train_config=lambda source=None: {"classes": [0, 1, 3]},
    )

    result = wrapped(wrapper, "data.yaml", resume=True)

    assert result["classes"] == [0, 1, 3]


# ---------------------------------------------------------------------------
# on_num_classes_resolved: the wrapper's real names must survive the head
# rebuild, not just its class count. _rebuild_for_new_classes() always resets
# to generic class_N placeholders; only _sync_wrapped_model_num_classes
# restores real names afterward, via _resolved_class_names stashed by
# _resolve_num_classes_from_data_config. This holds regardless of classes=,
# which never changes nc/names here -- see build_class_remap's docstring.
# ---------------------------------------------------------------------------


class _FakeDetector(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
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


def _build_rtdetr_trainer(data_yaml, wrapper, **overrides):
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    kwargs = dict(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="r18",
        num_classes=wrapper.nb_classes,
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
    kwargs.update(overrides)
    return RTDETRTrainer(**kwargs)


def test_sync_ignores_classes_for_nc_and_names_resolution(tmp_path):
    """classes= is a loss-time filter only: nc/names resolve exactly as they
    would without it (the full 10-class dataset), even though only 7 ids are
    requested for training."""
    data_yaml = _write_data_yaml(tmp_path, nc=10, names=NAMES10)
    wrapper = _FakeWrapper(num_classes=80)
    trainer = _build_rtdetr_trainer(data_yaml, wrapper, classes=[0, 3, 5, 6, 7, 8, 9])

    trainer.on_num_classes_resolved()

    assert wrapper.nb_classes == 10
    assert wrapper.names == {i: name for i, name in enumerate(NAMES10)}


def test_sync_restores_real_names_for_plain_full_dataset_training(tmp_path):
    """Not classes=-specific: any dataset-driven head resize (nc mismatch
    against the loaded checkpoint) must end up with the dataset's real names,
    not generic placeholders -- this held for single_cls already, now for
    every case _resolved_class_names can supply."""
    data_yaml = _write_data_yaml(tmp_path, nc=3, names=("car", "bicycle", "dog"))
    wrapper = _FakeWrapper(num_classes=80)
    trainer = _build_rtdetr_trainer(data_yaml, wrapper)

    trainer.on_num_classes_resolved()

    assert wrapper.nb_classes == 3
    assert wrapper.names == {0: "car", 1: "bicycle", 2: "dog"}


def test_sync_no_rebuild_still_applies_names(tmp_path):
    """When the wrapper already has the right class count (no rebuild
    needed), the not-needs-rebuild branch must still stamp the correct
    names -- it used to only special-case single_cls there."""
    data_yaml = _write_data_yaml(tmp_path, nc=10, names=NAMES10)
    wrapper = _FakeWrapper(num_classes=10)  # already the right count
    trainer = _build_rtdetr_trainer(data_yaml, wrapper, classes=[0, 3, 5, 6, 7, 8, 9])

    trainer.on_num_classes_resolved()

    assert wrapper.names == {i: name for i, name in enumerate(NAMES10)}


# ---------------------------------------------------------------------------
# End-to-end through LibreYOLO9.train() -- the real user-facing entry point.
#
# Confirms the three places that used to independently resolve nc/names
# (Trainer._resolve_num_classes_from_data_config, BaseTrainer._setup_data,
# and YOLO9's own model.py::train()) agree: none of them need to know about
# classes= at all anymore, since it never changes nc/names. That is what
# makes this design change smaller and safer than the compacting-remap one
# it replaces -- see build_class_remap's docstring.
# ---------------------------------------------------------------------------


def test_libreyolo9_train_end_to_end_keeps_original_nc_and_names(tmp_path):
    from libreyolo import LibreYOLO9

    data_yaml = _write_train_dataset(tmp_path, nc=10, names=NAMES10)
    model = LibreYOLO9(None, size="t", device="cpu")

    model.train(
        data=str(data_yaml),
        classes=[0, 3, 5, 6, 7, 8, 9],
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        workers=0,
        amp=False,
        ema=False,
        project=str(tmp_path / "runs"),
        name="exp",
    )

    expected_names = {i: name for i, name in enumerate(NAMES10)}
    assert model.nb_classes == 10
    assert model.names == expected_names

    checkpoint_path = tmp_path / "runs" / "exp" / "weights" / "last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["nc"] == 10
    assert checkpoint["names"] == expected_names


# ---------------------------------------------------------------------------
# nc must be derivable from len(names) when the yaml omits an explicit nc:
# key -- BaseTrainer._setup_data() already did this; _resolve_num_classes_
# from_data_config() and DFINE/DEIM's by-hand _setup_data() copies did not,
# so a trainer built with a stale initial num_classes (e.g. from a
# previously loaded checkpoint) never got corrected and produced exactly
# the out-of-range warning this test guards against.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trainer_import", "size"),
    [
        ("libreyolo.models.rtdetr.trainer:RTDETRTrainer", "r18"),
        ("libreyolo.models.dfine.trainer:DFINETrainer", "n"),
        ("libreyolo.models.deim.trainer:DEIMTrainer", "n"),
    ],
)
def test_nc_derives_from_names_when_yaml_omits_nc_even_with_stale_initial_value(
    tmp_path, trainer_import, size
):
    module_name, class_name = trainer_import.split(":")
    module = __import__(module_name, fromlist=[class_name])
    trainer_cls = getattr(module, class_name)

    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color="white").save(img_dir / "sample.jpg")
    (lbl_dir / "sample.txt").write_text("9 0.1 0.1 0.05 0.05\n0 0.5 0.5 0.05 0.05\n")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "names": [f"c{i}" for i in range(11)],
                # deliberately no "nc" key
            }
        )
    )

    trainer = trainer_cls(
        model=torch.nn.Identity(),
        size=size,
        num_classes=7,  # stale, e.g. from a previously loaded checkpoint
        data=str(data_yaml),
        classes=[0, 1, 2, 3, 4, 5, 9],
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    trainer.on_num_classes_resolved()
    assert trainer.num_classes == 11

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer._setup_data()
    assert not any(issubclass(w.category, UserWarning) for w in caught)

    assert trainer.num_classes == 11
    labels = trainer.train_loader.dataset.dataset.annotations[0][0]
    assert sorted(labels[:, 4].tolist()) == [0.0, 9.0]


# ---------------------------------------------------------------------------
# log_classes_subset_notice: classes= must be impossible to miss in the
# logs. It never changes nc/names (see build_class_remap's docstring), so
# there is nothing else in a run's resolved config that flags it as
# non-standard -- without this, a user sees a normally-sized, normally-named
# model and has no clue some classes were silently excluded from training.
# ---------------------------------------------------------------------------


def test_log_classes_subset_notice_fires_for_plain_classes(caplog):
    from libreyolo.training.config import TrainConfig
    from libreyolo.training.trainer import log_classes_subset_notice

    config = TrainConfig(classes=[0, 1, 3])

    with caplog.at_level("WARNING"):
        log_classes_subset_notice(config, num_classes=4)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "SUBSET" in message
    assert "[0, 1, 3]" in message
    assert "3 of 4" in message
    assert "Training" in message


def test_log_classes_subset_notice_mentions_single_cls_collapse(caplog):
    from libreyolo.training.config import TrainConfig
    from libreyolo.training.trainer import log_classes_subset_notice

    config = TrainConfig(classes=[0, 1, 3], single_cls=True)

    with caplog.at_level("WARNING"):
        log_classes_subset_notice(config, num_classes=4)

    message = caplog.records[0].getMessage()
    assert "single_cls" in message
    assert "collapsed" in message


def test_log_classes_subset_notice_respects_context(caplog):
    from libreyolo.training.config import TrainConfig
    from libreyolo.training.trainer import log_classes_subset_notice

    config = TrainConfig(classes=[0, 1, 3])

    with caplog.at_level("WARNING"):
        log_classes_subset_notice(config, num_classes=4, context="Validating")

    assert caplog.records[0].getMessage().startswith("Validating on a SUBSET")


def test_log_classes_subset_notice_silent_when_classes_not_set(caplog):
    from libreyolo.training.config import TrainConfig
    from libreyolo.training.trainer import log_classes_subset_notice

    config = TrainConfig()

    with caplog.at_level("WARNING"):
        log_classes_subset_notice(config, num_classes=80)

    assert caplog.records == []


def test_trainer_setup_data_logs_classes_subset_notice(tmp_path, caplog):
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer

    data_yaml = _write_train_dataset(tmp_path)
    trainer = RTDETRTrainer(
        model=torch.nn.Identity(),
        size="r18",
        num_classes=4,
        data=str(data_yaml),
        classes=[0, 1, 3],
        epochs=1,
        batch=1,
        imgsz=64,
        device="cpu",
        amp=False,
        ema=False,
        workers=0,
        eval_interval=-1,
    )

    with caplog.at_level("WARNING"):
        trainer._setup_data()

    assert any("SUBSET" in r.getMessage() for r in caplog.records)


def test_validator_setup_dataloader_logs_classes_subset_notice(tmp_path, caplog):
    data_yaml = _write_train_dataset(tmp_path)
    model = SimpleNamespace(
        nb_classes=4,
        _checkpoint_train_config=lambda: {"classes": [0, 1, 3]},
        _get_val_preprocessor=lambda img_size: None,
    )
    config = ValidationConfig(
        data=str(data_yaml), batch_size=1, num_workers=0, device="cpu"
    )
    validator = DetectionValidator(model, config)

    with caplog.at_level("WARNING"):
        validator._setup_dataloader()

    messages = [r.getMessage() for r in caplog.records]
    assert any("Validating on a SUBSET" in m for m in messages)
