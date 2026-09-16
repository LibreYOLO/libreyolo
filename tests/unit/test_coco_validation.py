"""
Unit tests for COCO evaluation in DetectionValidator.

Tests that COCO evaluator is properly integrated into the validation pipeline
using mock data (no GPU or real datasets needed).
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


def _coco_metrics(base: float):
    return {
        "precision": base + 0.001,
        "recall": base + 0.002,
        "mAP": base,
        "mAP50": base + 0.01,
        "mAP75": base + 0.02,
        "mAP_small": base + 0.03,
        "mAP_medium": base + 0.04,
        "mAP_large": base + 0.05,
        "AR1": base + 0.06,
        "AR10": base + 0.07,
        "AR100": base + 0.08,
        "AR_max_det": base + 0.08,
        "max_det": 100.0,
        "AR_small": base + 0.09,
        "AR_medium": base + 0.10,
        "AR_large": base + 0.11,
    }


class _DummyEvaluator:
    def __init__(self, metrics=None):
        self.metrics = metrics or _coco_metrics(0.1)
        self.updates = []
        self.saved_paths = []

    def update(self, pred, image_id):
        self.updates.append((pred, image_id))

    def compute(self, save_json=None):
        self.saved_paths.append(save_json)
        return dict(self.metrics)


def _null_val_preprocessor(*, img_size: int) -> None:
    assert img_size > 0


def create_mock_yolo_dataset(tmp_path):
    """Create a minimal mock YOLO dataset for testing."""
    # Create directory structure
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create 3 dummy images
    for i in range(3):
        img = Image.new("RGB", (640, 640), color=(i * 50, i * 50, i * 50))
        img.save(images_dir / f"img{i}.jpg")

        # Create corresponding label
        # Format: class cx cy w h (normalized)
        labels = [
            "0 0.5 0.5 0.3 0.3\n",  # Center object
            "1 0.25 0.25 0.2 0.2\n",  # Top-left object
        ]
        (labels_dir / f"img{i}.txt").write_text("".join(labels[: i + 1]))

    # Create data.yaml
    data_yaml = {
        "path": str(tmp_path),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["cat", "dog"],
    }

    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    return yaml_path


def _dense_coco_fixture(count: int):
    """Build one COCO image with non-overlapping boxes and perfect predictions."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    annotations = []
    boxes = []
    scores = []
    classes = []
    columns = 25
    for index in range(count):
        x = float((index % columns) * 4)
        y = float((index // columns) * 4)
        box = [x, y, 2.0, 2.0]
        annotations.append(
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": box,
                "area": 4.0,
                "iscrowd": 0,
            }
        )
        boxes.append([x, y, x + 2.0, y + 2.0])
        scores.append(1.0 - index / (count * 2.0))
        classes.append(0)

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "dense.jpg",
                "width": 100,
                "height": max(100, ((count + columns - 1) // columns) * 4),
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    predictions = {"boxes": boxes, "scores": scores, "classes": classes}
    return coco, predictions


@pytest.mark.unit
def test_detection_validator_honors_explicit_imgsz_override():
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(imgsz=1920)
    validator.model = SimpleNamespace(_get_input_size=lambda: 640)

    assert validator._resolve_imgsz() == 1920


@pytest.mark.unit
def test_detection_validator_uses_model_imgsz_when_unspecified():
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(imgsz=None)
    validator.model = SimpleNamespace(_get_input_size=lambda: 416)

    assert validator._resolve_imgsz() == 416


@pytest.mark.unit
def test_coco_evaluator_integration(tmp_path):
    """Test that COCOEvaluator integrates into DetectionValidator."""
    from libreyolo.data import create_yolo_coco_api
    from libreyolo.validation import COCOEvaluator

    yaml_path = create_mock_yolo_dataset(tmp_path)

    # YOLOCocoAPI creation
    coco_api = create_yolo_coco_api(str(yaml_path), split="val")
    assert len(coco_api.imgs) == 3

    # COCOEvaluator initialization
    evaluator = COCOEvaluator(coco_api, iou_type="bbox")

    # Update with dummy predictions
    dummy_pred = {
        "boxes": [[100, 100, 200, 200]],
        "scores": [0.9],
        "classes": [0],
    }
    evaluator.update(dummy_pred, image_id=0)
    assert len(evaluator.results) == 1

    # Compute metrics
    metrics = evaluator.compute()
    expected_keys = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


@pytest.mark.unit
def test_coco_evaluator_headline_ap_uses_configured_500_cap():
    """Overall AP must not use stock summarize's hard-coded maxDets=100 lookup."""
    from libreyolo.validation import COCOEvaluator

    coco, predictions = _dense_coco_fixture(1)
    evaluator = COCOEvaluator(
        coco, iou_type="bbox", label_to_category_id={0: 1}, max_det=500
    )
    evaluator.update(predictions, image_id=1)

    metrics = evaluator.compute()

    assert evaluator._last_coco_eval.params.maxDets == [1, 10, 100, 500]
    assert metrics["max_det"] == 500
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["mAP50"] == pytest.approx(1.0)


@pytest.mark.unit
def test_coco_evaluator_dense_image_proves_ap500_exceeds_ap100():
    """A dense perfect prediction set must expose the configured cap in headline AP."""
    from libreyolo.validation import COCOEvaluator

    coco, predictions = _dense_coco_fixture(500)
    evaluator_100 = COCOEvaluator(
        coco, iou_type="bbox", label_to_category_id={0: 1}, max_det=100
    )
    evaluator_500 = COCOEvaluator(
        coco, iou_type="bbox", label_to_category_id={0: 1}, max_det=500
    )
    evaluator_100.update(predictions, image_id=1)
    evaluator_500.update(predictions, image_id=1)

    metrics_100 = evaluator_100.compute()
    metrics_500 = evaluator_500.compute()

    assert metrics_500["mAP"] == pytest.approx(1.0)
    assert metrics_500["mAP"] > metrics_100["mAP"]
    assert metrics_500["AR_max_det"] > metrics_100["AR_max_det"]


@pytest.mark.unit
def test_coco_evaluator_default_is_bit_exact_stock_pycocotools():
    """The opt-in evaluator cap must not rebaseline default AP@100 users."""
    from pycocotools.cocoeval import COCOeval

    from libreyolo.validation import COCOEvaluator

    coco, predictions = _dense_coco_fixture(125)
    evaluator = COCOEvaluator(
        coco,
        iou_type="bbox",
        label_to_category_id={0: 1},
    )
    evaluator.update(predictions, image_id=1)

    reference_dt = coco.loadRes(evaluator.results)
    reference = COCOeval(coco, reference_dt, "bbox")
    reference.params.imgIds = [1]
    reference.evaluate()
    reference.accumulate()
    reference.summarize()

    metrics = evaluator.compute()
    actual = evaluator._last_coco_eval

    assert actual.params.maxDets == reference.params.maxDets == [1, 10, 100]
    np.testing.assert_array_equal(actual.stats, reference.stats)
    assert metrics["mAP"] == float(reference.stats[0])
    assert metrics["mAP50"] == float(reference.stats[1])
    assert metrics["mAP75"] == float(reference.stats[2])
    assert metrics["AR100"] == float(reference.stats[8])
    reference_map_values = reference.eval["precision"][:, :, :, 0, -1]
    reference_ar_values = reference.eval["recall"][:, :, 0, -1]
    assert metrics["map_5095"] == float(
        reference_map_values[reference_map_values > -1].mean()
    )
    assert metrics["ar_100"] == float(
        reference_ar_values[reference_ar_values > -1].mean()
    )


@pytest.mark.unit
def test_coco_evaluator_preserves_zero_legacy_aliases_without_valid_gt():
    """Undefined legacy aliases remain 0.0, matching pre-change behavior."""
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    from libreyolo.validation import COCOEvaluator

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "empty.jpg", "width": 32, "height": 32}
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    evaluator = COCOEvaluator(coco, label_to_category_id={0: 1})
    evaluator.update(
        {
            "boxes": [[1.0, 1.0, 5.0, 5.0]],
            "scores": [0.9],
            "classes": [0],
        },
        image_id=1,
    )

    metrics = evaluator.compute()

    assert metrics["map_5095"] == 0.0
    assert metrics["ar_100"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["mAP"] == -1.0


def _imperfect_coco_fixture():
    """Dense fixture with the top-scored box misplaced so mAP is non-trivial."""
    coco, predictions = _dense_coco_fixture(8)
    predictions["boxes"][0] = [50.0, 50.0, 52.0, 52.0]
    return coco, predictions


@pytest.mark.unit
def test_detection_validator_save_json_disabled_writes_no_file(tmp_path):
    from libreyolo.validation import COCOEvaluator
    from libreyolo.validation.detection_validator import DetectionValidator

    coco, predictions = _dense_coco_fixture(4)
    evaluator = COCOEvaluator(coco, iou_type="bbox", label_to_category_id={0: 1})
    evaluator.update(predictions, image_id=1)

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = tmp_path
    validator.coco_evaluator = evaluator

    metrics = validator._compute_metrics()

    assert metrics["metrics/mAP50-95"] == pytest.approx(1.0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_detection_validator_save_json_round_trips_reported_map(tmp_path):
    from pycocotools.cocoeval import COCOeval

    from libreyolo.validation import COCOEvaluator
    from libreyolo.validation.detection_validator import DetectionValidator

    coco, predictions = _imperfect_coco_fixture()
    evaluator = COCOEvaluator(coco, iou_type="bbox", label_to_category_id={0: 1})
    evaluator.update(predictions, image_id=1)

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=True)
    validator.save_dir = tmp_path
    validator.coco_evaluator = evaluator

    # Snapshot the entries fed to the evaluator; pycocotools loadRes mutates
    # the originals in place during compute.
    expected = [dict(entry) for entry in evaluator.results]
    metrics = validator._compute_metrics()

    json_path = tmp_path / "predictions.json"
    assert json_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved == expected
    assert all(
        set(entry) == {"image_id", "category_id", "bbox", "score"} for entry in saved
    )
    assert {entry["category_id"] for entry in saved} == {1}

    reloaded = coco.loadRes(str(json_path))
    reference = COCOeval(coco, reloaded, "bbox")
    reference.params.imgIds = [1]
    reference.evaluate()
    reference.accumulate()
    reference.summarize()

    assert 0.0 < metrics["metrics/mAP50-95"] < 1.0
    assert metrics["metrics/mAP50-95"] == float(reference.stats[0])
    assert metrics["metrics/mAP50"] == float(reference.stats[1])


@pytest.mark.unit
def test_coco_evaluator_save_json_writes_empty_file_without_predictions(tmp_path):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO

    from libreyolo.validation import COCOEvaluator

    coco = COCO()
    coco.dataset = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "empty.jpg", "width": 32, "height": 32}
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    evaluator = COCOEvaluator(coco, label_to_category_id={0: 1})

    json_path = tmp_path / "predictions.json"
    metrics = evaluator.compute(save_json=str(json_path))

    assert metrics["mAP"] == 0.0
    assert json.loads(json_path.read_text(encoding="utf-8")) == []


@pytest.mark.unit
def test_save_json_stays_in_legacy_positional_config_layout():
    from inspect import Parameter, signature

    from libreyolo.validation.config import ValidationConfig

    parameters = signature(ValidationConfig).parameters

    # save_json predates the kw_only convention; converting it now would
    # shift every later positional field, so its slot stays pinned.
    assert parameters["save_json"].default is False
    assert parameters["save_json"].kind is Parameter.POSITIONAL_OR_KEYWORD


@pytest.mark.unit
def test_detection_validator_uses_explicit_coco_json_paths(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset
    from libreyolo.validation.detection_validator import DetectionValidator

    image_dir = tmp_path / "images" / "custom_val"
    ann_dir = tmp_path / "custom_annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "valid.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "sample.jpg", "width": 64, "height": 64}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 42,
                        "bbox": [8, 8, 16, 16],
                        "area": 256,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 42, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: images/custom_val\n"
        "val: images/custom_val\n"
        "annotations:\n"
        "  val: custom_annotations/valid.json\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(
        data=str(yaml_path),
        data_dir=None,
        split="val",
        allow_download_scripts=False,
        imgsz=64,
        batch_size=1,
        num_workers=0,
        verbose=False,
        save_plots=False,
        max_det=300,
        eval_max_det=None,
        classes=None,
    )
    validator.model = SimpleNamespace(
        _get_input_size=lambda: 64,
        _get_val_preprocessor=_null_val_preprocessor,
    )
    validator.device = torch.device("cpu")
    validator.nc = 1
    validator.class_names = None

    loader = validator._setup_dataloader()

    assert isinstance(loader.dataset, COCODataset)
    assert loader.dataset.json_file == str(ann_dir / "valid.json")
    assert loader.dataset.name == str(image_dir)
    assert loader.dataset._image_path(0) == image_dir / "sample.jpg"

    validator._init_metrics()
    assert validator.coco_evaluator.max_det == 100

    validator.config.eval_max_det = 500
    validator._init_metrics()
    assert validator.coco_evaluator.max_det == 500


@pytest.mark.unit
def test_obb_validator_uses_explicit_coco_json_paths(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset
    from libreyolo.validation.obb_validator import OBBValidator

    image_dir = tmp_path / "images" / "custom_val"
    ann_dir = tmp_path / "custom_annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    cx, cy = 50.0, 50.0
    box_w, box_h = 40.0, 20.0
    angle = math.radians(30.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for dx, dy in (
        (-box_w / 2, -box_h / 2),
        (box_w / 2, -box_h / 2),
        (box_w / 2, box_h / 2),
        (-box_w / 2, box_h / 2),
    ):
        corners.extend([cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a])
    (ann_dir / "valid.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "sample.jpg", "width": 100, "height": 100}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 42,
                        "bbox": [10, 20, 40, 20],
                        "obb": corners,
                        "area": 800,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 42, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: images/custom_val\n"
        "val: images/custom_val\n"
        "annotations:\n"
        "  val: custom_annotations/valid.json\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    validator = OBBValidator.__new__(OBBValidator)
    validator.config = SimpleNamespace(
        data=str(yaml_path),
        data_dir=None,
        split="val",
        allow_download_scripts=False,
        imgsz=100,
        batch_size=1,
        num_workers=0,
        save_json=False,
        save_plots=False,
    )
    validator.model = SimpleNamespace(
        _get_input_size=lambda: 100,
        _get_val_preprocessor=_null_val_preprocessor,
    )
    validator.device = torch.device("cpu")
    validator.nc = 1
    validator.class_names = None

    loader = validator._setup_dataloader()

    assert isinstance(loader.dataset, COCODataset)
    assert loader.dataset.load_obb is True
    assert loader.dataset.json_file == str(ann_dir / "valid.json")
    assert loader.dataset.name == str(image_dir)
    assert set(validator._gt_by_image) == {10}
    cls_id, xywhr = validator._gt_by_image[10][0]
    assert cls_id == 0
    np.testing.assert_allclose(xywhr, [50.0, 50.0, 40.0, 20.0, angle], atol=1e-5)

    validator.iou_thresholds = (0.50, 0.75)
    validator._init_metrics()
    validator._update_metrics(
        [
            {
                "obb": torch.tensor(
                    [[50.0, 50.0, 40.0, 20.0, angle, 0.9, 0.0]],
                    dtype=torch.float32,
                )
            }
        ],
        torch.zeros(1, 1, 6),
        [(100, 100)],
        [torch.tensor(10)],
    )
    metrics = validator._compute_metrics()

    assert metrics["metrics/mAP50(OBB)"] == pytest.approx(1.0)
    assert metrics["metrics/mAP50-95(OBB)"] == pytest.approx(1.0)


@pytest.mark.unit
def test_obb_validator_uses_default_coco_images_layout_with_data_dir(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset
    from libreyolo.validation.obb_validator import OBBValidator

    image_dir = tmp_path / "images" / "val2017"
    ann_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "sample.jpg", "width": 100, "height": 100}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 1,
                        "bbox": [10, 20, 40, 20],
                        "obb": [10, 20, 50, 20, 50, 40, 10, 40],
                        "area": 800,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    validator = OBBValidator.__new__(OBBValidator)
    validator.config = SimpleNamespace(
        data=None,
        data_dir=str(tmp_path),
        split="val",
        allow_download_scripts=False,
        imgsz=100,
        batch_size=1,
        num_workers=0,
        save_json=False,
        save_plots=False,
    )
    validator.model = SimpleNamespace(
        _get_input_size=lambda: 100,
        _get_val_preprocessor=_null_val_preprocessor,
    )
    validator.device = torch.device("cpu")
    validator.nc = 1
    validator.class_names = None

    loader = validator._setup_dataloader()

    assert isinstance(loader.dataset, COCODataset)
    assert loader.dataset.load_obb is True
    assert loader.dataset.json_file == "instances_val2017.json"
    assert loader.dataset.name == "images/val2017"
    assert loader.dataset._image_path(0) == image_dir / "sample.jpg"
    assert set(validator._gt_by_image) == {10}


@pytest.mark.unit
def test_detection_validator_accepts_txt_split_with_discovered_coco_json(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset
    from libreyolo.validation.detection_validator import DetectionValidator

    image_dir = tmp_path / "val2017"
    ann_dir = tmp_path / "annotations"
    image_dir.mkdir()
    ann_dir.mkdir()
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "instances_val2017.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "sample.jpg", "width": 64, "height": 64}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 42,
                        "bbox": [8, 8, 16, 16],
                        "area": 256,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 42, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "val2017.txt").write_text("val2017/sample.jpg\n", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: val2017.txt\n"
        "val: val2017.txt\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(
        data=str(yaml_path),
        data_dir=None,
        split="val",
        allow_download_scripts=False,
        imgsz=64,
        batch_size=1,
        num_workers=0,
        classes=None,
    )
    validator.model = SimpleNamespace(
        _get_input_size=lambda: 64,
        _get_val_preprocessor=_null_val_preprocessor,
    )
    validator.device = torch.device("cpu")
    validator.nc = 1
    validator.class_names = None

    loader = validator._setup_dataloader()

    assert isinstance(loader.dataset, COCODataset)
    assert loader.dataset.json_file == "instances_val2017.json"
    assert loader.dataset.name == "val2017"
    assert loader.dataset._image_path(0) == image_dir / "sample.jpg"


@pytest.mark.unit
def test_coco_evaluator_encodes_masks_json_safe():
    from libreyolo.validation.coco_evaluator import COCOEvaluator

    mask = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
    rle = COCOEvaluator._encode_mask(mask)

    assert rle["size"] == [3, 3]
    assert isinstance(rle["counts"], str)


@pytest.mark.unit
def test_coco_evaluator_uses_mask_area_for_segmentation(tmp_path):
    from libreyolo.data import create_yolo_coco_api
    from libreyolo.validation import COCOEvaluator

    yaml_path = create_mock_yolo_dataset(tmp_path)
    coco_api = create_yolo_coco_api(str(yaml_path), split="val")
    evaluator = COCOEvaluator(coco_api, iou_type="segm")

    mask = [[0, 0, 0], [0, 1, 1], [0, 1, 0]]
    evaluator.update(
        {
            "boxes": [[0, 0, 3, 3]],
            "scores": [0.9],
            "classes": [0],
            "masks": [mask],
        },
        image_id=0,
    )

    assert evaluator.results[0]["area"] == 3.0


@pytest.mark.unit
def test_detection_validator_emits_bbox_alias_metrics():
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.coco_evaluator = _DummyEvaluator(_coco_metrics(0.2))

    metrics = validator._compute_metrics()

    assert metrics["metrics/precision"] == pytest.approx(0.201)
    assert metrics["metrics/precision(B)"] == pytest.approx(0.201)
    assert metrics["metrics/recall"] == pytest.approx(0.202)
    assert metrics["metrics/recall(B)"] == pytest.approx(0.202)
    assert metrics["metrics/mAP50"] == pytest.approx(0.21)
    assert metrics["metrics/mAP50(B)"] == pytest.approx(0.21)
    assert metrics["metrics/mAP50-95"] == pytest.approx(0.2)
    assert metrics["metrics/mAP50-95(B)"] == pytest.approx(0.2)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("rfdetr", True),
        ("dfine", True),
        ("yolox", False),
    ],
)
def test_detection_validator_topk_coco_scoring_applies_to_yolo_coco_api(
    family, expected
):
    from types import SimpleNamespace

    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.model = SimpleNamespace(FAMILY=family)
    validator._coco_annotation_file = None

    assert validator._uses_topk_coco_scoring() is expected


@pytest.mark.unit
def test_detection_validator_parses_tiny_pixel_xyxy_targets():
    from libreyolo.validation.detection_validator import DetectionValidator

    validator = DetectionValidator.__new__(DetectionValidator)
    validator.val_preproc = SimpleNamespace(uses_letterbox=False)
    validator._actual_imgsz = 640
    validator.nc = 2

    targets = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    boxes, classes = validator._parse_gt_boxes(targets, orig_h=640, orig_w=640)

    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert classes.tolist() == [0]


@pytest.mark.unit
def test_segmentation_validator_updates_bbox_and_mask_evaluators():
    from types import SimpleNamespace

    from libreyolo.validation.detection_validator import SegmentationValidator

    validator = SegmentationValidator.__new__(SegmentationValidator)
    validator.bbox_evaluator = _DummyEvaluator()
    validator.coco_evaluator = validator.bbox_evaluator  # alias mirrors _init_metrics
    validator.mask_evaluator = _DummyEvaluator()

    pred = {"boxes": [], "scores": [], "classes": [], "masks": []}
    validator._update_metrics([pred], targets=None, img_info=[], img_ids=[123])

    assert validator.bbox_evaluator.updates == [(pred, 123)]
    assert validator.mask_evaluator.updates == [(pred, 123)]

    validator.config = SimpleNamespace(verbose=False, save_json=False)
    validator.save_dir = None
    validator.bbox_evaluator = _DummyEvaluator(_coco_metrics(0.2))
    validator.mask_evaluator = _DummyEvaluator(_coco_metrics(0.7))

    metrics = validator._compute_metrics()

    assert metrics["metrics/mAP50-95"] == pytest.approx(0.7)
    assert metrics["metrics/precision(B)"] == pytest.approx(0.201)
    assert metrics["metrics/recall(B)"] == pytest.approx(0.202)
    assert metrics["metrics/mAP50-95(B)"] == pytest.approx(0.2)
    assert metrics["metrics/precision(M)"] == pytest.approx(0.701)
    assert metrics["metrics/recall(M)"] == pytest.approx(0.702)
    assert metrics["metrics/mAP50-95(M)"] == pytest.approx(0.7)
