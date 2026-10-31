"""Unit tests for the YOLOv9 pre/post-processing helpers and trainer data paths."""

import json

import pytest
import numpy as np
import torch
from PIL import Image

from libreyolo.models.yolo9 import utils as yolo9_utils
from libreyolo.postprocess import yolo9 as yolo9_postprocess_mod
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.models.yolo9.transforms import YOLO9TrainTransform
from libreyolo.validation.preprocessors import YOLO9ValPreprocessor

pytestmark = pytest.mark.unit


class TestYOLO9Utils:
    """Test utility functions."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, original_img, original_size = yolo9_utils.preprocess_image(
            img, input_size=640
        )
        assert tensor.shape == (1, 3, 640, 640)
        assert original_size == (100, 100)

    def test_preprocess_image_letterboxes_non_square_like_validation(self):
        """Predict preprocessing must match YOLO9 validation geometry."""
        img = np.zeros((4, 8, 3), dtype=np.uint8)

        tensor, _, original_size = yolo9_utils.preprocess_image(
            img, input_size=8, color_format="rgb"
        )
        val_tensor, _ = YOLO9ValPreprocessor((8, 8), max_labels=1)(
            img[:, :, ::-1].copy(),
            np.zeros((0, 5), dtype=np.float32),
            (8, 8),
        )

        assert original_size == (8, 4)
        torch.testing.assert_close(tensor[0], torch.from_numpy(val_tensor))
        torch.testing.assert_close(
            tensor[0, :, 4:, :],
            torch.full((3, 4, 8), 114 / 255.0, dtype=tensor.dtype),
        )

    def test_preprocess_image_accepts_rectangular_input_size(self):
        img = np.zeros((4, 8, 3), dtype=np.uint8)

        tensor, _, original_size = yolo9_utils.preprocess_image(
            img, input_size=(8, 16), color_format="rgb"
        )

        assert original_size == (8, 4)
        assert tensor.shape == (1, 3, 8, 16)
        torch.testing.assert_close(
            tensor[0, :, :, :16],
            torch.zeros((3, 8, 16), dtype=tensor.dtype),
        )

    def test_postprocess_defaults_to_letterbox_inverse(self):
        """YOLO9 postprocess default matches letterboxed predict inputs."""
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
        pred[0, 4, 0] = 0.9

        out = yolo9_utils.postprocess(
            {"predictions": pred},
            input_size=640,
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
        )

    def test_postprocess_accepts_rectangular_input_size(self):
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
        pred[0, 4, 0] = 0.9

        out = yolo9_utils.postprocess(
            {"predictions": pred},
            input_size=(320, 640),
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["boxes"]),
            torch.tensor([[0.0, 0.0, 960.0, 960.0]]),
        )

    def test_postprocess_detection_is_multilabel(self):
        """Detection postprocess emits one detection per class above conf on an
        anchor (multi-label), matching MultimediaTechLab/YOLO ``bbox_nms``."""
        pred = torch.zeros(1, 6, 1)
        pred[0, :4, 0] = torch.tensor([0.0, 0.0, 100.0, 100.0])
        pred[0, 4:, 0] = torch.tensor([0.9, 0.8])  # two classes over conf

        out = yolo9_utils.postprocess(
            {"predictions": pred}, conf_thres=0.25, iou_thres=0.5
        )

        assert out["num_detections"] == 2
        assert sorted(out["classes"]) == [0, 1]

    def test_postprocess_detection_caps_multilabel_candidates(self, monkeypatch):
        """Detection limits low-threshold multi-label expansion before NMS."""
        # Patch the postprocess module — that's where postprocess() resolves it.
        monkeypatch.setattr(yolo9_postprocess_mod, "_YOLO9_NMS_PRE", 3)
        pred = torch.zeros(1, 6, 4)
        pred[0, :4] = torch.tensor(
            [
                [0.0, 20.0, 40.0, 60.0],
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 30.0, 50.0, 70.0],
                [10.0, 10.0, 10.0, 10.0],
            ]
        )
        pred[0, 4:] = torch.tensor(
            [[0.1, 0.9, 0.7, 0.5], [0.8, 0.2, 0.6, 0.4]]
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred}, conf_thres=0.01, iou_thres=0.5, max_det=3
        )

        assert out["num_detections"] == 3
        assert sorted(round(float(s), 1) for s in out["scores"]) == [
            0.7,
            0.8,
            0.9,
        ]

    def test_postprocess_obb_outputs_obb_payload(self):
        pred = torch.zeros(1, 7, 1)
        pred[0, :4, 0] = torch.tensor([10.0, 20.0, 50.0, 40.0])
        pred[0, 4, 0] = 0.25
        pred[0, 5:, 0] = torch.tensor([0.9, 0.1])

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
        )

        assert out["num_detections"] == 1
        assert len(out["obb"]) == 1
        torch.testing.assert_close(
            torch.as_tensor(out["obb"])[0, :5],
            torch.tensor([30.0, 30.0, 40.0, 20.0, 0.25]),
        )

    def test_postprocess_obb_uses_letterbox_inverse_for_non_square_images(self):
        pred = torch.zeros(1, 7, 1)
        pred[0, :4, 0] = torch.tensor([100.0, 50.0, 200.0, 150.0])
        pred[0, 4, 0] = 0.25
        pred[0, 5:, 0] = torch.tensor([0.9, 0.1])

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=640,
            original_size=(1280, 960),
        )

        assert out["num_detections"] == 1
        torch.testing.assert_close(
            torch.as_tensor(out["obb"])[0, :5],
            torch.tensor([300.0, 200.0, 200.0, 200.0, 0.25]),
        )

    def test_postprocess_obb_uses_classwise_rotated_nms(self):
        pred = torch.zeros(1, 7, 3)
        pred[0, :4] = torch.tensor(
            [
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
                [50.0, 50.0, 50.0],
                [40.0, 40.0, 40.0],
            ]
        )
        pred[0, 4] = 0.25
        pred[0, 5:] = torch.tensor(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.25,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
        )

        assert out["num_detections"] == 2
        assert out["classes"] == [0, 1]
        assert [round(score, 2) for score in out["scores"]] == [0.9, 0.7]

    def test_postprocess_obb_prefilters_candidates_before_rotated_nms(self, monkeypatch):
        num_candidates = 2000
        pred = torch.zeros(1, 7, num_candidates)
        pred[0, :4] = torch.tensor([[10.0], [20.0], [50.0], [40.0]]).expand(
            4, num_candidates
        )
        pred[0, 4] = 0.25
        pred[0, 5] = torch.linspace(0.9, 0.1, num_candidates)
        pred[0, 6] = 0.01

        exact_candidate_counts = []
        original_rotated_nms = yolo9_postprocess_mod._rotated_nms_keep_indices

        def wrapped_rotated_nms(xywhr, scores, class_ids, iou_thres, max_det):
            exact_candidate_counts.append(int(scores.numel()))
            return original_rotated_nms(xywhr, scores, class_ids, iou_thres, max_det)

        # Patch the postprocess module — that's where postprocess() resolves it.
        monkeypatch.setattr(
            yolo9_postprocess_mod,
            "_rotated_nms_keep_indices",
            wrapped_rotated_nms,
        )

        out = yolo9_utils.postprocess(
            {"predictions": pred, "obb": True},
            conf_thres=0.001,
            iou_thres=0.5,
            input_size=64,
            original_size=(64, 64),
            max_det=50,
        )

        assert out["num_detections"] == 1
        assert exact_candidate_counts
        assert exact_candidate_counts[0] <= yolo9_utils._YOLO9_OBB_MAX_NMS_CANDIDATES

    def test_obb_prefilter_does_not_apply_horizontal_nms(self):
        num_candidates = 400
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]]).expand(
            num_candidates, 4
        )
        scores = torch.linspace(1.0, 0.1, num_candidates)
        classes = torch.zeros(num_candidates, dtype=torch.long)

        keep = yolo9_utils._obb_prefilter_keep_indices(
            boxes,
            scores,
            classes,
            max_det=50,
        )

        assert keep.numel() == num_candidates
        torch.testing.assert_close(scores[keep], scores)


def test_yolo9_trainer_uses_explicit_coco_json_paths(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset

    image_dir = tmp_path / "images" / "custom_train"
    ann_dir = tmp_path / "custom_annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "train.json").write_text(
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
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: images/custom_train\n"
        "val: images/custom_train\n"
        "annotations:\n"
        "  train: custom_annotations/train.json\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    wrapper = type(
        "Wrapper",
        (),
        {"task": "detect", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        workers=0,
        device="cpu",
    )

    train_dataset = trainer._setup_data()

    assert isinstance(train_dataset.dataset, COCODataset)
    assert train_dataset.dataset.json_file == str(ann_dir / "train.json")
    assert train_dataset.dataset.name == str(image_dir)
    assert train_dataset.dataset._image_path(0) == image_dir / "sample.jpg"


def test_yolo9_trainer_uses_explicit_coco_json_paths_for_obb(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset

    image_dir = tmp_path / "images" / "custom_train"
    ann_dir = tmp_path / "custom_annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "train.json").write_text(
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
                        "obb": [10, 20, 50, 20, 50, 40, 10, 40],
                        "area": 800,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 42, "name": "vehicle"}],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: images/custom_train\n"
        "val: images/custom_train\n"
        "annotations:\n"
        "  train: custom_annotations/train.json\n"
        "nc: 1\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    wrapper = type(
        "Wrapper",
        (),
        {"task": "obb", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=100,
        workers=0,
        device="cpu",
    )

    train_dataset = trainer._setup_data()

    assert isinstance(train_dataset.dataset, COCODataset)
    assert train_dataset.dataset.load_obb is True
    assert train_dataset.dataset.json_file == str(ann_dir / "train.json")
    assert train_dataset.dataset.name == str(image_dir)
    labels = train_dataset.dataset.annotations[0][0]
    assert labels.shape == (1, 6)
    assert labels[0, 4] == 0


def test_yolo9_trainer_uses_default_coco_images_layout_for_obb_data_dir(tmp_path):
    pytest.importorskip("pycocotools")
    from libreyolo.data.dataset import COCODataset

    image_dir = tmp_path / "images" / "train2017"
    ann_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir()
    Image.new("RGB", (100, 100), color="white").save(image_dir / "sample.jpg")
    (ann_dir / "instances_train2017.json").write_text(
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
    wrapper = type(
        "Wrapper",
        (),
        {"task": "obb", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data_dir=str(tmp_path),
        num_classes=1,
        epochs=1,
        batch=1,
        imgsz=100,
        workers=0,
        device="cpu",
    )

    train_dataset = trainer._setup_data()

    assert isinstance(train_dataset.dataset, COCODataset)
    assert train_dataset.dataset.load_obb is True
    assert train_dataset.dataset.json_file == "instances_train2017.json"
    assert train_dataset.dataset.name == "images/train2017"
    assert train_dataset.dataset._image_path(0) == image_dir / "sample.jpg"


def test_yolo9_trainer_checkpoint_uses_resolved_data_classes_for_obb(tmp_path):
    from libreyolo.utils.serialization import load_trusted_torch_file

    image_dir = tmp_path / "train" / "images"
    label_dir = tmp_path / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: train/images\n"
        "nc: '1'\n"
        "names:\n"
        "  0: vehicle\n",
        encoding="utf-8",
    )
    wrapper = type(
        "Wrapper",
        (),
        {"task": "obb", "nb_classes": 1, "names": {0: "vehicle"}},
    )()
    trainer = YOLO9Trainer(
        model=torch.nn.Conv2d(3, 3, 1),
        wrapper_model=wrapper,
        data=str(data_yaml),
        epochs=1,
        batch=1,
        imgsz=64,
        workers=0,
        device="cpu",
    )

    trainer._setup_data()
    trainer.save_dir = tmp_path / "run"
    trainer.save_dir.mkdir()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer._save_checkpoint(epoch=0, loss=1.0, is_best=True)

    checkpoint = load_trusted_torch_file(
        trainer.save_dir / "weights" / "last.pt",
        map_location="cpu",
        context="unit test checkpoint",
    )
    assert trainer.config.num_classes == 1
    assert checkpoint["nc"] == 1
    assert checkpoint["config"]["num_classes"] == 1
