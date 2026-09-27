"""Unit tests for the FiftyOne integration.

The geometry conversions are tested without fiftyone installed: they return
plain dicts keyed by FiftyOne's field names, so the payloads can be checked
directly. Tests that need the package itself are marked ``fiftyone``.
"""

import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from libreyolo.integrations.fiftyone import (
    _bounding_box,
    _class_name,
    _classification_payload,
    _crop_mask,
    _detection_payloads,
    _import_fiftyone,
    _keypoint_payloads,
    _obb_payloads,
    _polyline_payloads,
    _sample_labels_from_rows,
    _yolo_rows,
)
from libreyolo.utils.results import (
    OBB,
    Boxes,
    Keypoints,
    Masks,
    Probs,
    Results,
)

pytestmark = pytest.mark.unit

NAMES = {0: "person", 1: "bicycle"}


def _detect_result(orig_shape=(100, 200)):
    boxes = Boxes(
        torch.tensor([[10.0, 20.0, 110.0, 70.0], [0.0, 0.0, 20.0, 10.0]]),
        torch.tensor([0.9, 0.4]),
        torch.tensor([0.0, 1.0]),
    )
    return Results(boxes, orig_shape=orig_shape, names=NAMES)


class TestGeometry:
    def test_bounding_box_is_normalized_xywh(self):
        assert _bounding_box([10, 20, 110, 70], 200, 100) == pytest.approx(
            [0.05, 0.2, 0.5, 0.5]
        )

    def test_bounding_box_clips_to_the_image(self):
        # A box that runs off the canvas cannot overlap ground truth outside
        # the image, so clipping keeps IoU honest and the App drawable.
        assert _bounding_box([-50, -10, 250, 120], 200, 100) == pytest.approx(
            [0.0, 0.0, 1.0, 1.0]
        )

    def test_class_name_falls_back_to_the_id(self):
        assert _class_name(NAMES, 0) == "person"
        assert _class_name(NAMES, 7) == "7"
        assert _class_name(["a", "b"], 1) == "b"
        assert _class_name(None, 3) == "3"

    def test_crop_mask_returns_the_box_region(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:70, 10:110] = True
        crop = _crop_mask(mask, [10, 20, 110, 70], 200, 100)
        assert crop.shape == (50, 100)
        assert crop.all()

    def test_crop_mask_rejects_a_degenerate_box(self):
        mask = np.zeros((10, 10), dtype=bool)
        assert _crop_mask(mask, [5, 5, 5, 5], 10, 10) is None


class TestDetectionPayloads:
    def test_boxes_become_normalized_detections(self):
        payloads = _detection_payloads(_detect_result())
        assert [p["label"] for p in payloads] == ["person", "bicycle"]
        assert payloads[0]["bounding_box"] == pytest.approx([0.05, 0.2, 0.5, 0.5])
        assert payloads[0]["confidence"] == pytest.approx(0.9)
        assert "index" not in payloads[0]
        assert "mask" not in payloads[0]

    def test_empty_results_produce_no_detections(self):
        boxes = Boxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)))
        result = Results(boxes, orig_shape=(100, 200), names=NAMES)
        assert _detection_payloads(result) == []

    def test_track_ids_become_the_detection_index(self):
        result = _detect_result()
        result.boxes = result.boxes.with_id(torch.tensor([7.0, 8.0]))
        payloads = _detection_payloads(result)
        assert [p["index"] for p in payloads] == [7, 8]

    def test_instance_masks_are_cropped_to_their_box(self):
        result = _detect_result()
        masks = np.zeros((2, 100, 200), dtype=np.float32)
        masks[0, 20:70, 10:110] = 1.0
        masks[1, 0:10, 0:20] = 1.0
        result.masks = Masks(masks, (100, 200))
        payloads = _detection_payloads(result)
        assert payloads[0]["mask"].shape == (50, 100)
        assert payloads[1]["mask"].shape == (10, 20)
        assert payloads[0]["mask"].dtype == bool


class TestOtherTaskPayloads:
    def test_obb_becomes_a_closed_filled_polyline(self):
        # An axis-aligned box at angle 0: cx, cy, w, h, r, conf, cls.
        data = torch.tensor([[100.0, 50.0, 40.0, 20.0, 0.0, 0.8, 1.0]])
        result = Results(None, orig_shape=(100, 200), names=NAMES, obb=OBB(data))
        payloads = _obb_payloads(result)
        assert len(payloads) == 1
        assert payloads[0]["label"] == "bicycle"
        assert payloads[0]["closed"] is True
        assert payloads[0]["filled"] is True
        ring = payloads[0]["points"][0]
        assert len(ring) == 4
        xs = sorted(x for x, _ in ring)
        ys = sorted(y for _, y in ring)
        assert xs[0] == pytest.approx(0.4) and xs[-1] == pytest.approx(0.6)
        assert ys[0] == pytest.approx(0.4) and ys[-1] == pytest.approx(0.6)

    def test_keypoints_carry_per_point_confidence(self):
        data = torch.tensor([[[100.0, 50.0, 0.9], [50.0, 25.0, 0.7]]])
        result = _detect_result()
        result.boxes = result.boxes[0:1]
        result.keypoints = Keypoints(data, (100, 200))
        payloads = _keypoint_payloads(result)
        assert len(payloads) == 1
        assert payloads[0]["label"] == "person"
        assert payloads[0]["points"][0] == pytest.approx([0.5, 0.5])
        assert payloads[0]["points"][1] == pytest.approx([0.25, 0.25])
        assert payloads[0]["confidence"] == pytest.approx([0.9, 0.7])

    def test_invisible_keypoints_become_nan(self):
        data = torch.tensor([[[100.0, 50.0, 0.9], [0.0, 0.0, 0.0]]])
        result = Results(None, orig_shape=(100, 200), names=NAMES)
        result.keypoints = Keypoints(data, (100, 200))
        points = _keypoint_payloads(result)[0]["points"]
        assert points[0] == pytest.approx([0.5, 0.5])
        assert all(np.isnan(v) for v in points[1])

    def test_classification_reports_top1(self):
        probs = Probs(torch.tensor([0.1, 0.9]))
        result = Results(None, orig_shape=(100, 200), names=NAMES, probs=probs)
        payload = _classification_payload(result)
        assert payload == {"label": "bicycle", "confidence": pytest.approx(0.9)}

    def test_masks_become_normalized_polygon_rings(self):
        masks = np.zeros((1, 100, 200), dtype=np.uint8)
        masks[0, 20:70, 10:110] = 1
        result = _detect_result()
        result.boxes = result.boxes[0:1]
        result.masks = Masks(masks, (100, 200))
        payloads = _polyline_payloads(result)
        assert len(payloads) == 1
        assert payloads[0]["label"] == "person"
        ring = payloads[0]["points"][0]
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in ring)


class _StubLabel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _stub_fiftyone():
    """A stand-in for the fiftyone label classes used by the yaml bridge."""
    return types.SimpleNamespace(
        Detection=_StubLabel,
        Detections=_StubLabel,
        Polyline=_StubLabel,
        Polylines=_StubLabel,
    )


class TestDatasetBridge:
    def test_yolo_rows_skips_blank_lines(self, tmp_path):
        label_file = tmp_path / "a.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.4\n\n1 0.1 0.1 0.1 0.1\n")
        assert _yolo_rows(label_file) == [
            [0.0, 0.5, 0.5, 0.2, 0.4],
            [1.0, 0.1, 0.1, 0.1, 0.1],
        ]

    def test_yolo_rows_treats_a_missing_file_as_no_objects(self, tmp_path):
        assert _yolo_rows(tmp_path / "missing.txt") == []

    def test_detect_rows_become_corner_normalized_boxes(self):
        labels = _sample_labels_from_rows(
            _stub_fiftyone(), [[0.0, 0.5, 0.5, 0.2, 0.4]], NAMES, "detect"
        )
        assert len(labels.detections) == 1
        assert labels.detections[0].label == "person"
        assert labels.detections[0].bounding_box == pytest.approx([0.4, 0.3, 0.2, 0.4])

    def test_segment_rows_become_polygons(self):
        rows = [[1.0, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5]]
        labels = _sample_labels_from_rows(_stub_fiftyone(), rows, NAMES, "segment")
        assert len(labels.polylines) == 1
        assert labels.polylines[0].label == "bicycle"
        assert labels.polylines[0].points == [[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5)]]
        assert labels.polylines[0].closed is True

    def test_a_box_row_in_a_segment_dataset_becomes_a_rectangle(self):
        labels = _sample_labels_from_rows(
            _stub_fiftyone(), [[0.0, 0.5, 0.5, 0.2, 0.4]], NAMES, "segment"
        )
        assert labels.polylines[0].points == [
            [(0.4, 0.3), (0.6, 0.3), (0.6, 0.7), (0.4, 0.7)]
        ]


class TestOptionalDependency:
    def test_importing_libreyolo_does_not_import_fiftyone(self):
        code = "import libreyolo, sys; print('fiftyone' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"

    def test_missing_fiftyone_raises_an_install_hint(self, monkeypatch):
        # A None entry in sys.modules makes ``import fiftyone`` fail the same
        # way an uninstalled package does.
        monkeypatch.setitem(sys.modules, "fiftyone", None)
        with pytest.raises(ImportError, match=r"pip install libreyolo\[fiftyone\]"):
            _import_fiftyone()


@pytest.mark.fiftyone
class TestWithFiftyOneInstalled:
    def test_labels_convert_to_fiftyone_types(self):
        fo = pytest.importorskip("fiftyone")
        from libreyolo.integrations.fiftyone import _to_fiftyone_labels

        labels = _to_fiftyone_labels(_detect_result())
        assert isinstance(labels, fo.Detections)
        assert len(labels.detections) == 2
        assert labels.detections[0].label == "person"

    def test_native_coco_json_annotations_are_loaded(self, tmp_path):
        pytest.importorskip("fiftyone")
        import json

        import yaml
        from PIL import Image

        from libreyolo.integrations.fiftyone import to_fiftyone

        root = tmp_path / "coco"
        (root / "images" / "val").mkdir(parents=True)
        (root / "annotations").mkdir()
        Image.new("RGB", (200, 100)).save(root / "images" / "val" / "a.jpg")
        (root / "annotations" / "val.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [80, 30, 40, 40],
                            "area": 1600,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": 1, "name": "person"}],
                }
            )
        )
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "val": "images/val",
                    "names": {0: "person"},
                    "annotations": {"val": "annotations/val.json"},
                }
            )
        )

        dataset = to_fiftyone(data_yaml, split="val", autodownload=False)
        try:
            assert len(dataset) == 1
            detections = dataset.first().ground_truth.detections
            assert len(detections) == 1
            assert detections[0].label == "person"
            assert detections[0].bounding_box == pytest.approx([0.4, 0.3, 0.2, 0.4])
        finally:
            dataset.delete()

    def test_export_keeps_the_class_ids_it_is_given(self, tmp_path):
        fo = pytest.importorskip("fiftyone")
        from PIL import Image

        from libreyolo.integrations.fiftyone import from_fiftyone

        image = tmp_path / "a.jpg"
        Image.new("RGB", (200, 100)).save(image)
        dataset = fo.Dataset()
        sample = fo.Sample(filepath=str(image))
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(label="bicycle", bounding_box=[0.4, 0.3, 0.2, 0.4])
            ]
        )
        dataset.add_samples([sample])
        try:
            yaml_path = from_fiftyone(
                dataset,
                tmp_path / "export",
                split="train",
                classes=["person", "bicycle"],
            )
            import yaml as pyyaml

            written = pyyaml.safe_load(yaml_path.read_text())
            assert written["names"] == {0: "person", 1: "bicycle"}
            label_file = next((tmp_path / "export").rglob("a.txt"))
            assert label_file.read_text().split()[0] == "1"
        finally:
            dataset.delete()

    def test_two_splits_accumulate_in_one_yaml(self, tmp_path):
        fo = pytest.importorskip("fiftyone")
        import yaml as pyyaml
        from PIL import Image

        from libreyolo.data.utils import load_data_config
        from libreyolo.integrations.fiftyone import from_fiftyone

        def make(name, label):
            image = tmp_path / f"{name}.jpg"
            Image.new("RGB", (200, 100)).save(image)
            sample = fo.Sample(filepath=str(image))
            sample["ground_truth"] = fo.Detections(
                detections=[
                    fo.Detection(label=label, bounding_box=[0.4, 0.3, 0.2, 0.4])
                ]
            )
            dataset = fo.Dataset()
            dataset.add_samples([sample])
            return dataset

        train = make("train_a", "person")
        val = make("val_a", "bicycle")
        export_dir = tmp_path / "export"
        try:
            classes = ["person", "bicycle"]
            from_fiftyone(train, export_dir, split="train", classes=classes)
            yaml_path = from_fiftyone(val, export_dir, split="val", classes=classes)

            # The second export must add its split rather than replace the
            # first: a training yaml needs train and val together.
            written = pyyaml.safe_load(yaml_path.read_text())
            assert "train" in written and "val" in written

            config = load_data_config(str(yaml_path), autodownload=False)
            assert len(config["train_img_files"]) == 1
            assert len(config["val_img_files"]) == 1
            assert config["train_img_files"][0].name == "train_a.jpg"
            assert config["val_img_files"][0].name == "val_a.jpg"
        finally:
            train.delete()
            val.delete()

    def test_round_trip_through_a_dataset_yaml(self, tmp_path):
        pytest.importorskip("fiftyone")
        import yaml
        from PIL import Image

        from libreyolo.integrations.fiftyone import from_fiftyone, to_fiftyone

        root = tmp_path / "ds"
        (root / "images" / "val").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)
        Image.new("RGB", (200, 100)).save(root / "images" / "val" / "a.jpg")
        (root / "labels" / "val" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/val",
                    "val": "images/val",
                    "names": {0: "person", 1: "bicycle"},
                }
            )
        )

        dataset = to_fiftyone(data_yaml, split="val", autodownload=False)
        try:
            assert len(dataset) == 1
            detections = dataset.first().ground_truth.detections
            assert len(detections) == 1
            assert detections[0].label == "person"

            out_yaml = from_fiftyone(dataset, tmp_path / "export", split="val")
            assert out_yaml.exists()

            from libreyolo.data.utils import load_data_config

            config = load_data_config(str(out_yaml), autodownload=False)
            assert len(config["val_img_files"]) == 1
        finally:
            dataset.delete()
