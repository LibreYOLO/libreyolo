"""COCO JSON datasets are out of LibreDoctor v1 scope (#1060).

diagnose() must reject any split with a resolved annotation file instead of
reporting every image as background, and the trainer must not suggest
`libreyolo doctor` for a dataset it reads from COCO JSON.
"""

import json

import pytest
import torch
import yaml

from libreyolo.doctor import DoctorError, diagnose
from libreyolo.doctor.config import NotADetectionDatasetError

pytestmark = pytest.mark.unit

DOCTOR_TIP = "libreyolo doctor"


def _coco_json(path, file_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": i, "file_name": name, "width": 64, "height": 64}
                    for i, name in enumerate(file_names, 1)
                ],
                "annotations": [
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1,
                        "bbox": [8, 8, 16, 16],
                        "area": 256,
                        "iscrowd": 0,
                    }
                    for i in range(1, len(file_names) + 1)
                ],
                "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# diagnose() guard
# ---------------------------------------------------------------------------


class TestDoctorRejectsCocoJson:
    def test_json_splits_rejected(self, make_dataset):
        ds = make_dataset()
        ds.image("train", "a.jpg", seed=1)
        ds.image("val", "b.jpg", seed=2)
        _coco_json(ds.root / "annotations" / "train.json", ["a.jpg"])
        _coco_json(ds.root / "annotations" / "val.json", ["b.jpg"])
        ds.set_yaml(
            annotations={
                "train": "annotations/train.json",
                "val": "annotations/val.json",
            }
        )
        with pytest.raises(DoctorError, match="COCO JSON annotations") as exc:
            diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert "train, val" in str(exc.value)

    def test_mixed_txt_train_and_json_val_rejected(self, make_dataset):
        ds = make_dataset()
        ds.sample("train", "a.jpg", seed=1)
        ds.image("val", "b.jpg", seed=2)
        _coco_json(ds.root / "annotations" / "val.json", ["b.jpg"])
        ds.set_yaml(annotations={"val": "annotations/val.json"})
        with pytest.raises(DoctorError, match=r"annotations set for: val\)"):
            diagnose(str(ds.yaml_path), fast=True, progress=False)

    def test_json_split_with_stale_txt_labels_rejected(self, make_dataset):
        # Leftover .txt files (here pose-shaped) must neither be checked as the
        # split's labels nor let the label-shape guard decide the verdict.
        ds = make_dataset()
        pose_line = "0 0.5 0.5 0.2 0.2 " + " ".join(["0.5 0.5 2"] * 17) + "\n"
        ds.sample("train", "a.jpg", seed=1, boxes=pose_line)
        ds.sample("val", "b.jpg", seed=2, boxes=pose_line)
        _coco_json(ds.root / "annotations" / "train.json", ["a.jpg"])
        ds.set_yaml(annotations={"train": "annotations/train.json"})
        with pytest.raises(DoctorError, match="COCO JSON annotations") as exc:
            diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert not isinstance(exc.value, NotADetectionDatasetError)

    @pytest.mark.parametrize("annotations", [{}, {"train": None, "val": ""}])
    def test_empty_annotations_mapping_passes_through(self, make_dataset, annotations):
        ds = make_dataset()
        ds.sample(
            "train", "a.jpg", seed=1, boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
        )
        ds.sample(
            "val", "b.jpg", seed=2, boxes="0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
        )
        ds.set_yaml(annotations=annotations)
        report = diagnose(str(ds.yaml_path), fast=True, progress=False)
        assert report.exit_code() == 0
        assert report.stats["splits"]["train"]["instances"] == 2


# ---------------------------------------------------------------------------
# Trainer tip
# ---------------------------------------------------------------------------


def _write_yaml(root, **extra):
    config = {"path": str(root), "nc": 2, "names": ["cat", "dog"], **extra}
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _image(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="white").save(path)


def _yolo_dataset(root):
    _image(root / "images" / "train" / "a.jpg")
    label = root / "labels" / "train" / "a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n")
    return _write_yaml(root, train="images/train", val="images/train")


def _explicit_json_dataset(root):
    _image(root / "images" / "train" / "a.jpg")
    _coco_json(root / "annotations" / "train.json", ["a.jpg"])
    return _write_yaml(
        root,
        train="images/train",
        val="images/train",
        annotations={"train": "annotations/train.json"},
    )


def _fallback_json_dataset(root):
    # No images where the YAML points, so training falls back to the
    # conventional annotations/instances_train2017.json layout.
    _image(root / "images" / "train2017" / "a.jpg")
    _coco_json(root / "annotations" / "instances_train2017.json", ["a.jpg"])
    return _write_yaml(root, train="images/missing", val="images/missing")


@pytest.mark.parametrize(
    ("build", "expect_tip"),
    [
        (_yolo_dataset, True),
        (_explicit_json_dataset, False),
        (_fallback_json_dataset, False),
    ],
    ids=["yolo_txt", "explicit_annotations", "instances_train2017_fallback"],
)
def test_trainer_doctor_tip_only_for_yolo_txt(tmp_path, monkeypatch, build, expect_tip):
    pytest.importorskip("pycocotools")
    from libreyolo.models.rtdetr.trainer import RTDETRTrainer
    from libreyolo.training import trainer as trainer_mod

    messages = []
    monkeypatch.setattr(
        trainer_mod.logger, "info", lambda msg, *a, **k: messages.append(str(msg))
    )

    data_yaml = build(tmp_path / "ds")
    trainer = RTDETRTrainer(
        model=torch.nn.Identity(),
        size="r18",
        num_classes=2,
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
    trainer._setup_data()

    assert any(m.startswith("Training dataset:") for m in messages)
    assert any(DOCTOR_TIP in m for m in messages) is expect_tip
