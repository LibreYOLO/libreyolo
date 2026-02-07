"""
RF1: Training smoke test for all 15 models.

Trains each model for 2 epochs on underwater-pipes-4ng4t (Roboflow 100,
5617 train / 1575 valid / 779 test images, 1 class), then validates on the
test split. This dataset converges fast — yolox-nano hits 0.56 mAP50-95
in just 2 epochs.

Usage:
    pytest tests/e2e/test_rf1_training.py -v -m e2e
    pytest tests/e2e/test_rf1_training.py::test_rf1_training[yolox-nano] -v
    pytest tests/e2e/test_rf1_training.py -k "rfdetr" -v
"""

import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LIBREYOLO

DATASET_ROOT = Path.home() / ".cache" / "rf100" / "underwater-pipes-4ng4t"

# (weights, size, family)
MODELS = [
    # YOLOX
    ("libreyoloXnano.pt",    "nano", "yolox"),
    ("libreyoloXtiny.pt",    "tiny", "yolox"),
    ("libreyoloXs.pt",       "s",    "yolox"),
    ("libreyoloXm.pt",       "m",    "yolox"),
    ("libreyoloXl.pt",       "l",    "yolox"),
    ("libreyoloXx.pt",       "x",    "yolox"),
    # YOLOv9
    ("libreyolo9t.pt",       "t",    "v9"),
    ("libreyolo9s.pt",       "s",    "v9"),
    ("libreyolo9m.pt",       "m",    "v9"),
    ("libreyolo9c.pt",       "c",    "v9"),
    # RF-DETR
    ("librerfdetrnano.pth",  "n",    "rfdetr"),
    ("librerfdetrsmall.pth", "s",    "rfdetr"),
    ("librerfdetrbase.pth",  "b",    "rfdetr"),
    ("librerfdetrmedium.pth","m",    "rfdetr"),
    ("librerfdetrlarge.pth", "l",    "rfdetr"),
]

IDS = [
    "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x",
    "v9-t", "v9-s", "v9-m", "v9-c",
    "rfdetr-n", "rfdetr-s", "rfdetr-b", "rfdetr-m", "rfdetr-l",
]


@pytest.fixture(scope="module")
def dataset_coco():
    """Convert YOLO labels to COCO JSON for RF-DETR training.

    Writes _annotations.coco.json into each split dir (train/valid/test).
    Idempotent — skips if annotations already exist.
    """
    assert DATASET_ROOT.exists(), (
        f"Dataset not found at {DATASET_ROOT}. "
        "Download it first via tests/e2e/rf5_datasets.py"
    )

    with open(DATASET_ROOT / "data.yaml") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    for split in ["train", "valid", "test"]:
        ann_file = DATASET_ROOT / split / "_annotations.coco.json"
        if ann_file.exists():
            continue

        images_dir = DATASET_ROOT / split / "images"
        labels_dir = DATASET_ROOT / split / "labels"

        images_list, annotations_list = [], []
        ann_id = 0

        for img_id, img_path in enumerate(sorted(images_dir.glob("*.jpg"))):
            with Image.open(img_path) as img:
                w, h = img.size

            images_list.append({
                "id": img_id,
                "file_name": f"images/{img_path.name}",
                "width": w, "height": h,
            })

            label_file = labels_dir / img_path.with_suffix(".txt").name
            if label_file.exists():
                for line in label_file.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    box_w, box_h = bw * w, bh * h

                    annotations_list.append({
                        "id": ann_id, "image_id": img_id,
                        "category_id": cls_id + 1,
                        "bbox": [round(x, 2), round(y, 2),
                                 round(box_w, 2), round(box_h, 2)],
                        "area": round(box_w * box_h, 2), "iscrowd": 0,
                    })
                    ann_id += 1

        coco = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }
        ann_file.write_text(json.dumps(coco))

    return DATASET_ROOT


@pytest.fixture(scope="module")
def dataset_data_yaml():
    """Return data.yaml path with absolute path for training code."""
    assert DATASET_ROOT.exists(), (
        f"Dataset not found at {DATASET_ROOT}. "
        "Download it first via tests/e2e/rf5_datasets.py"
    )

    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())

    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        yaml.dump(data, data_yaml.open("w"), default_flow_style=False)

    return str(data_yaml)


MIN_MAP = 0.18


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size,family", MODELS, ids=IDS)
def test_rf1_training(weights, size, family, dataset_coco, dataset_data_yaml,
                      tmp_path):
    """Train 2 epochs on underwater-pipes, validate on test split."""
    model = LIBREYOLO(weights, size=size)

    if family == "rfdetr":
        model.train(
            data=str(dataset_coco),
            epochs=2,
            batch_size=16,
            output_dir=str(tmp_path / f"rfdetr_{size}"),
        )
    else:
        model.train(
            data=dataset_data_yaml,
            epochs=2,
            batch=128,
            workers=2,
            project=str(tmp_path),
            name=f"{family}_{size}",
            exist_ok=True,
        )

    results = model.val(data=dataset_data_yaml, split="test",
                        batch=16, conf=0.001, iou=0.6)
    map50_95 = results["metrics/mAP50-95"]

    print(f"\n  {weights} post-training mAP50-95={map50_95:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"Post-training mAP50-95={map50_95:.4f} below {MIN_MAP}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
