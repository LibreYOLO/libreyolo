"""
RF1: Training smoke test for all 15 models.

Trains each model for 2 epochs on Libre-YOLO/marbles (HuggingFace, public,
56 train / 20 valid / 36 test images, 2 classes), then validates on the test
split. The dataset auto-downloads from HuggingFace — no API keys needed.

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
from huggingface_hub import snapshot_download
from PIL import Image

from libreyolo import LibreYOLO

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "marbles"
HF_REPO = "Libre-YOLO/marbles"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_REPO}"

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
    ("librerfdetrmedium.pth","m",    "rfdetr"),
    ("librerfdetrlarge.pth", "l",    "rfdetr"),
]

IDS = [
    "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x",
    "v9-t", "v9-s", "v9-m", "v9-c",
    "rfdetr-n", "rfdetr-s", "rfdetr-m", "rfdetr-l",
]


def download_marbles_dataset():
    """Download the marbles dataset from HuggingFace if not already cached.

    Uses huggingface_hub snapshot_download which handles auth, caching,
    and LFS automatically. Creates a symlink at DATASET_ROOT pointing
    to the HF cache snapshot.
    """
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        return

    print(f"\nDownloading dataset {HF_REPO} from HuggingFace ...")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATASET_ROOT),
    )
    print(f"Dataset downloaded to {snapshot_path}")


def patch_data_yaml():
    """Ensure data.yaml has an absolute path so training resolves splits."""
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))


@pytest.fixture(scope="module")
def dataset():
    """Download marbles dataset and patch data.yaml. Shared by all fixtures."""
    download_marbles_dataset()
    patch_data_yaml()
    return DATASET_ROOT


@pytest.fixture(scope="module")
def dataset_coco(dataset):
    """Convert YOLO labels to COCO JSON for RF-DETR training.

    Writes _annotations.coco.json into each split dir (train/valid/test).
    Idempotent — skips if annotations already exist.
    Reads class names from data.yaml dynamically.
    """
    with open(dataset / "data.yaml") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    # Handle both list and dict formats for names
    # RF-DETR (Roboflow format) uses 0-indexed category IDs
    if isinstance(class_names, dict):
        categories = [
            {"id": i, "name": class_names[i], "supercategory": "object"}
            for i in sorted(class_names.keys())
        ]
    else:
        categories = [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ]

    for split in ["train", "valid", "test"]:
        ann_file = dataset / split / "_annotations.coco.json"
        if ann_file.exists():
            continue

        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"

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
                        "category_id": cls_id,
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

    return dataset


@pytest.fixture(scope="module")
def dataset_data_yaml(dataset):
    """Return data.yaml path with absolute path for training code."""
    return str(dataset / "data.yaml")


MIN_MAP = 0.05


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size,family", MODELS, ids=IDS)
def test_rf1_training(weights, size, family, dataset_coco, dataset_data_yaml,
                      tmp_path):
    """Train 2 epochs on marbles, validate on test split."""
    model = LibreYOLO(weights, size=size)

    if family == "rfdetr":
        model.train(
            data=str(dataset_coco),
            epochs=10,
            batch_size=2,
            output_dir=str(tmp_path / f"rfdetr_{size}"),
        )
    else:
        model.train(
            data=dataset_data_yaml,
            epochs=10,
            batch=16,
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


# ---------------------------------------------------------------------------
# Phase 2: Reload fine-tuned checkpoints into fresh models
# ---------------------------------------------------------------------------

# YOLOX/V9: all models
RELOAD_MODELS = [
    ("libreyoloXnano.pt",    "nano", "yolox"),
    ("libreyoloXtiny.pt",    "tiny", "yolox"),
    ("libreyoloXs.pt",       "s",    "yolox"),
    ("libreyoloXm.pt",       "m",    "yolox"),
    ("libreyoloXl.pt",       "l",    "yolox"),
    ("libreyoloXx.pt",       "x",    "yolox"),
    ("libreyolo9t.pt",       "t",    "v9"),
    ("libreyolo9s.pt",       "s",    "v9"),
    ("libreyolo9m.pt",       "m",    "v9"),
    ("libreyolo9c.pt",       "c",    "v9"),
]
RELOAD_IDS = [
    "yolox-nano", "yolox-tiny", "yolox-s", "yolox-m", "yolox-l", "yolox-x",
    "v9-t", "v9-s", "v9-m", "v9-c",
]


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size,family", RELOAD_MODELS, ids=RELOAD_IDS)
def test_load_finetuned_checkpoint(weights, size, family, dataset_coco,
                                   dataset_data_yaml, tmp_path):
    """Train, save checkpoint, load into fresh model, validate.

    Verifies that fine-tuned checkpoints can be loaded in a new session
    with correct nc, names, and architecture auto-rebuild.
    """
    # 1. Train
    model = LibreYOLO(weights, size=size)
    model.train(
        data=dataset_data_yaml,
        epochs=10,
        batch=16,
        workers=2,
        project=str(tmp_path),
        name=f"{family}_{size}",
        exist_ok=True,
    )

    # 2. Find best.pt on disk
    best_pt = tmp_path / f"{family}_{size}" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = tmp_path / f"{family}_{size}" / "weights" / "last.pt"
    assert best_pt.exists(), f"No checkpoint found at {best_pt}"

    # 3. Verify checkpoint has metadata
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    assert "nc" in ckpt, "Checkpoint missing 'nc' metadata"
    assert "names" in ckpt, "Checkpoint missing 'names' metadata"
    assert "model_family" in ckpt, "Checkpoint missing 'model_family' metadata"
    assert ckpt["nc"] == 2, f"Expected nc=2 (marbles), got {ckpt['nc']}"
    assert ckpt["model_family"] == family
    print(f"\n  Checkpoint metadata: nc={ckpt['nc']}, family={ckpt['model_family']}, "
          f"names={ckpt['names']}")

    # 4. Load into a completely fresh model (default nc=80)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fresh_model = LibreYOLO(str(best_pt), size=size)

    # 5. Verify auto-rebuild happened
    assert fresh_model.nb_classes == 2, (
        f"Expected nb_classes=2 after loading, got {fresh_model.nb_classes}"
    )
    assert len(fresh_model.names) == 2, (
        f"Expected 2 names, got {len(fresh_model.names)}"
    )

    # 6. Validate on test split
    results = fresh_model.val(data=dataset_data_yaml, split="test",
                              batch=16, conf=0.001, iou=0.6)
    map50_95 = results["metrics/mAP50-95"]

    print(f"  {weights} reloaded checkpoint mAP50-95={map50_95:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"Reloaded model mAP50-95={map50_95:.4f} below {MIN_MAP}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# RF-DETR: reload fine-tuned checkpoint
RELOAD_RFDETR_MODELS = [
    ("librerfdetrnano.pth", "n", "rfdetr"),
]
RELOAD_RFDETR_IDS = ["rfdetr-n"]


@pytest.mark.e2e
@pytest.mark.parametrize("weights,size,family", RELOAD_RFDETR_MODELS, ids=RELOAD_RFDETR_IDS)
def test_load_finetuned_checkpoint_rfdetr(weights, size, family, dataset_coco,
                                          dataset_data_yaml, tmp_path):
    """Train RF-DETR, save checkpoint, load into fresh model, validate.

    RF-DETR uses a different checkpoint format (checkpoint_best_total.pth)
    and requires manual detection head reinitialization.
    """
    from pathlib import Path as P

    # 1. Train
    model = LibreYOLO(weights, size=size)
    output_dir = str(tmp_path / f"rfdetr_{size}")
    model.train(
        data=str(dataset_coco),
        epochs=10,
        batch_size=4,
        output_dir=output_dir,
    )

    # 2. Find checkpoint on disk
    best_ckpt = P(output_dir) / "checkpoint_best_total.pth"
    if not best_ckpt.exists():
        # Fall back to any checkpoint
        ckpts = sorted(P(output_dir).glob("checkpoint*.pth"))
        assert ckpts, f"No checkpoint found in {output_dir}"
        best_ckpt = ckpts[-1]

    # 3. Verify checkpoint structure
    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    assert "model" in ckpt, "RF-DETR checkpoint missing 'model' key"
    state_dict = ckpt["model"]
    assert "class_embed.bias" in state_dict, "Missing class_embed in state dict"
    num_classes_internal = state_dict["class_embed.bias"].shape[0]
    num_classes = num_classes_internal - 1  # RF-DETR uses nc+1 (background)
    assert num_classes == 2, f"Expected nc=2 (marbles), got {num_classes}"
    print(f"\n  RF-DETR checkpoint: nc={num_classes}, internal={num_classes_internal}")

    # 4. Load into a fresh model and manually load checkpoint
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fresh_model = LibreYOLO(weights, size=size)

    # Reinitialize detection head and load trained weights
    if num_classes_internal != fresh_model.model.model.class_embed.bias.shape[0]:
        fresh_model.model.model.reinitialize_detection_head(num_classes_internal)
    fresh_model.model.model.load_state_dict(state_dict, strict=False)
    fresh_model.model.model.eval()
    fresh_model.model.model.to(fresh_model.device)
    fresh_model.nb_classes = num_classes
    fresh_model.model.nb_classes = num_classes

    # 5. Validate on test split
    results = fresh_model.val(data=dataset_data_yaml, split="test",
                              batch=16, conf=0.001, iou=0.6)
    map50_95 = results["metrics/mAP50-95"]

    print(f"  {weights} reloaded checkpoint mAP50-95={map50_95:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"Reloaded model mAP50-95={map50_95:.4f} below {MIN_MAP}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
