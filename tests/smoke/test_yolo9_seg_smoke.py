"""Offline smoke test for YOLOv9 instance segmentation.

Validates that the full training pipeline runs end-to-end on CPU with tiny
synthetic data: model construction, polygon label parsing, mask target
generation, forward/backward, loss computation, optimizer step, checkpoint
writing. Deliberately does NOT download weights or datasets.

Run: `pytest tests/smoke/test_yolo9_seg_smoke.py -v -m smoke`

Target: <60 seconds wall-clock on a MacBook (CPU).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model


pytestmark = pytest.mark.smoke


def _make_tiny_seg_dataset(root: Path, imgsz: int = 128, num_imgs: int = 2) -> Path:
    """Create a tiny YOLO-seg dataset with 1 polygon per image (2 classes).

    Layout:
        root/
            images/train/0.jpg, 1.jpg
            images/val/0.jpg
            labels/train/0.txt, 1.txt   (class cx cy w h  or  class x1 y1 x2 y2 ... polygon)
            labels/val/0.txt
            data.yaml
    """
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    def _write_sample(split: str, idx: int, cls: int):
        # Synthetic image: solid base + a bright square
        img = rng.integers(60, 100, size=(imgsz, imgsz, 3), dtype=np.uint8)
        x0, y0 = 24 + 12 * idx, 24 + 12 * idx
        x1, y1 = x0 + 40, y0 + 40
        img[y0:y1, x0:x1] = 220
        Image.fromarray(img).save(root / "images" / split / f"{idx}.jpg", quality=85)

        # Polygon label: rectangle as 4 points, normalized to [0, 1]
        xs = [x0, x1, x1, x0]
        ys = [y0, y0, y1, y1]
        coords = []
        for x, y in zip(xs, ys):
            coords.append(f"{x / imgsz:.6f}")
            coords.append(f"{y / imgsz:.6f}")
        line = f"{cls} " + " ".join(coords)
        (root / "labels" / split / f"{idx}.txt").write_text(line + "\n")

    for i in range(num_imgs):
        _write_sample("train", i, cls=i % 2)
    _write_sample("val", 0, cls=0)

    data_yaml = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["classA", "classB"],
    }
    (root / "data.yaml").write_text(yaml.dump(data_yaml))
    return root / "data.yaml"


def _make_seg_checkpoint(path: Path, size: str = "t", nb_classes: int = 2) -> Path:
    """Build a fresh YOLOv9 seg model and save its state_dict as a libreyolo checkpoint."""
    net = LibreYOLO9Model(config=size, nb_classes=nb_classes, segmentation=True)
    torch.save({"model": net.state_dict()}, path)
    return path


def test_yolo9_seg_training_runs_end_to_end(tmp_path):
    """Smoke: instantiate seg model, train 1 epoch on tiny data, verify finite loss + checkpoint."""
    # 1. Build dataset
    data_yaml = _make_tiny_seg_dataset(tmp_path / "data", imgsz=128, num_imgs=2)

    # 2. Build fresh seg checkpoint (no network)
    ckpt = _make_seg_checkpoint(tmp_path / "seg-init.pt", size="t", nb_classes=2)

    # 3. Load via wrapper with explicit segmentation=True
    model = LibreYOLO9(model_path=str(ckpt), size="t", nb_classes=2, segmentation=True, device="cpu")
    assert model._is_segmentation, "Model should be in segmentation mode"

    # 4. Train briefly (CPU, tiny batch, small imgsz)
    out = tmp_path / "runs"
    results = model.train(
        data=str(data_yaml),
        epochs=1,
        batch=2,
        imgsz=128,
        lr0=0.01,
        optimizer="SGD",
        device="cpu",
        workers=0,
        project=str(out),
        name="smoke",
        amp=False,
        patience=1,
    )

    # 5. Verify outputs
    assert "final_loss" in results
    assert math.isfinite(results["final_loss"]), f"loss not finite: {results['final_loss']}"
    assert Path(results["last_checkpoint"]).exists(), "last checkpoint missing"

    # 6. Verify checkpoint has seg head keys
    saved = torch.load(results["last_checkpoint"], map_location="cpu", weights_only=False)
    state = saved.get("model", saved)
    seg_keys = [k for k in state if k.startswith("head.proto") or k.startswith("head.cv4")]
    assert seg_keys, "Seg head keys missing from saved checkpoint"


def test_yolo9_seg_inference_returns_masks(tmp_path):
    """Smoke: run inference with a fresh seg model, confirm Results.masks is populated shape-wise."""
    ckpt = _make_seg_checkpoint(tmp_path / "seg-init.pt", size="t", nb_classes=2)
    model = LibreYOLO9(model_path=str(ckpt), size="t", nb_classes=2, segmentation=True, device="cpu")

    # Synthetic RGB image
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)
    pil = Image.fromarray(img)

    # Run with a very low confidence threshold so we're likely to get at least one detection
    results = model(pil, conf=0.0001, iou=0.5)
    assert results is not None
    # Results may be a list or a single object depending on input type — normalize
    res = results[0] if isinstance(results, list) else results

    # If there are any detections, masks must be present and shape-consistent
    if len(res.boxes) > 0:
        assert res.masks is not None, "seg model returned boxes but no masks"
        assert res.masks.data.shape[0] == len(res.boxes), "mask count != box count"
