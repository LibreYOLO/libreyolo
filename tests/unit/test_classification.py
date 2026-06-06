"""Image-classification task tests for YOLO9 and RF-DETR.

Covers the shared classification stack (ImageFolder dataset, collate,
ClassifyValidator, Results.probs) and the per-family model wiring. All tests
run on CPU with a tiny synthetic ImageFolder so they need no network or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _make_imagefolder(root, n_classes=3, n_per=5, size=64):
    """Create a tiny train/val ImageFolder where each class has a distinct hue.

    Distinct per-class colors make the set trivially separable so a couple of
    training steps demonstrably reduce the loss.
    """
    classes = [f"c{i}" for i in range(n_classes)]
    for split in ("train", "val"):
        for ci, name in enumerate(classes):
            cls_dir = root / split / name
            cls_dir.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200  # dominant channel per class
            for j in range(n_per):
                noisy = np.clip(
                    base + np.random.randint(0, 40, base.shape, dtype=np.int16),
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(noisy).save(cls_dir / f"{name}_{j}.png")
    return classes


def test_classify_dataset_and_collate(tmp_path):
    from libreyolo.data import ClassifyDataset, classify_collate_fn, get_class_names

    classes = _make_imagefolder(tmp_path, n_classes=3, n_per=4)
    assert get_class_names(tmp_path, "train") == sorted(classes)

    ds = ClassifyDataset(tmp_path, split="train", imgsz=32, augment=False)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)

    batch = [ds[i] for i in range(4)]
    imgs, labels, infos, ids = classify_collate_fn(batch)
    assert imgs.shape == (4, 3, 32, 32)
    assert labels.shape == (4,) and labels.dtype == torch.long
    assert len(infos) == 4 and len(ids) == 4


def test_yolo9_classify_forward_and_rebuild():
    from libreyolo import LibreYOLO9

    m = LibreYOLO9(None, size="t", task="classify", nb_classes=4, device="cpu")
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.model.neck is None  # detection neck/head skipped

    x = torch.randn(2, 3, 224, 224)
    m.model.train()
    out = m.model(x, targets=torch.tensor([0, 3]))
    assert "total_loss" in out and out["total_loss"].requires_grad

    m.model.eval()
    with torch.no_grad():
        logits = m.model(x)
    assert logits.shape == (2, 4)

    m._rebuild_for_new_classes(7)
    with torch.no_grad():
        assert m.model(x).shape == (2, 7)


def test_classify_validator_top1_top5(tmp_path):
    from libreyolo import LibreYOLO9
    from libreyolo.validation import ClassifyValidator, ValidationConfig

    classes = _make_imagefolder(tmp_path, n_classes=3, n_per=4)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=len(classes), device="cpu")
    m.model.eval()

    cfg = ValidationConfig(
        data=str(tmp_path), batch_size=4, imgsz=32, device="cpu",
        num_workers=0, split="val", verbose=False,
    )
    metrics = ClassifyValidator(model=m, config=cfg).run()
    assert "metrics/accuracy_top1" in metrics
    assert "metrics/accuracy_top5" in metrics
    assert 0.0 <= metrics["metrics/accuracy_top1"] <= 1.0
    # With 3 classes, top-5 collapses to top-3 and must cover everything.
    assert metrics["metrics/accuracy_top5"] == pytest.approx(1.0)


def test_yolo9_classify_predict_returns_probs(tmp_path):
    from libreyolo import LibreYOLO9

    classes = _make_imagefolder(tmp_path, n_classes=3, n_per=2)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=len(classes), device="cpu")
    m.names = {i: n for i, n in enumerate(classes)}

    img_path = next((tmp_path / "val").rglob("*.png"))
    result = m.predict(str(img_path))
    assert result.probs is not None
    assert 0 <= result.probs.top1 < len(classes)
    assert len(result.probs.top5) <= len(classes)
    assert result.boxes is None


def test_yolo9_classify_train_smoke(tmp_path):
    """A couple of epochs on the synthetic set run end-to-end and reduce loss."""
    from libreyolo import LibreYOLO9

    _make_imagefolder(tmp_path, n_classes=3, n_per=8, size=64)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=3, device="cpu")

    res = m.train(
        data=str(tmp_path), epochs=3, batch=8, imgsz=64, optimizer="adamw",
        lr0=1e-3, workers=0, eval_interval=1, project=str(tmp_path / "runs"),
        name="cls_smoke", exist_ok=True, amp=False, ema=False, warmup_epochs=0,
    )
    losses = res["epoch_losses"]
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    # Trivially-separable data: loss should fall over the run.
    assert losses[-1] < losses[0]
    assert res["epoch_metrics"][-1]["val_metrics"].get("metrics/accuracy_top1") is not None


@pytest.mark.slow
def test_rfdetr_classify_forward():
    """RF-DETR classify build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo import LibreRFDETR

    m = LibreRFDETR(model_path=None, size="n", task="classify", nb_classes=4, device="cpu")
    assert m.task == "classify"
    assert m.input_size == 224
    assert m.model.classification

    x = torch.randn(1, 3, 224, 224)
    m.model.train()
    out = m.model(x, targets=torch.tensor([2]))
    assert "total_loss" in out

    m.model.eval()
    with torch.no_grad():
        assert m.model(x).shape == (1, 4)
