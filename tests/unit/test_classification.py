"""Image-classification task tests for YOLO9 and RF-DETR.

Covers the shared classification stack (ImageFolder dataset, collate,
ClassifyValidator, Results.probs) and the per-family model wiring. All tests
run on CPU with a tiny synthetic ImageFolder so they need no network or GPU.
"""

from __future__ import annotations

from pathlib import Path

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


def _make_named_imagefolder(root, classes, n_per=2, size=32):
    for split in ("train", "val"):
        for ci, name in enumerate(classes):
            cls_dir = root / split / name
            cls_dir.mkdir(parents=True, exist_ok=True)
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[:, :, ci % 3] = 200
            for j in range(n_per):
                Image.fromarray(base).save(cls_dir / f"{name}_{j}.png")


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


def test_classify_dataset_rejects_unknown_split_classes(tmp_path):
    from libreyolo.data import ClassifyDataset

    _make_named_imagefolder(tmp_path, ["cat"])
    extra_dir = tmp_path / "val" / "dog"
    extra_dir.mkdir()
    Image.new("RGB", (16, 16)).save(extra_dir / "dog.png")

    with pytest.raises(ValueError, match="unknown classes"):
        ClassifyDataset(
            tmp_path,
            split="val",
            imgsz=32,
            augment=False,
            class_to_idx={"cat": 0},
        )


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
    m = LibreYOLO9(
        None, size="t", task="classify", nb_classes=len(classes), device="cpu"
    )
    m.model.eval()

    cfg = ValidationConfig(
        data=str(tmp_path),
        batch_size=4,
        imgsz=32,
        device="cpu",
        num_workers=0,
        split="val",
        verbose=False,
    )
    metrics = ClassifyValidator(model=m, config=cfg).run()
    assert "metrics/accuracy_top1" in metrics
    assert "metrics/accuracy_top5" in metrics
    assert 0.0 <= metrics["metrics/accuracy_top1"] <= 1.0
    # With 3 classes, top-5 collapses to top-3 and must cover everything.
    assert metrics["metrics/accuracy_top5"] == pytest.approx(1.0)


def test_classify_validator_uses_model_name_order(tmp_path):
    from libreyolo.validation import ClassifyValidator, ValidationConfig

    _make_named_imagefolder(tmp_path, ["cat", "dog"])

    class _Model:
        names = {0: "dog", 1: "cat"}
        nb_classes = 2

    validator = ClassifyValidator(
        model=_Model(),
        config=ValidationConfig(
            data=str(tmp_path),
            batch_size=4,
            imgsz=32,
            device="cpu",
            num_workers=0,
            split="val",
            verbose=False,
        ),
    )

    dataloader = validator._setup_dataloader()
    labels_by_path = {
        Path(path).parent.name: int(label)
        for path, label in dataloader.dataset._impl.samples
    }

    assert labels_by_path["dog"] == 0
    assert labels_by_path["cat"] == 1


def test_yolo9_classify_predict_returns_probs(tmp_path):
    from libreyolo import LibreYOLO9

    classes = _make_imagefolder(tmp_path, n_classes=3, n_per=2)
    m = LibreYOLO9(
        None, size="t", task="classify", nb_classes=len(classes), device="cpu"
    )
    m.names = {i: n for i, n in enumerate(classes)}

    img_path = next((tmp_path / "val").rglob("*.png"))
    result = m.predict(str(img_path))
    assert result.probs is not None
    assert 0 <= result.probs.top1 < len(classes)
    assert len(result.probs.top5) <= len(classes)
    assert result.boxes is None

    aug_result = m.predict(str(img_path), augment=True)
    assert aug_result.probs is not None
    assert aug_result.boxes is None


def test_yolo9_classify_predict_save_and_tiling_do_not_require_boxes(tmp_path):
    from libreyolo import LibreYOLO9

    _make_imagefolder(tmp_path, n_classes=2, n_per=2)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=2, device="cpu")
    m.model.eval()
    img_path = next((tmp_path / "val").rglob("*.png"))

    save_path = tmp_path / "plain.jpg"
    result = m.predict(str(img_path), save=True, output_path=str(save_path), imgsz=32)

    assert result.boxes is None
    assert result.probs is not None
    assert save_path.exists()

    tiled_path = tmp_path / "tiled.jpg"
    tiled_result = m.predict(
        str(img_path),
        save=True,
        output_path=str(tiled_path),
        tiling=True,
        imgsz=32,
    )

    assert tiled_result.boxes is None
    assert tiled_result.probs is not None
    assert tiled_path.exists()


def test_yolo9_classify_track_rejected():
    from libreyolo import LibreYOLO9

    m = LibreYOLO9(None, size="t", task="classify", nb_classes=3, device="cpu")

    with pytest.raises(NotImplementedError, match="classification models"):
        next(m.track("missing.mp4"))


def test_yolo9_classify_train_smoke(tmp_path):
    """A couple of epochs on the synthetic set run end-to-end and reduce loss."""
    from libreyolo import LibreYOLO9

    _make_imagefolder(tmp_path, n_classes=3, n_per=8, size=64)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=3, device="cpu")

    res = m.train(
        data=str(tmp_path),
        epochs=3,
        batch=8,
        imgsz=64,
        optimizer="adamw",
        lr0=1e-3,
        workers=0,
        eval_interval=1,
        project=str(tmp_path / "runs"),
        name="cls_smoke",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    losses = res["epoch_losses"]
    assert len(losses) == 3
    assert all(np.isfinite(losses))
    # Trivially-separable data: loss should fall over the run.
    assert losses[-1] < losses[0]
    assert (
        res["epoch_metrics"][-1]["val_metrics"].get("metrics/accuracy_top1") is not None
    )


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_rfdetr_classify_forward():
    """RF-DETR classify build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo import LibreRFDETR

    m = LibreRFDETR(
        model_path=None, size="n", task="classify", nb_classes=4, device="cpu"
    )
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


def test_safe_zip_extraction_rejects_path_traversal(tmp_path):
    import zipfile

    from libreyolo.data.classify_dataset import _safe_extract_zip

    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "payload")
    with zipfile.ZipFile(bad_zip) as zf:
        with pytest.raises(ValueError):
            _safe_extract_zip(zf, tmp_path / "dest")


def test_yolo9_classify_task_inferred_on_load(tmp_path):
    """A saved classification checkpoint loads without re-specifying task=."""
    from libreyolo import LibreYOLO9

    _make_imagefolder(tmp_path, n_classes=3, n_per=6, size=64)
    m = LibreYOLO9(None, size="t", task="classify", nb_classes=3, device="cpu")
    res = m.train(
        data=str(tmp_path),
        epochs=1,
        batch=8,
        imgsz=64,
        optimizer="adamw",
        lr0=1e-3,
        workers=0,
        eval_interval=0,
        project=str(tmp_path / "runs"),
        name="ckpt",
        exist_ok=True,
        amp=False,
        ema=False,
        warmup_epochs=0,
    )
    ckpt = res.get("last_checkpoint") or res.get("best_checkpoint")
    assert ckpt is not None

    reloaded = LibreYOLO9(ckpt, size="t", device="cpu")  # no task= passed
    assert reloaded.task == "classify"
    assert reloaded.nb_classes == 3


def test_yolo9_classify_checkpoint_metadata_beats_stale_filename_suffix(tmp_path):
    from libreyolo import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    m = LibreYOLO9(None, size="t", task="classify", nb_classes=2, device="cpu")
    ckpt = wrap_libreyolo_checkpoint(
        m.model.state_dict(),
        model_family="yolo9",
        size="t",
        task="classify",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=224,
    )
    path = tmp_path / "LibreYOLO9t-seg.pt"
    torch.save(ckpt, path)

    reloaded = LibreYOLO9(str(path), size="t", device="cpu")

    assert reloaded.task == "classify"
    assert reloaded.nb_classes == 2


def test_yolo9_classify_rejects_metadata_less_detection_weights(tmp_path):
    from libreyolo import LibreYOLO9

    detect = LibreYOLO9(None, size="t", task="detect", nb_classes=3, device="cpu")
    path = tmp_path / "detect.pt"
    torch.save(detect.model.state_dict(), path)

    with pytest.raises(RuntimeError, match="cannot be loaded as task='classify'"):
        LibreYOLO9(str(path), size="t", task="classify", nb_classes=3, device="cpu")


def test_yolo9_classify_allows_detect_transfer_weights(tmp_path):
    from libreyolo import LibreYOLO9
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    detect = LibreYOLO9(None, size="t", task="detect", nb_classes=80, device="cpu")
    transfer_path = tmp_path / "detect.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            detect.model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=80,
            names=detect.names,
            imgsz=640,
        ),
        transfer_path,
    )

    classify = LibreYOLO9(None, size="t", task="classify", nb_classes=3, device="cpu")
    stats = classify._load_transfer_weights(transfer_path)

    assert stats["loaded"] > 0
    assert classify.model.head.linear.out_features == 3


def test_yolo9_raw_classify_checkpoint_infers_task_from_head(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9(None, size="t", task="classify", nb_classes=2, device="cpu")
    path = tmp_path / "best.pt"
    torch.save(model.model.state_dict(), path)

    loaded = LibreYOLO(str(path), device="cpu")

    assert loaded.task == "classify"
    assert loaded.nb_classes == 2


def test_rfdetr_classify_detect_size_uses_metadata():
    pytest.importorskip("transformers")
    from libreyolo.models.rfdetr.model import LibreRFDETR

    weights = {
        "backbone.encoder.encoder.embeddings.position_embeddings": torch.empty(
            1, 1370, 384
        ),
        "linear.weight": torch.empty(4, 256),
    }
    checkpoint = {"model_family": "rfdetr", "size": "n", "task": "classify"}

    assert LibreRFDETR.detect_size(weights, state_dict=checkpoint) == "n"


def test_rfdetr_classify_load_infers_nc_from_linear_weight(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from libreyolo.models.rfdetr.model import LibreRFDETR

    class _LoadResult:
        missing_keys = []
        unexpected_keys = []

    class _FakeClassifier(torch.nn.Module):
        def __init__(self, nb_classes):
            super().__init__()
            self.linear = torch.nn.Linear(2, nb_classes)
            self.nb_classes = nb_classes

    class _FakeRFDETRModel(torch.nn.Module):
        classification = True

        def __init__(self):
            super().__init__()
            self.classifier = _FakeClassifier(80)
            self.nb_classes = 80

        def load_state_dict(self, loaded, strict=False):
            state = loaded.get("model", loaded)
            expected = self.classifier.linear.out_features
            actual = state["linear.weight"].shape[0]
            if expected != actual:
                raise RuntimeError(f"expected {expected} classifier rows, got {actual}")
            return _LoadResult()

    monkeypatch.setattr(LibreRFDETR, "_init_model", lambda self: _FakeRFDETRModel())
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": {
                "backbone.stem.weight": torch.ones(1),
                "linear.weight": torch.ones(4, 2),
                "linear.bias": torch.ones(4),
            },
            "model_family": "rfdetr",
            "size": "n",
            "task": "classify",
        },
        path,
    )

    model = LibreRFDETR(str(path), size="n", task="classify", device="cpu")

    assert model.nb_classes == 4
    assert model.model.classifier.linear.out_features == 4


def test_yolo9_classify_export_onnx(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO9

    m = LibreYOLO9(None, size="t", task="classify", nb_classes=5, device="cpu")
    path = m.export(format="onnx", output_path=str(tmp_path / "y9_cls.onnx"))

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    out = sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
    assert out[0].shape == (1, 5)
