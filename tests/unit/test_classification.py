"""Image-classification task tests.

Covers the shared classification stack (ImageFolder dataset, collate,
ClassifyValidator, Results.probs) and the LibreDINOv2 model wiring (classify is
a DINOv2 linear probe, not the RF-DETR detector, so it lives in the dinov2
family). All tests run on CPU with a tiny synthetic ImageFolder so they need no
network or GPU.
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


def test_classify_validator_maps_imagenet_wnid_subset_to_head_indices(tmp_path):
    from libreyolo.data.imagenet import imagenet1k_names
    from libreyolo.validation import ClassifyValidator, ValidationConfig

    _make_named_imagefolder(tmp_path, ["n01440764", "n15075141"])

    class _Model:
        names = imagenet1k_names()
        nb_classes = 1000

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
    assert labels_by_path == {"n01440764": 0, "n15075141": 999}
    assert validator._num_classes == 1000


@pytest.mark.external_data
@pytest.mark.network
@pytest.mark.slow
def test_dinov2_classify_forward():
    """LibreDINOv2 classify build + forward (DINOv2 backbone; random-init if offline)."""
    from libreyolo.models.dinov2.model import LibreDINOv2

    m = LibreDINOv2(
        model_path=None, size="n", task="classify", nb_classes=4, device="cpu"
    )
    assert m.task == "classify"
    assert m.input_size == 224
    # The classify wrapper exposes the linear-probe classifier and builds no
    # DETR decoder (model is None signals the non-detection head path).
    assert m.model.model is None
    assert m.model.classifier is not None

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


def test_dinov2_classify_can_load_and_detect_nb_classes():
    pytest.importorskip("transformers")
    from libreyolo.models.dinov2.model import LibreDINOv2
    from libreyolo.models.rfdetr.model import LibreRFDETR

    weights = {
        "backbone.encoder.encoder.embeddings.position_embeddings": torch.empty(
            1, 1370, 384
        ),
        "linear.weight": torch.empty(4, 256),
    }

    # Classify checkpoints (backbone + linear head) belong to LibreDINOv2 now;
    # RF-DETR must no longer claim them.
    assert LibreDINOv2.can_load(weights)
    assert not LibreRFDETR.can_load(weights)
    assert LibreDINOv2.detect_nb_classes(weights) == 4


def test_dinov2_classify_load_infers_nc_from_linear_weight(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    from libreyolo.models.dinov2.model import LibreDINOv2

    class _LoadResult:
        missing_keys = []
        unexpected_keys = []

    class _FakeClassifier(torch.nn.Module):
        def __init__(self, nb_classes):
            super().__init__()
            self.linear = torch.nn.Linear(2, nb_classes)
            self.nb_classes = nb_classes

    class _FakeWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = None  # signals the non-detection (linear-head) path
            self.classifier = _FakeClassifier(80)
            self.nb_classes = 80

        def load_state_dict(self, loaded, strict=False):
            state = loaded.get("model", loaded)
            expected = self.classifier.linear.out_features
            actual = state["linear.weight"].shape[0]
            if expected != actual:
                raise RuntimeError(f"expected {expected} classifier rows, got {actual}")
            return _LoadResult()

    monkeypatch.setattr(LibreDINOv2, "_init_model", lambda self: _FakeWrapper())
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": {
                "backbone.stem.weight": torch.ones(1),
                "linear.weight": torch.ones(4, 2),
                "linear.bias": torch.ones(4),
            },
            "model_family": "dinov2",
            "size": "n",
            "task": "classify",
        },
        path,
    )

    model = LibreDINOv2(str(path), size="n", task="classify", device="cpu")

    assert model.nb_classes == 4
    assert model.model.classifier.linear.out_features == 4


def test_classify_family_train_end_to_end(tmp_path):
    """A classify family must actually fine-tune through the shared trainer.

    Regression guard for two integration bugs that unit-level dataset/forward
    tests never exercised:

    * the trainer built ``ClassifyDataset`` with ``crop_pct=``/``interpolation=``
      as direct kwargs while the dataset takes them via ``transform_kwargs`` —
      every ``model.train(...)`` raised ``TypeError`` on setup; and
    * the classify configs inherited the detection ``eval_interval=10``, so the
      documented short ``epochs=5`` fine-tune never validated and never wrote a
      ``best.pt``.

    Uses the smallest family (MobileNetV4-s) on a random-init model + tiny
    synthetic ImageFolder, so it stays CPU-only and needs no network.
    """
    from libreyolo import LibreMobileNetV4

    _make_imagefolder(tmp_path / "data", n_classes=2, n_per=6, size=64)

    model = LibreMobileNetV4(size="s", device="cpu")
    assert model.nb_classes == 1000

    metrics = model.train(
        data=str(tmp_path / "data"),
        epochs=1,
        batch=4,
        imgsz=32,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="cls_smoke",
        exist_ok=True,
    )

    # Head was rebuilt 1000 -> 2 for the dataset's class count.
    assert model.nb_classes == 2

    # eval_interval=1 => the single epoch validated and produced a best.pt.
    best = tmp_path / "runs" / "cls_smoke" / "weights" / "best.pt"
    assert best.exists(), "classification training must save best.pt"
    epoch_metrics = metrics.get("epoch_metrics", [])
    assert epoch_metrics and epoch_metrics[-1].get("validated") is True
    val = epoch_metrics[-1].get("val_metrics") or {}
    scalars = val.get("metrics", val)
    assert "metrics/accuracy_top1" in scalars

    # The saved checkpoint reloads as a 2-class classifier and predicts.
    reloaded = LibreMobileNetV4(str(best), device="cpu")
    assert reloaded.nb_classes == 2
    result = reloaded(str(tmp_path / "data" / "val" / "c0" / "c0_0.png"))
    assert result.probs.data.shape[0] == 2


def test_classify_validator_macro_precision_recall_f1():
    """Macro P/R/F1 from the confusion matrix, over classes present in targets."""
    from contextlib import nullcontext

    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    validator.loss_adapter = None
    validator._autocast_context = nullcontext
    validator._init_metrics()

    def logits_for(preds, nc=4):
        out = torch.full((len(preds), nc), -5.0)
        for row, cls in enumerate(preds):
            out[row, cls] = 5.0
        return out

    # Batch 1: targets [0, 0, 1, 1], preds [0, 1, 1, 1]
    validator._update_metrics(
        logits_for([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]), None
    )
    # Batch 2: targets [2, 2], preds [2, 0]
    validator._update_metrics(logits_for([2, 0]), torch.tensor([2, 2]), None)

    metrics = validator._compute_metrics()

    # Confusion (rows=target, cols=pred):
    #   c0: tp=1 fp=1 fn=1 -> P=0.5  R=0.5  F1=0.5
    #   c1: tp=2 fp=1 fn=0 -> P=2/3  R=1.0  F1=0.8
    #   c2: tp=1 fp=0 fn=1 -> P=1.0  R=0.5  F1=2/3
    #   c3: absent from targets -> excluded from the macro mean
    assert metrics["metrics/accuracy_top1"] == pytest.approx(4 / 6)
    assert metrics["metrics/precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert metrics["metrics/recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert metrics["metrics/f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert metrics["fitness"] == pytest.approx(metrics["metrics/accuracy_top1"])


def test_classify_validator_macro_metrics_are_zero_without_samples():
    from contextlib import nullcontext

    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    validator.loss_adapter = None
    validator._autocast_context = nullcontext
    validator._init_metrics()

    metrics = validator._compute_metrics()

    assert metrics["metrics/precision"] == 0.0
    assert metrics["metrics/recall"] == 0.0
    assert metrics["metrics/f1"] == 0.0


def test_classify_validator_precision_is_zero_for_never_predicted_class():
    """A class present in the targets but never predicted gets precision 0, not NaN."""
    from contextlib import nullcontext

    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    validator.loss_adapter = None
    validator._autocast_context = nullcontext
    validator._init_metrics()

    def logits_for(preds, nc=3):
        out = torch.full((len(preds), nc), -5.0)
        for row, cls in enumerate(preds):
            out[row, cls] = 5.0
        return out

    # targets [0, 0, 1, 2], preds [0, 0, 0, 0]: class 1 and class 2 have
    # support 1 each but are never predicted (tp + fp == 0 for both).
    validator._update_metrics(
        logits_for([0, 0, 0, 0]), torch.tensor([0, 0, 1, 2]), None
    )

    metrics = validator._compute_metrics()

    # Confusion (rows=target, cols=pred):
    #   c0: tp=2 fp=2 fn=0 -> P=0.5    R=1.0  F1=2/3
    #   c1: tp=0 fp=0 fn=1 -> P=0      R=0    F1=0
    #   c2: tp=0 fp=0 fn=1 -> P=0      R=0    F1=0
    assert metrics["metrics/precision"] == pytest.approx((0.5 + 0 + 0) / 3)
    assert metrics["metrics/recall"] == pytest.approx((1.0 + 0 + 0) / 3)
    assert metrics["metrics/f1"] == pytest.approx((2 / 3 + 0 + 0) / 3)
    for key in ("metrics/precision", "metrics/recall", "metrics/f1"):
        assert metrics[key] == metrics[key]  # NaN check: NaN != NaN.


def test_classify_validator_ignores_out_of_range_targets_in_confusion():
    """A target >= nc (dataset/head class-count mismatch) must not crash.

    Before this feature such a sample simply counted as wrong; it must keep
    doing so rather than raising out of ``torch.bincount``.
    """
    from contextlib import nullcontext

    from libreyolo.validation.classify_validator import ClassifyValidator

    validator = object.__new__(ClassifyValidator)
    validator.loss_adapter = None
    validator._autocast_context = nullcontext
    validator._init_metrics()

    def logits_for(preds, nc=2):
        out = torch.full((len(preds), nc), -5.0)
        for row, cls in enumerate(preds):
            out[row, cls] = 5.0
        return out

    # preds [0, 0], targets [2, 0]: target 2 is out of range for nc=2.
    validator._update_metrics(logits_for([0, 0]), torch.tensor([2, 0]), None)

    metrics = validator._compute_metrics()

    # The out-of-range sample counts as wrong (unchanged semantics): 1/2 correct.
    assert metrics["metrics/accuracy_top1"] == pytest.approx(0.5)
    assert validator._confusion.tolist() == [[1, 0], [0, 0]]
    # Only class 0 has support; it is predicted perfectly.
    assert metrics["metrics/precision"] == pytest.approx(1.0)
    assert metrics["metrics/recall"] == pytest.approx(1.0)
    assert metrics["metrics/f1"] == pytest.approx(1.0)


def test_classify_train_with_best_metric_f1_records_key(tmp_path):
    """best_metric=f1 drives best.pt selection and lands in checkpoint metadata."""
    from libreyolo import LibreMobileNetV4

    _make_imagefolder(tmp_path / "data", n_classes=2, n_per=6, size=64)

    model = LibreMobileNetV4(size="s", device="cpu")
    metrics = model.train(
        data=str(tmp_path / "data"),
        epochs=1,
        batch=4,
        imgsz=32,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="cls_f1",
        exist_ok=True,
        best_metric="f1",
    )

    epoch_metrics = metrics.get("epoch_metrics", [])
    assert epoch_metrics and epoch_metrics[-1].get("validated") is True
    # epoch_metrics' val_metrics is the flattened scalar dict the validator
    # reports (see BaseTrainer._validation_metrics_for_event), not the
    # internal dict the trainer builds around it -- it does not carry
    # best_metric_key. The checkpoint's top-level best_metric_key is the
    # contract for that, asserted below.
    val = epoch_metrics[-1].get("val_metrics") or {}
    scalars = val.get("metrics", val)
    assert "metrics/f1" in scalars

    best = tmp_path / "runs" / "cls_f1" / "weights" / "best.pt"
    assert best.exists()
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["best_metric_key"] == "metrics/f1"


# ---------------------------------------------------------------------------
# Classification augmentation pack: auto_augment / erasing / mixup / cutmix.
# ---------------------------------------------------------------------------


def _train_op_names(**kwargs):
    from libreyolo.data.classify_dataset import build_classify_transforms

    compose = build_classify_transforms(224, True, **kwargs)
    return [type(op).__name__ for op in compose.transforms]


def test_transforms_default_off_is_unchanged():
    """Default-off knobs must not alter the train transform list (regression)."""
    assert _train_op_names() == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "ToTensor",
        "Normalize",
    ]
    # Passing the knobs at their off-defaults is byte-identical to omitting them.
    assert _train_op_names(auto_augment=None, erasing=0.0) == _train_op_names()


@pytest.mark.parametrize(
    "policy,cls_name",
    [
        ("randaugment", "RandAugment"),
        ("autoaugment", "AutoAugment"),
        ("augmix", "AugMix"),
    ],
)
def test_auto_augment_inserted_before_totensor(policy, cls_name):
    names = _train_op_names(auto_augment=policy)
    assert cls_name in names
    # Auto-augment acts on PIL/uint8, so it sits after the flip and before ToTensor.
    assert names.index(cls_name) == names.index("RandomHorizontalFlip") + 1
    assert names.index(cls_name) < names.index("ToTensor")


def test_erasing_appended_after_normalize():
    names = _train_op_names(erasing=0.4)
    assert names[-1] == "RandomErasing"
    assert names.index("RandomErasing") > names.index("Normalize")


def test_square_resize_with_augment_raises():
    """square_resize is a val-only transform; the augment branch used to return
    first and ignore it silently."""
    from libreyolo.data.classify_dataset import build_classify_transforms

    with pytest.raises(ValueError, match="square_resize"):
        build_classify_transforms(224, True, square_resize=True)
    # The val pipeline still honours it.
    assert build_classify_transforms(224, False, square_resize=True) is not None


def test_val_pipeline_ignores_augment_knobs():
    from libreyolo.data.classify_dataset import build_classify_transforms

    compose = build_classify_transforms(
        224, False, auto_augment="randaugment", erasing=0.5
    )
    names = [type(op).__name__ for op in compose.transforms]
    assert names == ["Resize", "CenterCrop", "ToTensor", "Normalize"]


def test_unknown_auto_augment_raises():
    from libreyolo.data.classify_dataset import build_classify_transforms

    with pytest.raises(ValueError, match="randaugment, autoaugment, augmix"):
        build_classify_transforms(224, True, auto_augment="cutout")


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_erasing_bounds_validated(bad):
    from libreyolo.data.classify_dataset import build_classify_transforms

    with pytest.raises(ValueError, match="erasing must be in"):
        build_classify_transforms(224, True, erasing=bad)


def test_build_classify_collate_off_returns_plain_fn():
    from libreyolo.data.classify_dataset import (
        build_classify_collate,
        classify_collate_fn,
    )

    assert build_classify_collate(5, 0.0, 0.0) is classify_collate_fn


@pytest.mark.parametrize(
    "mixup,cutmix",
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
)
def test_mixing_produces_soft_labels(mixup, cutmix):
    """MixUp / CutMix keep image shape and emit soft labels that sum to 1."""
    from libreyolo.data.classify_dataset import build_classify_collate

    num_classes = 4
    collate = build_classify_collate(num_classes, mixup=mixup, cutmix=cutmix)
    batch = [(torch.rand(3, 16, 16), i % num_classes) for i in range(6)]
    imgs, labels, infos, ids = collate(batch)

    assert imgs.shape == (6, 3, 16, 16)
    assert labels.shape == (6, num_classes)
    assert labels.dtype.is_floating_point
    assert torch.allclose(labels.sum(dim=1), torch.ones(6), atol=1e-5)
    assert len(infos) == 6 and len(ids) == 6


def test_soft_labels_feed_cross_entropy():
    """The classify criterion (cross-entropy) must accept mixed soft targets."""
    from libreyolo.data.classify_dataset import build_classify_collate

    num_classes = 3
    collate = build_classify_collate(num_classes, mixup=1.0)
    batch = [(torch.rand(3, 16, 16), i % num_classes) for i in range(4)]
    _, soft_labels, _, _ = collate(batch)

    logits = torch.randn(4, num_classes, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, soft_labels)
    loss.backward()
    assert loss.item() > 0
    assert logits.grad is not None


def test_build_classify_collate_rejects_out_of_range_prob():
    from libreyolo.data.classify_dataset import build_classify_collate

    with pytest.raises(ValueError, match="mixup must be in"):
        build_classify_collate(4, mixup=1.5)
    with pytest.raises(ValueError, match="cutmix must be in"):
        build_classify_collate(4, cutmix=-0.2)


def test_augment_knobs_plumb_through_trainer(tmp_path):
    """auto_augment / erasing / mixup must flow config -> dataset -> collate.

    Drives the shared classification data setup the way ``model.train(...)``
    does, then inspects the built loader so the plumbing is verified without a
    full training run.
    """
    from libreyolo import LibreMobileNetV4
    from libreyolo.data.classify_dataset import _ClassifyBatchMixer
    from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer

    _make_imagefolder(tmp_path / "data", n_classes=3, n_per=4, size=64)
    model = LibreMobileNetV4(size="s", device="cpu")

    trainer = MobileNetV4Trainer(
        model=model.model,
        wrapper_model=model,
        size="s",
        num_classes=model.nb_classes,
        data=str(tmp_path / "data"),
        imgsz=32,
        batch=4,
        workers=0,
        device="cpu",
        auto_augment="randaugment",
        erasing=0.3,
        mixup=0.5,
    )

    assert trainer.config.auto_augment == "randaugment"
    assert trainer.config.erasing == 0.3
    assert trainer.config.mixup == 0.5

    trainer._setup_classify_data()

    # Mixing wraps the collate; the train transform carries the aug ops.
    assert isinstance(trainer.train_loader.collate_fn, _ClassifyBatchMixer)
    op_names = [
        type(op).__name__
        for op in trainer.train_loader.dataset._impl.transform.transforms
    ]
    assert "RandAugment" in op_names
    assert op_names[-1] == "RandomErasing"


def test_public_train_api_accepts_augment_pack(tmp_path):
    """End-to-end: model.train(...) accepts the pack and completes a run.

    Exercises the full public path (wrapper.train -> trainer -> dataset/collate
    -> soft-target cross-entropy) with MixUp on so the soft-label loss is hit.
    """
    from libreyolo import LibreMobileNetV4

    _make_imagefolder(tmp_path / "data", n_classes=2, n_per=6, size=64)
    model = LibreMobileNetV4(size="s", device="cpu")

    metrics = model.train(
        data=str(tmp_path / "data"),
        epochs=1,
        batch=4,
        imgsz=32,
        workers=0,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="aug_smoke",
        exist_ok=True,
        auto_augment="randaugment",
        erasing=0.25,
        mixup=1.0,
    )

    assert model.nb_classes == 2
    best = tmp_path / "runs" / "aug_smoke" / "weights" / "best.pt"
    assert best.exists()
    epoch_metrics = metrics.get("epoch_metrics", [])
    assert epoch_metrics and epoch_metrics[-1].get("validated") is True
