"""predict(), val() and exported backends share one eval transform (#886).

A classifier must be scored on exactly the preprocessing it is deployed with.
Each family declares its eval pipeline once (``model.eval_transform``); these
tests pin that ``predict()`` preprocessing, the validation dataset, and the
exported-backend transform all produce the same tensor for the same image.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import libreyolo
from libreyolo.validation.classify_validator import ClassifyValidator
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit

# Built for real (cheap on CPU), so the test sees what __init__ sets.
BUILT = [
    ("LibreAlexNet", "b"),
    ("LibreVGG", "16"),
    ("LibreResNet", "18"),
    ("LibreEfficientNetV2", "b1"),
    ("LibreConvNeXt", "t"),
    ("LibreConvNeXtV2", "atto"),
    ("LibreMobileNetV4", "m"),
    ("LibreDeiT", "t"),
    ("LibreSwin", "t"),
    ("LibreViT", "ti"),
]
# Large towers: their eval settings are class attributes, so a bare instance
# (no network) carries the exact pipeline.
BARE = [
    ("LibreCLIP", "b32"),
    ("LibreSigLIP2", "b16"),
    ("LibrePE", "b16"),
    ("LibreVJEPA2", "l256"),
]

# Odd, non-square size so resize, crop and squash all matter.
_IMAGE = Image.fromarray(
    np.random.default_rng(886).integers(0, 256, (293, 411, 3), dtype=np.uint8)
)


def _built(name, size):
    return getattr(libreyolo, name)(size=size, nb_classes=3, device="cpu")


def _bare(name, size):
    cls = getattr(libreyolo, name)
    model = cls.__new__(cls)
    model.size = size
    model.input_size = cls.INPUT_SIZES[size]
    model.task = "classify"
    model.device = torch.device("cpu")
    return model


_MODELS = {}


def _model(name, size):
    key = (name, size)
    if key not in _MODELS:
        _MODELS[key] = _built(name, size) if (name, size) in BUILT else _bare(name, size)
    return _MODELS[key]


def _validator(model, **cfg):
    v = ClassifyValidator.__new__(ClassifyValidator)
    v.model = model
    cfg.setdefault("imgsz", model.input_size)
    v.config = ValidationConfig(data="x", **cfg)
    return v


ALL = BUILT + BARE


@pytest.mark.parametrize("name,size", ALL, ids=[n for n, _ in ALL])
def test_predict_and_val_produce_the_same_tensor(name, size):
    model = _model(name, size)
    predicted, *_ = model._preprocess(_IMAGE)
    validated = _validator(model)._dataset_transform()["transform"](_IMAGE)
    assert predicted.shape[0] == 1
    assert torch.equal(predicted[0], validated), name


@pytest.mark.parametrize("name,size", ALL, ids=[n for n, _ in ALL])
def test_calibration_uses_the_same_transform(name, size):
    """INT8 calibration unpacks ``(array, ratio)`` and sees the same pixels."""
    model = _model(name, size)
    if name == "LibreVJEPA2":
        pytest.skip("V-JEPA 2 calibration works on clips, not still images")
    array, ratio = model._get_preprocess_numpy()(np.asarray(_IMAGE), model.input_size)
    assert ratio == 1.0
    np.testing.assert_array_equal(array, model.eval_transform()(_IMAGE).numpy())


def test_the_val_dataset_applies_the_models_transform(tmp_path):
    from libreyolo.data.classify_dataset import ClassifyDataset

    for split in ("train", "val"):
        (tmp_path / split / "a").mkdir(parents=True)
        _IMAGE.save(tmp_path / split / "a" / "x.png")
    model = _model("LibreResNet", "18")
    dataset = ClassifyDataset(
        tmp_path, "val", imgsz=224, augment=False,
        **_validator(model)._dataset_transform(),
    )
    image, label = dataset[0]
    assert label == 0
    assert torch.equal(image, model._preprocess(_IMAGE)[0][0])


def test_explicit_transform_is_evaluation_only(tmp_path):
    from libreyolo.data.classify_dataset import ClassifyDataset

    (tmp_path / "train" / "a").mkdir(parents=True)
    with pytest.raises(ValueError, match="evaluation only"):
        ClassifyDataset(tmp_path, "train", 224, augment=True, transform=lambda x: x)


class TestOverrides:
    def test_crop_pct_override_reaches_val_only_when_asked(self):
        model = _model("LibreResNet", "18")
        default = _validator(model)._dataset_transform()["transform"](_IMAGE)
        override = _validator(model, crop_pct=0.5)._dataset_transform()["transform"](_IMAGE)
        assert torch.equal(default, model.eval_transform()(_IMAGE))
        assert torch.equal(override, model.eval_transform(crop_pct=0.5)(_IMAGE))
        assert not torch.equal(default, override)

    def test_siglip_crop_override_leaves_square_resize(self):
        model = _model("LibreSigLIP2", "b16")
        square = model.eval_transform()
        cropped = model.eval_transform(crop_pct=0.9)
        assert "CenterCrop" not in repr(square)
        assert "CenterCrop" in repr(cropped)

    @pytest.mark.parametrize("name,size", [("LibreVGG", "16"), ("LibreDeiT", "t"), ("LibreSwin", "t")])
    def test_fixed_resolution_families_reject_other_sizes_in_val_too(self, name, size):
        with pytest.raises(ValueError):
            _validator(_model(name, size), imgsz=288)._dataset_transform()

    def test_vjepa2_rejects_settings_it_cannot_honor(self):
        model = _model("LibreVJEPA2", "l256")
        with pytest.raises(ValueError, match="fixed crop"):
            model.eval_transform(320)
        with pytest.raises(ValueError, match="crop_pct"):
            model.eval_transform(crop_pct=0.9)


class TestFormerMismatches:
    """The pipelines #886 found scored on different preprocessing than predict."""

    def test_pe_validates_with_its_own_normalization_and_square_resize(self):
        transform = repr(_model("LibrePE", "b16").eval_transform())
        assert "CenterCrop" not in transform
        assert "mean=(0.5, 0.5, 0.5)" in transform

    def test_vjepa2_val_runs_the_predict_frame_code(self):
        from libreyolo.models.vjepa2.preprocess import preprocess_frames

        model = _model("LibreVJEPA2", "l256")
        expected = preprocess_frames([np.asarray(_IMAGE)], 256)[0, 0]
        assert torch.equal(model.eval_transform()(_IMAGE), expected)


@pytest.mark.parametrize(
    "name,size", [("LibreViT", "ti"), ("LibreCLIP", "b32"), ("LibreSigLIP2", "b16"), ("LibrePE", "b16"), ("LibreResNet", "18")]
)
def test_exported_backend_matches_native(name, size):
    """Backend predict and val share the native transform, from export metadata."""
    from libreyolo.backends.base import BaseBackend

    class _Backend(BaseBackend):
        def _run_inference(self, blob):  # pragma: no cover - never called
            raise NotImplementedError

    model = _model(name, size)
    backend = _Backend.__new__(_Backend)
    backend.model_family = model.FAMILY
    backend.imgsz = model.input_size
    # What the exporter records (crop_pct / interpolation) for this model.
    backend.crop_pct = model.crop_pct
    backend.interpolation = model.interpolation
    native = model.eval_transform()(_IMAGE)
    assert torch.equal(backend.eval_transform()(_IMAGE), native)
    tensor, *_ = backend._preprocess_classify(_IMAGE, model.input_size, "auto")
    assert torch.equal(tensor[0], native)
    validated = _validator(SimpleNamespace(eval_transform=backend.eval_transform, input_size=model.input_size))
    assert torch.equal(validated._dataset_transform()["transform"](_IMAGE), native)


def test_backend_imports_an_unregistered_family(monkeypatch):
    """A backend loaded before the native package still finds its family."""
    import importlib

    from libreyolo.backends.base import BaseBackend
    from libreyolo.models.base.model import BaseModel

    class _Backend(BaseBackend):
        def _run_inference(self, blob):  # pragma: no cover - never called
            raise NotImplementedError

    pe_cls = libreyolo.LibrePE
    registry = [c for c in BaseModel._registry if c is not pe_cls]
    monkeypatch.setattr(BaseModel, "_registry", registry)
    imported = []
    real_import = importlib.import_module

    def _import(name):
        imported.append(name)
        registry.append(pe_cls)  # what registering on import does
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", _import)
    backend = _Backend.__new__(_Backend)
    backend.model_family = "pe"
    assert backend._family_class() is pe_cls
    assert imported == ["libreyolo.models.pe"]
