"""Classification predict, val, calibration and exports share one eval pipeline (#886).

A classifier must be scored on the preprocessing it is deployed with. Each
family's eval pipeline is read from ``_get_eval_transform``; these tests pin,
for every registered classification family, that ``predict()``, ``val()``,
INT8 calibration and exported backends produce the same tensor.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import libreyolo
from libreyolo.validation.classify_validator import ClassifyValidator

pytestmark = pytest.mark.unit

# family -> (class name, size, build). "real" builds the family (cheap on CPU,
# so the test sees what __init__ sets); "bare" skips the large towers, whose
# eval settings are class attributes.
FAMILIES = {
    "alexnet": ("LibreAlexNet", "b", "real"),
    "vgg": ("LibreVGG", "16", "real"),
    "resnet": ("LibreResNet", "18", "real"),
    "efficientnetv2": ("LibreEfficientNetV2", "b1", "real"),
    "convnext": ("LibreConvNeXt", "t", "real"),
    "convnextv2": ("LibreConvNeXtV2", "atto", "real"),
    "mobilenetv4": ("LibreMobileNetV4", "m", "real"),
    "deit": ("LibreDeiT", "t", "real"),
    "swin": ("LibreSwin", "t", "real"),
    "vit": ("LibreViT", "ti", "real"),
    "clip": ("LibreCLIP", "b32", "bare"),
    "siglip2": ("LibreSigLIP2", "b16", "bare"),
    "pe": ("LibrePE", "b16", "bare"),
    "dinov2": ("LibreDINOv2", "s", "bare"),
    "vjepa2": ("LibreVJEPA2", "l256", "bare"),
}

# Odd, non-square size so resize, crop and squash all matter.
_IMAGE = Image.fromarray(
    np.random.default_rng(886).integers(0, 256, (293, 411, 3), dtype=np.uint8)
)
_MODELS: dict = {}


def _cls(family):
    from libreyolo.models import try_ensure_rfdetr

    try_ensure_rfdetr()  # DINOv2 registers lazily with RF-DETR
    name = FAMILIES[family][0]
    cls = getattr(libreyolo, name, None)
    if cls is None:
        from libreyolo.models.base.model import BaseModel

        cls = next(c for c in BaseModel._registry if c.__name__ == name)
    return cls


def _model(family):
    if family not in _MODELS:
        name, size, build = FAMILIES[family]
        cls = _cls(family)
        if build == "real":
            model = cls(size=size, nb_classes=3, device="cpu")
        else:
            model = cls.__new__(cls)
            model.size = size
            model.input_size = cls.INPUT_SIZES[size]
            model.task = "classify"
            model.device = torch.device("cpu")
        _MODELS[family] = model
    return _MODELS[family]


def _validator(model, validator_cls=None, crop_pct=None):
    validator_cls = validator_cls or getattr(model, "validator_class", None) or ClassifyValidator
    if isinstance(validator_cls, type) and not issubclass(validator_cls, ClassifyValidator):
        validator_cls = ClassifyValidator
    v = validator_cls.__new__(validator_cls)
    v.model = model
    v.config = SimpleNamespace(imgsz=model.input_size, crop_pct=crop_pct)
    return v


def _val_tensor(model, **kwargs):
    return _validator(model, **kwargs)._dataset_transform()["transform"](_IMAGE)


def test_every_registered_classifier_is_pinned():
    """A new classification family must be added here, so it is checked too."""
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.models.base.model import BaseModel

    try_ensure_rfdetr()
    registered = {
        c.FAMILY
        for c in BaseModel._registry
        if "classify" in getattr(c, "SUPPORTED_TASKS", ())
    }
    assert registered == set(FAMILIES)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_val_uses_the_predict_transform(family):
    model = _model(family)
    predicted = model._preprocess(_IMAGE)[0]
    assert predicted.shape[0] == 1
    assert torch.equal(_val_tensor(model), predicted[0].cpu())


@pytest.mark.parametrize("family", sorted(set(FAMILIES) - {"vjepa2"}))
def test_calibration_uses_the_eval_transform(family):
    """INT8 calibration unpacks ``(chw, ratio)`` and must see predict's pixels."""
    model = _model(family)
    array, ratio = model._get_preprocess_numpy()(np.asarray(_IMAGE), model.input_size)
    assert isinstance(array, np.ndarray) and array.dtype == np.float32
    assert ratio == 1.0
    np.testing.assert_array_equal(array, model._preprocess(_IMAGE)[0][0].numpy())


def test_calibration_loader_gets_batches(tmp_path):
    """The loader dropped every image when the family returned a bare tensor."""
    from libreyolo.export.calibration import get_calibration_dataloader

    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        _IMAGE.save(images / f"{i}.jpg")
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path}\ntrain: images\nval: images\nnames:\n  0: a\n"
    )
    model = _model("resnet")
    loader = get_calibration_dataloader(
        data=str(tmp_path / "data.yaml"),
        imgsz=224,
        batch=3,
        fraction=1.0,
        preprocess_fn=model._get_preprocess_numpy(),
    )
    batches = list(loader)
    assert [b.shape for b in batches] == [(3, 3, 224, 224)]


class TestOverrides:
    def test_crop_pct_override_reaches_val(self):
        model = _model("resnet")
        default = _val_tensor(model)
        override = _val_tensor(model, crop_pct=0.5)
        assert not torch.equal(default, override)
        assert torch.equal(override, model._get_eval_transform(crop_pct=0.5)(_IMAGE))

    @pytest.mark.parametrize("family", ["siglip2", "pe"])
    def test_crop_override_leaves_square_resize(self, family):
        from torchvision import transforms

        model = _model(family)
        square = model._get_eval_transform().transforms
        cropped = model._get_eval_transform(crop_pct=0.9).transforms
        assert not any(isinstance(op, transforms.CenterCrop) for op in square)
        assert isinstance(cropped[1], transforms.CenterCrop)

    def test_rectangular_imgsz_is_rejected_not_squared(self):
        model = _model("resnet")
        with pytest.raises(NotImplementedError, match="square imgsz"):
            model._get_eval_transform((224, 320))
        assert model._get_eval_transform((224, 224)) is not None

    def test_vjepa2_rejects_what_it_cannot_honor(self):
        model = _model("vjepa2")
        with pytest.raises(ValueError, match="fixed crop"):
            model._get_eval_transform(320)
        with pytest.raises(ValueError, match="crop_pct"):
            model._get_eval_transform(crop_pct=0.9)


class TestFormerMismatches:
    """Pipelines #886 found scored on different preprocessing than predict."""

    def test_pe_validates_with_its_square_resize_and_normalization(self):
        from torchvision import transforms

        ops = _model("pe")._get_eval_transform().transforms
        assert not any(isinstance(op, transforms.CenterCrop) for op in ops)
        assert tuple(ops[-1].mean) == (0.5, 0.5, 0.5)

    def test_vjepa2_validates_with_its_frame_preprocessing(self):
        from libreyolo.models.vjepa2.preprocess import preprocess_frames

        expected = preprocess_frames([np.asarray(_IMAGE)], 256)[0, 0]
        assert torch.equal(_val_tensor(_model("vjepa2")), expected)

    def test_dinov2_semantic_calibration_is_unchanged(self):
        model = _cls("dinov2").__new__(_cls("dinov2"))
        model.task = "semantic"
        array, ratio = model._get_preprocess_numpy()(np.asarray(_IMAGE), 64)
        assert array.shape == (3, 64, 64) and ratio == 1.0
        assert 0.0 <= float(array.min()) and float(array.max()) <= 1.0


# ---------------------------------------------------------------------------
# Exported backends
# ---------------------------------------------------------------------------


class TestExportMetadata:
    def test_non_imagenet_families_record_their_pipeline(self):
        from libreyolo.export.exporter import _classify_eval_metadata

        meta = _classify_eval_metadata(_model("siglip2"))
        assert meta["crop_pct"] == 1.0
        assert meta["interpolation"] == "bilinear"
        assert json.loads(meta["norm_mean"]) == [0.5, 0.5, 0.5]
        assert meta["resize_mode"] == "stretch"

    def test_imagenet_families_keep_their_existing_keys(self):
        from libreyolo.export.exporter import _classify_eval_metadata

        assert set(_classify_eval_metadata(_model("resnet"))) == {"crop_pct", "interpolation"}

    def test_metadata_parses_from_strings_and_native_values(self):
        from libreyolo.backends.base import classify_eval_kwargs

        as_strings = classify_eval_kwargs(
            {"crop_pct": "0.9", "norm_mean": "[0.5, 0.5, 0.5]", "resize_mode": "stretch"}
        )
        native = classify_eval_kwargs({"crop_pct": 0.9, "norm_mean": [0.5, 0.5, 0.5]})
        assert as_strings["crop_pct"] == native["crop_pct"] == 0.9
        assert as_strings["norm_mean"] == native["norm_mean"] == (0.5, 0.5, 0.5)
        assert classify_eval_kwargs({}) == {
            "crop_pct": None,
            "interpolation": None,
            "norm_mean": None,
            "norm_std": None,
            "resize_mode": None,
        }


def _backend(family, **metadata_kwargs):
    from libreyolo.backends.base import BaseBackend, classify_eval_kwargs

    class _Backend(BaseBackend):
        def _run_inference(self, blob):  # pragma: no cover - never called
            raise NotImplementedError

    model = _model(family)
    return _Backend(
        model_path="x.onnx",
        nb_classes=3,
        device="cpu",
        imgsz=model.input_size,
        model_family=family,
        names={0: "a", 1: "b", 2: "c"},
        task="classify",
        supported_tasks=("classify",),
        default_task="classify",
        **classify_eval_kwargs(metadata_kwargs),
    )


@pytest.mark.parametrize("family", ["clip", "siglip2", "pe", "vit"])
def test_exports_without_the_new_keys_keep_their_family_pipeline(family):
    """Artifacts written before norm metadata existed still preprocess natively."""
    model = _model(family)
    legacy = {"crop_pct": model.crop_pct, "interpolation": model.interpolation} if family == "vit" else {}
    backend = _backend(family, **legacy)
    native = model._preprocess(_IMAGE)[0][0]
    assert torch.equal(backend._preprocess_classify(_IMAGE, model.input_size, "auto")[0][0], native)
    assert torch.equal(_val_tensor(backend), native)


@pytest.mark.parametrize("family", ["vit", "resnet"])
def test_onnx_export_predict_and_val_match_native(family, tmp_path):
    """Round trip through a real export: backend predict and val == native."""
    pytest.importorskip("onnxruntime")
    model = _model(family)
    path = model.export(format="onnx", output_path=str(tmp_path / f"{family}.onnx"))
    backend = libreyolo.LibreYOLO(path, device="cpu")
    native = model._preprocess(_IMAGE)[0][0]
    predicted = backend._preprocess_classify(_IMAGE, model.input_size, "auto")[0][0]
    assert torch.equal(predicted, native)
    assert torch.equal(_val_tensor(backend), native)


def test_runtime_metadata_filter_keeps_the_eval_pipeline():
    """TorchScript/TensorRT/OpenVINO/Paddle/NCNN/Triton read through this filter.

    Values differ from the legacy family table, so a dropped key cannot be
    masked by the fallback.
    """
    from libreyolo.backends.base import _read_runtime_metadata, classify_eval_kwargs

    exported = {
        "crop_pct": "0.8",
        "interpolation": "bicubic",
        "norm_mean": "[0.1, 0.2, 0.3]",
        "norm_std": "[0.4, 0.5, 0.6]",
        "resize_mode": "stretch",
    }
    kwargs = classify_eval_kwargs(_read_runtime_metadata(exported))
    assert kwargs == {
        "crop_pct": 0.8,
        "interpolation": "bicubic",
        "norm_mean": (0.1, 0.2, 0.3),
        "norm_std": (0.4, 0.5, 0.6),
        "resize_mode": "stretch",
    }
    backend = _backend("vit", **exported)
    assert backend.norm_mean == (0.1, 0.2, 0.3)
    assert backend.resize_mode == "stretch"
