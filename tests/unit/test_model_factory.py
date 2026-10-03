"""Unit tests for model factory heuristics."""

import torch
import pytest

from libreyolo import LibreYOLO
from libreyolo.models import _needs_rfdetr_registration
from libreyolo.models.base.model import BaseModel
from libreyolo.models.yolo9.model import LibreYOLO9
from libreyolo.models.yolo9.nn import LibreYOLO9Model
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


def test_rfdetr_lazy_registration_detects_enc_out_markers():
    weights_dict = {
        "transformer.enc_out_class_embed.0.weight": object(),
        "transformer.enc_out_bbox_embed.0.layers.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is True


def test_rfdetr_lazy_registration_detects_classifier_signature():
    weights_dict = {
        "backbone.encoder.encoder.embeddings.position_embeddings": object(),
        "linear.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is True


def test_rfdetr_lazy_registration_ignores_rtdetr_signature():
    weights_dict = {
        "backbone.stages.0.conv.weight": object(),
        "encoder.input_proj.0.0.weight": object(),
        "decoder.input_proj.0.conv.weight": object(),
        "decoder.dec_score_head.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is False


@pytest.mark.parametrize("device_arg", ["0", 0])
def test_base_model_normalizes_bare_numeric_device(device_arg):
    class _NoopModule(torch.nn.Module):
        def to(self, device):
            self.moved_to = device
            return self

    class _DeviceDummy(BaseModel):
        FAMILY = "_device_dummy"
        FILENAME_PREFIX = "_DeviceDummy"
        INPUT_SIZES = {"n": 32}

        def _init_model(self):
            self.inner = _NoopModule()
            return self.inner

        def _get_available_layers(self):
            return {}

        @staticmethod
        def _get_preprocess_numpy():
            return None

        def _preprocess(self, image, color_format="auto", input_size=None):
            raise NotImplementedError

        def _forward(self, input_tensor):
            raise NotImplementedError

        def _postprocess(
            self,
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=300,
            ratio=1.0,
            **kwargs,
        ):
            raise NotImplementedError

    try:
        model = _DeviceDummy(None, size="n", device=device_arg)

        assert str(model.device) == "cuda:0"
        assert model.inner.moved_to == torch.device("cuda:0")
    finally:
        if _DeviceDummy in BaseModel._registry:
            BaseModel._registry.remove(_DeviceDummy)


def test_factory_loads_yolo9_t_metadata_checkpoint_with_coco_class_width(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=80)

    # Mimic a fine-tuned checkpoint saved from a COCO-width YOLO9-t model:
    # only the final class conv is rebuilt to 2 classes, while the class
    # branch hidden width stays at 80.
    for seq in model.head.class_convs:
        in_channels = seq[-1].weight.shape[1]
        seq[-1] = torch.nn.Conv2d(in_channels, 2, 1)

    ckpt_path = tmp_path / "yolo9_t_best.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            model.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=2,
            names={0: "red", 1: "white"},
            imgsz=640,
        ),
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.nb_classes == 2
    assert loaded.names == {0: "red", 1: "white"}
    assert loaded.model.head.class_convs[0][0].conv.weight.shape[0] == 80


def test_factory_warns_for_legacy_libreyolo_metadata_checkpoint(tmp_path, caplog):
    model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "nc": 80,
            "names": {i: f"class_{i}" for i in range(80)},
            "model_family": "yolo9",
            "size": "t",
        },
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.nb_classes == 80
    assert "legacy compatibility path" in caplog.text


def test_factory_autoconverts_partial_metadata_upstream_yolo9(
    tmp_path, monkeypatch
):
    upstream_path = tmp_path / "v9-t-custom.pt"
    converted_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        {
            "model": {
                "0.conv.weight": torch.zeros(16, 3, 3, 3),
                "0.bn.weight": torch.zeros(16),
                "22.heads.0.class_conv.2.weight": torch.zeros(3, 16, 1, 1),
                "22.heads.0.class_conv.2.bias": torch.zeros(3),
            },
            "names": ["bolt", "nut", "washer"],
        },
        upstream_path,
    )

    converted = LibreYOLO9Model(config="t", nb_classes=3)
    torch.save(
        wrap_libreyolo_checkpoint(
            converted.state_dict(),
            model_family="yolo9",
            size="t",
            task="detect",
            nc=3,
            names={0: "bolt", 1: "nut", 2: "washer"},
            imgsz=640,
        ),
        converted_path,
    )

    import libreyolo.models.autoconvert as autoconvert_module

    calls = {}

    def fake_autoconvert(model_path, *, loaded=None):
        calls["model_path"] = model_path
        calls["loaded"] = loaded
        return str(converted_path)

    monkeypatch.setattr(
        autoconvert_module, "autoconvert_upstream_checkpoint", fake_autoconvert
    )

    loaded = LibreYOLO(str(upstream_path), device="cpu")

    assert calls["model_path"] == str(upstream_path)
    assert calls["loaded"]["names"] == ["bolt", "nut", "washer"]
    assert loaded.FAMILY == "yolo9"
    assert loaded.nb_classes == 3
    assert loaded.names == {0: "bolt", 1: "nut", 2: "washer"}


def test_factory_reloads_same_path_autoconverted_checkpoint(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        {
            "model": {
                "0.conv.weight": torch.zeros(16, 3, 3, 3),
                "0.bn.weight": torch.zeros(16),
                "22.heads.0.class_conv.2.weight": torch.zeros(2, 16, 1, 1),
                "22.heads.0.class_conv.2.bias": torch.zeros(2),
            },
            "names": ["bolt", "nut"],
        },
        ckpt_path,
    )

    converted = LibreYOLO9Model(config="t", nb_classes=2)
    converted_checkpoint = wrap_libreyolo_checkpoint(
        converted.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "bolt", 1: "nut"},
        imgsz=640,
    )

    import libreyolo.models.autoconvert as autoconvert_module

    calls = 0

    def fake_autoconvert(model_path, *, loaded=None):
        nonlocal calls
        calls += 1
        torch.save(converted_checkpoint, model_path)
        return model_path

    monkeypatch.setattr(
        autoconvert_module, "autoconvert_upstream_checkpoint", fake_autoconvert
    )

    loaded = LibreYOLO(str(ckpt_path), device="cpu")

    assert calls == 1
    assert loaded.FAMILY == "yolo9"
    assert loaded.nb_classes == 2
    assert loaded.names == {0: "bolt", 1: "nut"}


def test_factory_warns_for_foreign_metadata_less_checkpoint(tmp_path, caplog):
    model = LibreYOLO9Model(config="t", nb_classes=80)
    ckpt_path = tmp_path / "upstream_yolo9.pt"
    torch.save(model.state_dict(), ckpt_path)

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.FAMILY == "yolo9"
    assert "LibreYOLO metadata was not found" in caplog.text


def test_factory_rejects_unsupported_explicit_task_from_filename():
    with pytest.raises(ValueError, match="not supported"):
        LibreYOLO("LibreYOLOXs.pt", task="segment")


def test_factory_routes_onnx_before_native_task_validation(monkeypatch, tmp_path):
    import libreyolo.backends.onnx as onnx_backend

    class _DetectOnlyMatcher:
        DEFAULT_TASK = "detect"
        SUPPORTED_TASKS = ("detect",)

        @classmethod
        def detect_size_from_filename(cls, filename):
            return "n"

    captured = {}

    class _FakeOnnxBackend:
        def __init__(self, onnx_path, nb_classes=80, device="auto", task=None):
            captured.update(
                {
                    "onnx_path": onnx_path,
                    "nb_classes": nb_classes,
                    "device": device,
                    "task": task,
                }
            )

    monkeypatch.setattr(BaseModel, "_registry", [_DetectOnlyMatcher])
    monkeypatch.setattr(onnx_backend, "OnnxBackend", _FakeOnnxBackend)
    onnx_path = tmp_path / "rfdetr-l-segment.onnx"
    onnx_path.write_bytes(b"not a real onnx model")

    loaded = LibreYOLO(str(onnx_path), task="segment", device="cpu")

    assert isinstance(loaded, _FakeOnnxBackend)
    assert captured == {
        "onnx_path": str(onnx_path),
        "nb_classes": 80,
        "device": "cpu",
        "task": "segment",
    }
