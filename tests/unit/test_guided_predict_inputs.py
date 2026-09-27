"""Unit coverage for family-declared guided prediction inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.base import BaseModel
from libreyolo.models.base import inference as inference_base
from libreyolo.models.base.inference import InferenceRunner
from libreyolo.utils.image_loader import ImageLoader
from libreyolo.utils.source import SourceKind, SourceSpec


pytestmark = pytest.mark.unit


class _GuidedModel:
    PREDICT_INPUT_KWARGS = ("mask", "trimap")
    REQUIRED_PREDICT_INPUT_KWARGS = ("mask",)
    task = "detect"
    TTA_ENABLED = False
    names = {0: "thing"}
    device = torch.device("cpu")

    def __init__(self):
        self.preprocess_inputs = None
        self.postprocess_kwargs = None

    def _get_input_size(self):
        return 32

    def _preprocess(self, image, color_format="auto", input_size=None):
        pil = ImageLoader.load(image, color_format=color_format)
        return torch.zeros(1, 3, 32, 32), pil, pil.size, 1.0

    def _preprocess_predict(
        self,
        image,
        color_format="auto",
        input_size=None,
        *,
        mask,
        trimap=None,
    ):
        self.preprocess_inputs = {
            "mask": mask,
            "trimap": trimap,
            "input_size": input_size,
        }
        return self._preprocess(image, color_format, input_size=input_size)

    def _forward(self, tensor):
        return tensor

    def _postprocess(
        self,
        output,
        conf,
        iou,
        original_size,
        max_det=300,
        ratio=1.0,
        classes=None,
        **kwargs,
    ):
        self.postprocess_kwargs = kwargs
        return {
            "boxes": [[1.0, 1.0, 5.0, 5.0]],
            "scores": [0.9],
            "classes": [0],
            "num_detections": 1,
        }


class _PlainModel(_GuidedModel):
    PREDICT_INPUT_KWARGS = ()
    REQUIRED_PREDICT_INPUT_KWARGS = ()

    def _preprocess_predict(self, image, color_format="auto", input_size=None):
        return self._preprocess(image, color_format, input_size=input_size)


def _image():
    return np.zeros((12, 16, 3), dtype=np.uint8)


def test_base_predict_preprocess_hook_delegates_to_existing_preprocess():
    calls = []

    class Probe:
        def _preprocess(self, image, color_format="auto", input_size=None):
            calls.append((image, color_format, input_size))
            return "delegated"

    probe = Probe()
    image = object()

    assert (
        BaseModel._preprocess_predict(probe, image, color_format="rgb", input_size=48)
        == "delegated"
    )
    assert calls == [(image, "rgb", 48)]


def test_single_image_forwards_only_declared_inputs_to_preprocessing():
    model = _GuidedModel()
    mask = np.ones((12, 16), dtype=np.uint8)
    trimap = np.full((12, 16), 128, dtype=np.uint8)

    result = InferenceRunner(model)(_image(), mask=mask, trimap=trimap, num_select=7)

    assert result.orig_shape == (12, 16)
    assert model.preprocess_inputs["mask"] is mask
    assert model.preprocess_inputs["trimap"] is trimap
    assert model.preprocess_inputs["input_size"] == 32
    assert model.postprocess_kwargs["num_select"] == 7
    assert "mask" not in model.postprocess_kwargs
    assert "trimap" not in model.postprocess_kwargs


@pytest.mark.parametrize(
    "value", [pytest.param(None, id="none"), pytest.param("missing", id="omitted")]
)
def test_required_predict_input_is_validated(value):
    kwargs = {} if value == "missing" else {"mask": value}

    with pytest.raises(
        ValueError, match=r"requires prediction input option\(s\): mask"
    ):
        InferenceRunner(_GuidedModel())(_image(), **kwargs)


@pytest.mark.parametrize("key", ["mask", "trimap"])
def test_ordinary_models_reject_undeclared_guided_inputs(key):
    with pytest.raises(TypeError, match=key):
        InferenceRunner(_PlainModel())(_image(), **{key: _image()})


@pytest.mark.parametrize(
    "source_spec,predict_options,error_text",
    [
        (SourceSpec(SourceKind.VIDEO, "clip.mp4"), {}, "video"),
        (
            SourceSpec(SourceKind.STREAM, 0, (0,)),
            {"stream": True},
            "live source",
        ),
        (SourceSpec(SourceKind.SCREEN, "screen"), {}, "screen capture"),
        (SourceSpec(SourceKind.DIRECTORY, "images"), {}, "directory source"),
        (
            SourceSpec(SourceKind.IMAGE_BATCH, [_image()], (_image(),)),
            {},
            "image batch",
        ),
        (SourceSpec(SourceKind.IMAGE, "image.png"), {"stream": True}, "stream=True"),
        (SourceSpec(SourceKind.IMAGE, "image.png"), {"tiling": True}, "tiling=True"),
        (SourceSpec(SourceKind.IMAGE, "image.png"), {"augment": True}, "augment=True"),
    ],
)
def test_guided_inputs_reject_ambiguous_source_modes(
    monkeypatch, source_spec, predict_options, error_text
):
    monkeypatch.setattr(inference_base, "classify_source", lambda source: source_spec)

    with pytest.raises(ValueError, match=error_text):
        InferenceRunner(_GuidedModel())(
            _image(), mask=np.ones((12, 16), dtype=np.uint8), **predict_options
        )


def test_flagship_detection_families_keep_empty_input_contracts():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.yolo9.model import LibreYOLO9

    for model_class in (LibreYOLO9, LibreRFDETR):
        assert model_class.PREDICT_INPUT_KWARGS == ()
        assert model_class.REQUIRED_PREDICT_INPUT_KWARGS == ()

        uninitialized = object.__new__(model_class)
        with pytest.raises(TypeError, match="mask"):
            InferenceRunner(uninitialized)(_image(), mask=_image())
