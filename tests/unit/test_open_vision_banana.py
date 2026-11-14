"""Offline tests for the Open Vision Banana shortcut (fake pipeline, no weights)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreGen, LibreOpenVisionBanana
from libreyolo.models.gen.diffusers_family import DiffusersGenFamily

pytestmark = [pytest.mark.unit, pytest.mark.gen]


class FakePipeline:
    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "image" in kwargs:
            size = kwargs["image"].size
        else:
            size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(images=[Image.new("RGB", size, (7, 8, 9))])


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DiffusersGenFamily,
        "_require_pipeline_class",
        classmethod(lambda cls: FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "diffusers", SimpleNamespace(__version__="0.39.0")
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"})
    )
    return LibreGen("open-vision-banana", checkpoint_dir=str(tmp_path), device="cpu")


def test_aliases_and_mirrors():
    assert LibreGen("open-vision-banana", device="cpu").size == "4b"
    assert LibreGen("open-vision-banana-4b", device="cpu").size == "4b"
    assert LibreOpenVisionBanana.HF_REPOS == {"4b": "LibreYOLO/LibreOpenVisionBanana4b"}
    for size, repo in LibreOpenVisionBanana.HF_REPOS.items():
        revision = LibreOpenVisionBanana.HF_REVISIONS[size]
        assert len(revision) == 40
        url = LibreOpenVisionBanana.get_download_url(f"LibreOpenVisionBanana{size}")
        assert url == f"https://huggingface.co/{repo}/tree/{revision}"


def test_source_goes_to_the_pipeline_with_paper_settings(model, tmp_path):
    photo = tmp_path / "photo.png"
    Image.new("RGB", (40, 24), (1, 2, 3)).save(photo)
    result = model("Estimate depth.", source=str(photo), seed=3)
    call = model.pipeline.calls[0]
    generator = call.pop("generator")
    image = call.pop("image")
    assert call == {
        "prompt": "Estimate depth.",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
    }
    assert image.size == (40, 24) and image.mode == "RGB"
    assert generator.device.type == "cpu" and generator.initial_seed() == 3
    assert result.generated.array.shape == (24, 40, 3)
    assert result.generated.seed == 3
    assert result.generated.params["steps"] == 28


def test_source_accepts_bgr_arrays_and_knobs(model):
    bgr = np.zeros((16, 32, 3), dtype=np.uint8)
    model("Segment.", source=bgr, seed=0, steps=4, guidance=2.0)
    call = model.pipeline.calls[0]
    assert call["image"].size == (32, 16)
    assert (call["num_inference_steps"], call["guidance_scale"]) == (4, 2.0)


def test_source_save_writes_the_output(model, tmp_path):
    out = tmp_path / "mask.png"
    model(
        "Segment.",
        source=Image.new("RGB", (16, 16)),
        seed=0,
        save=True,
        output_path=str(out),
    )
    assert Image.open(out).getpixel((0, 0)) == (7, 8, 9)


def test_text_to_image_still_works(model):
    result = model("a lighthouse", imgsz=32, seed=1)
    assert "image" not in model.pipeline.calls[0]
    assert result.generated.array.shape == (32, 32, 3)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"prompt": None}, ValueError),
        ({"prompt": "  "}, ValueError),
        ({"prompt": "x", "output_path": "a.png"}, ValueError),
        ({"prompt": "x", "steps": 4, "ensemble": 2}, TypeError),
    ],
)
def test_bad_source_calls_fail_before_loading(model, kwargs, error):
    with pytest.raises(error):
        model(source=Image.new("RGB", (8, 8)), **kwargs)
    assert model._pipeline is None
