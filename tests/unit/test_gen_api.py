"""Offline unit tests for the LibreGen tier (no diffusers, no weights, no network).

Covers the ``generate`` task registration, the ``GeneratedImage`` payload and
its ``Results`` integration, the factory, the shared call contract (prompts,
``imgsz``, ``seed``, ``n``, family knobs, ``save``, ``stream``) through a fake
family, and the diffusers adapter through a fake pipeline class.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import GeneratedImage, LibreFlux2Klein, LibreGen, LibreSanaSprint, Results
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.base import LibreGenModel
from libreyolo.models.gen.diffusers_family import DiffusersGenFamily
from libreyolo.tasks import TASKS, normalize_task, suffix_to_task, task_to_suffix

pytestmark = [pytest.mark.unit, pytest.mark.gen]


class FakeGen(LibreGenModel):
    """Paints each image with its seed so tests can read the seed back."""

    FAMILY = "fake_gen"
    FILENAME_PREFIX = "LibreFakeGen"
    INPUT_SIZES = {"s": 64}
    SIZE_MULTIPLE = 16
    KNOBS = ("steps", "guidance")
    DEFAULTS = {"steps": 4}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def _ensure_weights(self) -> str:
        return "unused"

    def _load(self, snapshot_dir: str) -> None:
        self._pipeline = object()

    def _generate(self, prompt, height, width, seeds, knobs):
        self.calls.append((prompt, height, width, list(seeds), dict(knobs)))
        return [
            np.full((height, width, 3), seed % 256, dtype=np.uint8) for seed in seeds
        ]


@pytest.fixture
def model():
    return FakeGen("s", device="cpu")


def _posix(path) -> str:
    """Compare saved paths the same way on Windows."""
    return Path(path).as_posix()


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------


def test_generate_task_is_canonical():
    assert "generate" in TASKS
    for alias in ("generate", "generation", "text-to-image", "txt2img", "t2i"):
        assert normalize_task(alias) == "generate"
    assert task_to_suffix("generate") == "generate"
    assert suffix_to_task("-generate") == "generate"


# ---------------------------------------------------------------------------
# GeneratedImage payload
# ---------------------------------------------------------------------------


def test_payload_shape_and_metadata():
    payload = GeneratedImage(
        np.zeros((8, 12, 3), dtype=np.uint8), prompt="a fox", seed=5, params={"steps": 4}
    )
    assert payload.orig_shape == (8, 12)
    assert payload.image.size == (12, 8)
    assert len(payload) == 1
    assert payload.to_dict() == {
        "name": "generated",
        "shape": [8, 12, 3],
        "prompt": "a fox",
        "seed": 5,
        "steps": 4,
    }
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        GeneratedImage(np.zeros((8, 12), dtype=np.uint8))


def test_payload_keeps_metadata_across_conversions():
    payload = GeneratedImage(
        torch.zeros((4, 4, 3), dtype=torch.uint8), prompt="a fox", seed=5
    )
    for converted in (payload.numpy(), payload.cpu(), payload[0]):
        assert (converted.prompt, converted.seed) == ("a fox", 5)
        assert converted.array.shape == (4, 4, 3)


def test_png_records_how_to_reproduce(tmp_path):
    payload = GeneratedImage(
        np.zeros((4, 4, 3), dtype=np.uint8), prompt="a fóx", seed=5, params={"steps": 4}
    )
    payload.save(tmp_path / "sub" / "a.png")
    recorded = json.loads(Image.open(tmp_path / "sub" / "a.png").info["libreyolo"])
    assert recorded["prompt"] == "a fóx" and recorded["seed"] == 5
    payload.save(tmp_path / "a.jpg")
    assert Image.open(tmp_path / "a.jpg").size == (4, 4)


def test_results_integration(tmp_path):
    payload = GeneratedImage(np.full((4, 6, 3), 9, dtype=np.uint8), prompt="p", seed=1)
    result = Results(None, (4, 6), generated=payload)
    assert len(result) == 1
    assert "generated=GeneratedImage" in repr(result)
    assert result.summary() == [payload.to_dict()]
    assert json.loads(result.to_json())[0]["seed"] == 1
    assert result.plot().size == (6, 4)
    assert result.cpu().generated.prompt == "p"
    assert result.save(str(tmp_path / "out.png")) == str(tmp_path / "out.png")
    assert Image.open(tmp_path / "out.png").size == (6, 4)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_aliases_resolve_without_loading():
    for alias, (family_cls, size) in _ALIASES.items():
        loaded = LibreGen(alias, device="cpu")
        assert type(loaded) is family_cls and loaded.size == size
        assert loaded._pipeline is None
    assert type(LibreGen(device="cpu")) is LibreFlux2Klein
    assert type(LibreGen("Sana_Sprint", device="cpu")) is LibreSanaSprint


def test_factory_rejects_unknown_names(tmp_path):
    with pytest.raises(ValueError, match="Known aliases"):
        LibreGen("nope")
    with pytest.raises(ValueError, match="not a model directory"):
        LibreGen(str(tmp_path))


def test_factory_loads_a_directory_by_pipeline_class(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaSprintPipeline"})
    )
    loaded = LibreGen(str(tmp_path), device="cpu")
    assert type(loaded) is LibreSanaSprint
    assert loaded._ensure_weights() == str(tmp_path)
    assert not LibreFlux2Klein.matches_directory(tmp_path)


def test_alias_wins_over_a_same_named_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sana-sprint").mkdir()
    assert LibreGen("sana-sprint", device="cpu")._checkpoint_dir is None
    with pytest.raises(ValueError, match="not a model directory"):
        LibreGen("./sana-sprint", device="cpu")


def test_families_pin_a_commit_and_stay_out_of_the_checkpoint_factory():
    from libreyolo.models.base.model import BaseModel

    for family_cls in (LibreFlux2Klein, LibreSanaSprint):
        assert not issubclass(family_cls, BaseModel)
        assert family_cls.SUPPORTED_TASKS == ("generate",)
        for size, repo in family_cls.HF_REPOS.items():
            revision = family_cls.HF_REVISIONS[size]
            assert len(revision) == 40
            url = family_cls.get_download_url(f"{family_cls.FILENAME_PREFIX}{size}")
            assert url == f"https://huggingface.co/{repo}/tree/{revision}"


def test_constructor_validation():
    with pytest.raises(ValueError, match="Invalid size"):
        FakeGen("xl")
    with pytest.raises(ValueError, match="generate"):
        FakeGen("s", task="detect")
    with pytest.raises(ValueError, match="dtype"):
        FakeGen("s", dtype="int8")
    with pytest.raises(ValueError, match="offload"):
        FakeGen("s", device="cpu", offload=True)


# ---------------------------------------------------------------------------
# Call contract
# ---------------------------------------------------------------------------


def test_one_prompt_returns_one_result(model):
    result = model("a red fox", seed=3)
    assert isinstance(result, Results)
    assert result.boxes is None and result.orig_shape == (64, 64)
    generated = result.generated
    assert (generated.prompt, generated.seed) == ("a red fox", 3)
    assert generated.params == {
        "model": "fake-gen-s",
        "imgsz": [64, 64],
        "steps": 4,
    }
    assert result.speed["inference"] >= 0
    assert model.predict.__func__ is model.generate.__func__


def test_n_and_prompt_lists_return_flat_lists(model):
    results = model(["a", "b"], seed=10, n=2)
    assert [(r.generated.prompt, r.generated.seed) for r in results] == [
        ("a", 10),
        ("a", 11),
        ("b", 10),
        ("b", 11),
    ]
    assert isinstance(model("a", n=2), list)
    assert isinstance(model(["a"]), list)


def test_seed_reproduces_and_no_seed_is_recorded(model):
    first = model("a", seed=7).generated.array
    assert np.array_equal(first, model("a", seed=7).generated.array)
    # The same prompt and seed give the same image wherever it sits in a list.
    assert np.array_equal(first, model(["b", "a"], seed=7)[1].generated.array)
    drawn = model("a").generated.seed
    assert isinstance(drawn, int) and 0 <= drawn < 2**32
    assert model("a", seed=2**32 - 1, n=2)[1].generated.seed == 0


def test_imgsz_forms(model, caplog):
    assert model("a", imgsz=32).orig_shape == (32, 32)
    assert model("a", imgsz=(32, 48)).orig_shape == (32, 48)
    with caplog.at_level("WARNING"):
        assert model("a", imgsz=(30, 40)).orig_shape == (32, 48)
    assert "multiple of 16" in caplog.text
    for bad in ("big", (32,), True, 0, (32, -1), 3.5):
        with pytest.raises((TypeError, ValueError)):
            model("a", imgsz=bad)


def test_numpy_integers_are_accepted(model):
    results = model("a", imgsz=np.int64(32), seed=np.int64(3), n=np.int64(2))
    assert [r.generated.seed for r in results] == [3, 4]
    assert results[0].orig_shape == (32, 32)
    assert type(results[0].generated.seed) is int
    json.dumps(results[0].summary())


def test_unavailable_device_fails_at_construction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    for device in ("cuda", "cuda:1", 0, "0", "mps"):
        with pytest.raises(ValueError, match="not available"):
            FakeGen("s", device=device)
    with pytest.raises(ValueError, match="not available"):
        FakeGen("s", device="cpu").to("cuda")


def test_family_knobs(model):
    model("a", steps=9, guidance=2.5)
    assert model.calls[-1][4] == {"steps": 9, "guidance": 2.5}
    model("a", steps=None)
    assert model.calls[-1][4] == {"steps": 4}
    with pytest.raises(TypeError, match="does not accept 'negative'") as excinfo:
        model("a", negative="blurry")
    assert "Supported: imgsz, seed, n, steps, guidance." in str(excinfo.value)
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"prompt": None}, ValueError),
        ({"prompt": ""}, ValueError),
        ({"prompt": []}, ValueError),
        ({"prompt": ["a", 3]}, ValueError),
        ({"prompt": 3}, TypeError),
        ({"prompt": "a", "n": 0}, ValueError),
        ({"prompt": "a", "seed": -1}, ValueError),
        ({"prompt": "a", "seed": 1.5}, ValueError),
        ({"prompt": "a", "output_path": "x.png"}, ValueError),
        ({"prompt": "a", "n": 2, "save": True, "output_path": "x.png"}, ValueError),
        ({"prompt": "a", "save": True, "output_path": "noext"}, ValueError),
        ({"prompt": "a", "source": "photo.jpg"}, NotImplementedError),
    ],
)
def test_bad_calls_fail_before_generating(model, kwargs, error):
    with pytest.raises(error):
        model.generate(**kwargs)
    assert model.calls == []


def test_stream_is_lazy(model):
    stream = model(["a", "b"], seed=1, stream=True)
    assert model.calls == []
    assert next(stream).generated.prompt == "a"
    assert len(model.calls) == 1
    assert [r.generated.prompt for r in stream] == ["b"]


def test_concurrent_first_calls_load_once(model, monkeypatch):
    import threading
    import time

    loads = []

    def slow_load(snapshot_dir):
        loads.append(snapshot_dir)
        time.sleep(0.05)
        model._pipeline = object()

    monkeypatch.setattr(model, "_load", slow_load)
    threads = [threading.Thread(target=model, args=("a",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(loads) == 1
    assert len(model.calls) == 4


def test_save_writes_one_run_directory_per_call(model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = model(["A red fox!", "a blue jay"], seed=1, save=True)
    assert [_posix(r.saved_path) for r in results] == [
        "runs/generate/predict/000_a-red-fox_1.png",
        "runs/generate/predict/001_a-blue-jay_1.png",
    ]
    again = model("a", seed=1, save=True).saved_path
    assert _posix(again).startswith("runs/generate/predict2/")
    result = model("a", seed=1, save=True, output_path="out/mine.png")
    assert _posix(result.saved_path) == "out/mine.png"
    assert Image.open("out/mine.png").size == (64, 64)


def test_interleaved_streams_keep_their_own_run_directory(model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = model(["a", "b"], seed=1, stream=True, save=True)
    second = model(["a"], seed=1, stream=True, save=True, steps=9)
    paths = [
        _posix(next(stream).saved_path) for stream in (first, second, first)
    ]
    assert paths == [
        "runs/generate/predict/000_a_1.png",
        "runs/generate/predict2/000_a_1.png",
        "runs/generate/predict/001_b_1.png",
    ]


def test_auto_dtype_follows_the_device(model, monkeypatch):
    import platform

    assert model._resolve_dtype() is torch.float32
    model.device = torch.device("mps")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64"))
    assert model._resolve_dtype() is torch.bfloat16
    # MPS has no bfloat16 before macOS 14.
    monkeypatch.setattr(platform, "mac_ver", lambda: ("13.6", ("", "", ""), "arm64"))
    assert model._resolve_dtype() is torch.float32
    model.dtype = "fp16"
    assert model._resolve_dtype() is torch.float16


def test_unsupported_surfaces_say_why(model):
    with pytest.raises(NotImplementedError, match="LibreGen\\(path\\)"):
        model.train(data="folder")
    for call in (model.val, model.export, model.track, model.benchmark):
        with pytest.raises(NotImplementedError):
            call()


# ---------------------------------------------------------------------------
# diffusers adapter
# ---------------------------------------------------------------------------


class FakePipeline:
    loaded = []

    def __init__(self):
        self.calls = []
        self.device = None
        self.offloaded = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        pipeline = cls()
        cls.loaded.append((path, kwargs))
        return pipeline

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self, device=None):
        self.offloaded = device

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = (kwargs["width"], kwargs["height"])
        return SimpleNamespace(
            images=[Image.new("RGBA", size) for _ in kwargs["generator"]]
        )


@pytest.fixture
def diffusers_model(tmp_path, monkeypatch):
    FakePipeline.loaded = []
    monkeypatch.setattr(
        DiffusersGenFamily,
        "_require_pipeline_class",
        classmethod(lambda cls: FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "diffusers", SimpleNamespace(__version__="0.41.0")
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"})
    )
    return LibreGen(str(tmp_path), device="cpu")


def test_diffusers_adapter_maps_the_call(diffusers_model, tmp_path):
    results = diffusers_model("a fox", imgsz=(32, 48), seed=4, n=2, steps=6)
    path, load_kwargs = FakePipeline.loaded[0]
    assert path == str(tmp_path)
    assert load_kwargs == {"dtype": torch.float32}
    pipeline = diffusers_model.pipeline
    assert pipeline.device == torch.device("cpu")
    call = pipeline.calls[0]
    generators = call.pop("generator")
    assert call == {
        "prompt": "a fox",
        "height": 32,
        "width": 48,
        "num_images_per_prompt": 2,
        "num_inference_steps": 6,
        "guidance_scale": 1.0,
    }
    assert [g.device.type for g in generators] == ["cpu", "cpu"]
    assert [g.initial_seed() for g in generators] == [4, 5]
    # RGBA pipeline output still lands as (H, W, 3) uint8 RGB.
    assert results[0].generated.array.shape == (32, 48, 3)
    assert results[0].generated.params["model"] == str(tmp_path)


def test_distilled_flux2_klein_refuses_guidance(diffusers_model):
    diffusers_model("a fox", imgsz=32, guidance=1)
    diffusers_model.pipeline.config = SimpleNamespace(is_distilled=True)
    diffusers_model("a fox", imgsz=32)
    with pytest.raises(ValueError, match="step-distilled"):
        diffusers_model("a fox", imgsz=32, guidance=4.0)
    assert len(diffusers_model.pipeline.calls) == 2
    # A non-distilled directory of the same pipeline class keeps the knob.
    diffusers_model.pipeline.config = SimpleNamespace(is_distilled=False)
    diffusers_model("a fox", imgsz=32, guidance=4.0)
    assert diffusers_model.pipeline.calls[-1]["guidance_scale"] == 4.0


def test_sana_sprint_step_counts_other_than_two(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DiffusersGenFamily,
        "_require_pipeline_class",
        classmethod(lambda cls: FakePipeline),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "diffusers", SimpleNamespace(__version__="0.41.0")
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaSprintPipeline"})
    )
    sana = LibreGen(str(tmp_path), device="cpu")
    sana("a fox", imgsz=32)
    assert "intermediate_timesteps" not in sana.pipeline.calls[-1]
    for steps in (1, 4):
        sana("a fox", imgsz=32, steps=steps)
        assert sana.pipeline.calls[-1]["intermediate_timesteps"] is None
        assert sana.pipeline.calls[-1]["num_inference_steps"] == steps


def test_to_moves_the_auto_dtype_with_the_device(diffusers_model, monkeypatch):
    diffusers_model("a fox", imgsz=32)
    moves = []
    pipeline = diffusers_model.pipeline
    monkeypatch.setattr(pipeline, "to", lambda *args: moves.append(args))
    monkeypatch.setattr(
        type(diffusers_model), "_resolve_device", staticmethod(torch.device)
    )
    # The MPS auto dtype depends on the macOS release, so pin one.
    monkeypatch.setattr(
        "platform.mac_ver", lambda: ("15.1", ("", "", ""), "arm64")
    )
    diffusers_model.to("mps")
    assert moves == [(torch.device("mps"), torch.bfloat16)]


def test_diffusers_adapter_uses_the_old_dtype_keyword(diffusers_model, monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "diffusers", SimpleNamespace(__version__="0.38.0")
    )
    diffusers_model("a fox", imgsz=32)
    assert FakePipeline.loaded[0][1] == {"torch_dtype": torch.float32}


def test_missing_diffusers_names_the_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "diffusers", None)
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2KleinPipeline"})
    )
    with pytest.raises(ImportError, match=r"libreyolo\[gen\]"):
        LibreGen(str(tmp_path), device="cpu")("a fox")


def test_old_diffusers_names_the_missing_pipeline(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("sys").modules, "diffusers", SimpleNamespace(__version__="0.30.0")
    )
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "SanaSprintPipeline"})
    )
    with pytest.raises(ImportError, match="SanaSprintPipeline is missing"):
        LibreGen(str(tmp_path), device="cpu")("a fox")


def test_snapshot_download_is_pinned_and_marked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloads = []

    def fake_snapshot_download(repo, local_dir, **kwargs):
        downloads.append((repo, kwargs))
        target = tmp_path / local_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_index.json").write_text(
            json.dumps({"_class_name": "SanaSprintPipeline"})
        )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    model = LibreSanaSprint("0.6b", device="cpu")
    # A missing runtime is reported before anything is downloaded.
    monkeypatch.setitem(__import__("sys").modules, "diffusers", None)
    with pytest.raises(ImportError, match=r"libreyolo\[gen\]"):
        model("a fox")
    assert downloads == []
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert _posix(model._ensure_weights()) == "weights/LibreSanaSprint0.6b"
    repo, kwargs = downloads[0]
    assert repo == LibreSanaSprint.HF_REPOS["0.6b"]
    assert kwargs["revision"] == LibreSanaSprint.HF_REVISIONS["0.6b"]
    assert "*/*.safetensors" in kwargs["allow_patterns"]
    assert "*.safetensors" not in kwargs["allow_patterns"]
    # A complete snapshot is not downloaded again.
    assert _posix(model._ensure_weights()) == "weights/LibreSanaSprint0.6b"
    assert len(downloads) == 1
