"""Offline unit tests for Speridlabs Iris-Z: no weights, no network.

A tiny Iris-Z snapshot is written to disk in the upstream layout (sharded
safetensors, index, empty prompt, config) so the real loader runs; the
transformer parameters, the FLUX.2 VAE and the text encoder are shrunk. The
reference values pin the upstream Iris-Z SDK's schedule, resolution picker
and sky decoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGen, LibreSperidlabsIrisZ, Results
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.speridlabs_irisz import nn as irisz_nn
from libreyolo.models.gen.speridlabs_irisz import pipeline as irisz_pipeline
from libreyolo.models.gen.speridlabs_irisz import text as irisz_text

pytestmark = [pytest.mark.unit, pytest.mark.gen]

CONTEXT_DIM = 48


@dataclass
class TinyParams:
    in_channels: int = 128
    context_in_dim: int = CONTEXT_DIM
    hidden_size: int = 64
    num_heads: int = 2
    depth: int = 1
    depth_single_blocks: int = 1
    axes_dim: list = field(default_factory=lambda: [8, 8, 8, 8])
    theta: int = 2000
    mlp_ratio: float = 2.0
    use_guidance_embed: bool = False


@dataclass
class TinyGuidedParams(TinyParams):
    use_guidance_embed: bool = True


def _tiny_vae():
    from diffusers.models import AutoencoderKLFlux2

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(32, 32, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        latent_channels=32,
        layers_per_block=1,
        norm_num_groups=32,
    )
    vae.bn.running_mean.normal_(0, 0.1)
    vae.bn.running_var.uniform_(0.5, 2.0)
    return vae.eval()


class StubTextEncoder(torch.nn.Module):
    """Deterministic per-prompt embeddings of the FLUX.2 shape."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.calls = []
        self.device_seen = []
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.to(device)

    def forward(self, prompts):
        self.calls.append(list(prompts))
        self.device_seen.append(self.anchor.device.type)
        rows = []
        for text in prompts:
            generator = torch.Generator().manual_seed(len(text))
            rows.append(torch.randn(12, CONTEXT_DIM, generator=generator))
        return torch.stack(rows)


def _write_snapshot(root, variant="small", base_model="flux.2-klein-base-9b", params=TinyParams, **extra):
    from safetensors.torch import save_file

    folder = root / ("Iris-Z-small" if variant == "small" else "Iris-Z-big")
    folder.mkdir(parents=True)
    config = {
        "variant": variant,
        "base_model": base_model,
        "depth_encoding": "per_image_log",
        "sky_ceiling": 0.95,
        "given_timestep_zero": True,
        "num_modalities": 4,
        "default_guidance": 4.0,
        "trained_resolutions": [512, 1024],
        **extra,
    }
    (folder / "config.json").write_text(json.dumps(config))
    torch.manual_seed(1)
    network = irisz_nn.IrisZTransformer(params(), num_modalities=4)
    for parameter in network.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in network.state_dict().items()}
    keys = sorted(state)
    half = len(keys) // 2
    shards = {
        "transformer-00001-of-00002.safetensors": keys[:half],
        "transformer-00002-of-00002.safetensors": keys[half:],
    }
    weight_map = {}
    for name, shard_keys in shards.items():
        save_file({k: state[k] for k in shard_keys}, str(folder / name))
        weight_map.update({k: name for k in shard_keys})
    (folder / "transformer.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    empty = torch.randn(1, 12, CONTEXT_DIM, generator=torch.Generator().manual_seed(9))
    save_file({"empty_prompt": empty.to(torch.bfloat16)}, str(folder / "empty_prompt.safetensors"))
    return root


@pytest.fixture
def tiny_runtime(monkeypatch):
    """Shrink every network the loader builds and count text encoder loads."""
    # The loader needs the gen extra (diffusers for the FLUX.2 VAE).
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    monkeypatch.setattr(irisz_nn, "Klein9BParams", TinyParams)
    monkeypatch.setattr(irisz_nn, "Flux2Params", TinyGuidedParams)
    monkeypatch.setattr(LibreSperidlabsIrisZ, "_load_vae", staticmethod(lambda: _tiny_vae().float()))
    loads = []

    def factory(device="cpu"):
        loads.append(str(device))
        return StubTextEncoder(device)

    monkeypatch.setattr(
        irisz_text, "TEXT_ENCODERS", {"mistral_small": factory, "qwen3_8b": factory}
    )
    return loads


@pytest.fixture
def snapshot(tmp_path):
    return _write_snapshot(tmp_path / "iris-z")


@pytest.fixture
def photo(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "street.png"
    Image.fromarray(rng.integers(0, 255, (64, 88, 3), dtype=np.uint8)).save(path)
    return path


# ---------------------------------------------------------------------------
# Names and construction (no loading)
# ---------------------------------------------------------------------------


def test_aliases_keep_the_maker_in_the_name():
    for alias, size in (
        ("speridlabs-iris-z", "9b"),
        ("speridlabs-iris-z-9b", "9b"),
        ("speridlabs-iris-z-small", "9b"),
        ("speridlabs-iris-z-32b", "32b"),
        ("speridlabs-iris-z-big", "32b"),
    ):
        assert _ALIASES[alias] == (LibreSperidlabsIrisZ, size)
    assert "iris-z" not in _ALIASES and "irisz" not in _ALIASES
    model = LibreGen("speridlabs_iris_z", device="cpu")
    assert isinstance(model, LibreSperidlabsIrisZ)
    assert model.model_name == "speridlabs-iris-z-9b"
    assert model._pipeline is None


def test_weights_point_to_speridlabs_at_a_pinned_revision():
    for size, folder in (("9b", "Iris-Z-small"), ("32b", "Iris-Z-big")):
        assert LibreSperidlabsIrisZ.HF_REPOS[size] == "speridlabs/iris-z"
        assert len(LibreSperidlabsIrisZ.HF_REVISIONS[size]) == 40
        model = LibreSperidlabsIrisZ(size, device="cpu")
        assert model.SNAPSHOT_ALLOW_PATTERNS == (f"{folder}/*",)
    for revision in (irisz_text.MISTRAL_REVISION, irisz_text.MISTRAL_PROCESSOR_REVISION, irisz_text.QWEN3_REVISION):
        assert len(revision) == 40


def test_tasks_and_knobs_per_task():
    generate = LibreSperidlabsIrisZ("9b", device="cpu")
    assert generate.task == "generate"
    assert generate.DEFAULTS == {"steps": 50, "guidance": 4.0, "depth": False}
    assert LibreSperidlabsIrisZ("32b", device="cpu").DEFAULTS == {"steps": 50, "guidance": 3.5, "depth": False}
    depth = LibreSperidlabsIrisZ("9b", device="cpu", task="depth-estimation")
    assert depth.task == "depth"
    assert depth.DEFAULTS == {"steps": 4, "ensemble": 1}
    with pytest.raises(ValueError, match="task only"):
        LibreSperidlabsIrisZ("9b", device="cpu", task="detect")


def test_constructor_validation():
    with pytest.raises(ValueError, match="bfloat16"):
        LibreSperidlabsIrisZ("9b", device="cpu", dtype="fp16")
    with pytest.raises(ValueError, match="Invalid size"):
        LibreSperidlabsIrisZ("4b", device="cpu")


def test_export_and_training_are_refused_with_reasons():
    model = LibreSperidlabsIrisZ("9b", device="cpu", task="depth")
    with pytest.raises(NotImplementedError, match="sampling"):
        model.export("onnx")
    with pytest.raises(NotImplementedError, match="Training"):
        model.train(data="x")
    from libreyolo.export.support import get_support

    for task in ("generate", "depth"):
        for fmt in ("onnx", "tensorrt", "openvino"):
            entry = get_support("speridlabs_irisz", task, fmt)
            assert entry.tier == "blocked" and "Iris-Z" in entry.reason


# ---------------------------------------------------------------------------
# Upstream reference values
# ---------------------------------------------------------------------------


def test_schedule_matches_upstream():
    assert irisz_pipeline.get_schedule(4, 1024) == pytest.approx(
        [1.0, 0.9580852985, 0.8839818835, 0.7174965739, 0.0], abs=1e-8
    )
    assert irisz_pipeline.get_schedule(4, 5000) == pytest.approx(
        [1.0, 0.9169514775, 0.7863422036, 0.5509239435, 0.0], abs=1e-8
    )
    schedule = irisz_pipeline.get_schedule(50, 4096)
    assert schedule[:4] == pytest.approx([1.0, 0.9973090887, 0.9945214391, 0.9916315079], abs=1e-8)
    assert schedule[-2] == pytest.approx(0.1337189525, abs=1e-8)


@pytest.mark.parametrize(
    "size, target, expected",
    [
        ((640, 480), 512, (576, 432)),
        ((640, 480), 1024, (640, 480)),
        ((1920, 1080), 512, (656, 368)),
        ((1920, 1080), 1024, (1280, 720)),
        ((300, 200), 1024, (288, 192)),
        ((4000, 3000), 1024, (1152, 864)),
    ],
)
def test_depth_resolution_matches_upstream(size, target, expected):
    assert irisz_pipeline.choose_resolution(*size, target) == expected


def test_sky_decoding_matches_upstream():
    values = irisz_pipeline.unsqueeze_from_ceiling(np.array([-1.0, -0.5, 0.0, 0.9, 0.96, 1.0]), 0.95)
    np.testing.assert_allclose(
        values, [-1.0, -0.4871794871794871, 0.025641025641025772, 0.9487179487179487, 1.0, 1.0]
    )
    np.testing.assert_array_equal(irisz_pipeline.unsqueeze_from_ceiling(np.array([0.3, 1.0]), 1.0), [0.3, 1.0])


def test_small_images_are_enlarged_instead_of_refused():
    with pytest.raises(ValueError):
        irisz_pipeline.choose_resolution(40, 40, 512)
    assert irisz_pipeline.working_size(40, 40, 512) == (512, 512)
    assert irisz_pipeline.working_size(12, 30, 512) == (320, 800)
    # Sizes upstream accepts are unchanged.
    assert irisz_pipeline.working_size(1920, 1080, 1024) == (1280, 720)


def test_depth_on_a_tiny_image(tiny_runtime, snapshot):
    model = LibreSperidlabsIrisZ(device="cpu", task="depth", checkpoint_dir=str(snapshot))
    result = model(np.zeros((20, 24, 3), dtype=np.uint8), steps=3, imgsz=512)
    assert tuple(result.depth_map.data.shape) == (20, 24)


def test_ensemble_aligns_members_and_keeps_sky():
    base = np.linspace(-1, 0.5, 12).reshape(3, 4)
    base[0, 0] = 1.0
    members = [base, 2.0 * base + 0.1, base.copy()]
    members[1][0, 0] = 1.0
    merged = irisz_pipeline.ensemble(members)
    np.testing.assert_allclose(merged[1:], base[1:], atol=1e-6)
    assert merged[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_text_layers_stack_layer_major_per_token():
    hidden = [torch.full((1, 2, 3), float(i)) for i in range(5)]
    stacked = irisz_text._stack_layers(hidden, (1, 3))
    assert stacked.shape == (1, 2, 6)
    assert stacked[0, 0].tolist() == [1, 1, 1, 3, 3, 3]


def test_vae_round_trip_shapes_and_normalization():
    pytest.importorskip("diffusers")
    latents = irisz_pipeline.Flux2Latents(_tiny_vae())
    pixels = torch.rand(1, 3, 64, 48) * 2 - 1
    latent = latents.encode(pixels)
    assert latent.shape == (1, 128, 4, 3)
    decoded = latents.decode(latent)
    assert decoded.shape == (1, 3, 64, 48)
    # The patchify is undone exactly by decode's unpatchify.
    raw = latents.vae.encode(pixels).latent_dist.mean
    mean, std = latents._stats(latent)
    unpatched = (latent * std + mean).view(1, 32, 2, 2, 4, 3).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 8, 6)
    torch.testing.assert_close(unpatched, raw, rtol=1e-5, atol=1e-5)


def test_samplers_shapes_and_seeds():
    torch.manual_seed(0)
    network = irisz_nn.IrisZTransformer(TinyParams(), num_modalities=4).eval()
    context = irisz_pipeline.text_context(torch.randn(2, 5, CONTEXT_DIM))
    rgb = irisz_pipeline.sample_rgb(network, context, 48, 32, seed=3, guidance=4.0, num_steps=2)
    assert rgb.shape == (1, 128, 3, 2)
    again = irisz_pipeline.sample_rgb(network, context, 48, 32, seed=3, guidance=4.0, num_steps=2)
    torch.testing.assert_close(rgb, again, rtol=0, atol=0)
    single = (context[0][:1], context[1][:1])
    depths = irisz_pipeline.sample_depth(network, single, torch.randn(1, 128, 3, 2), [0, 1], 3)
    assert [tuple(d.shape) for d in depths] == [(1, 128, 3, 2)] * 2
    assert not torch.equal(depths[0], depths[1])
    joint_rgb, joint_depth = irisz_pipeline.sample_joint(network, context, 48, 32, seed=3, guidance=4.0, num_steps=2)
    assert joint_rgb.shape == joint_depth.shape == (1, 128, 3, 2)
    given = torch.randn(1, 128, 3, 2)
    painted = irisz_pipeline.sample_rgb_from_depth(network, context, given, seed=3, guidance=4.0, num_steps=2)
    assert painted.shape == (1, 128, 3, 2)
    other = irisz_pipeline.sample_rgb_from_depth(network, context, -given, seed=3, guidance=4.0, num_steps=2)
    assert not torch.equal(painted, other), "the given depth must condition the image"


def test_depth_encoding_matches_upstream():
    rng = np.random.default_rng(0)
    log_depth = rng.normal(size=(20, 30))
    sky = np.zeros((20, 30), dtype=bool)
    sky[:4] = True
    encoded = irisz_pipeline.encode_log_depth(log_depth, sky, sky_ceiling=0.8)
    assert encoded.dtype == np.float32
    assert (encoded[sky] == 1.0).all()
    real = encoded[~sky]
    assert real.min() == pytest.approx(-1.0) and real.max() == pytest.approx(0.8)
    # The 2nd and 98th percentiles land on the ends of [-1, sky_ceiling].
    assert np.mean(real <= -1.0 + 1e-6) == pytest.approx(0.02, abs=0.01)
    decoded = irisz_pipeline.unsqueeze_from_ceiling(encoded, 0.8)
    assert (decoded[sky] == 1.0).all() and decoded[~sky].max() <= 1.0
    # Grayscale depth maps: bright = far for "depth", bright = near for "disparity".
    ramp = Image.fromarray(np.tile(np.linspace(10, 250, 64, dtype=np.uint8), (32, 1)))
    far_right = irisz_pipeline.read_depth_map(ramp, "depth", 32)
    near_right = irisz_pipeline.read_depth_map(ramp, "disparity", 32)
    assert far_right[:, -1].mean() > far_right[:, 0].mean()
    assert near_right[:, -1].mean() < near_right[:, 0].mean()
    assert far_right.shape[0] % 16 == 0 and far_right.shape[1] % 16 == 0
    # Upstream keeps a depth that fits the target side and resizes one that does not.
    fits = np.zeros((32, 48), dtype=np.float32)
    assert irisz_pipeline.fit_encoded_depth(fits, 48).shape == (32, 48)
    assert irisz_pipeline.fit_encoded_depth(np.zeros((40, 60), dtype=np.float32), 32).shape != (40, 60)


# ---------------------------------------------------------------------------
# Loading and the two tasks, end to end on the tiny snapshot
# ---------------------------------------------------------------------------


def test_matches_the_snapshot_and_infers_the_size(snapshot, tmp_path):
    assert LibreSperidlabsIrisZ.matches_directory(snapshot)
    assert LibreSperidlabsIrisZ.matches_directory(snapshot / "Iris-Z-small")
    assert not LibreSperidlabsIrisZ.matches_directory(tmp_path)
    model = LibreGen(str(snapshot), device="cpu", task="depth")
    assert isinstance(model, LibreSperidlabsIrisZ) and model.size == "9b"
    with pytest.raises(FileNotFoundError, match="Iris-Z-big"):
        LibreSperidlabsIrisZ("32b", device="cpu", checkpoint_dir=str(snapshot))


def test_depth_end_to_end(tiny_runtime, snapshot, photo, tmp_path, monkeypatch):
    model = LibreSperidlabsIrisZ(device="cpu", task="depth", checkpoint_dir=str(snapshot))
    result = model(str(photo), steps=3)
    assert tiny_runtime == [], "depth must not load a text encoder"
    assert isinstance(result, Results)
    depth = result.depth_map
    assert depth.encoding == "log_depth" and not depth.metric
    assert tuple(depth.data.shape) == (64, 88) == result.orig_shape
    assert depth.valid_mask.shape == (64, 88)
    values = np.asarray(depth.data)
    assert np.isfinite(values).all() and values.min() >= -1 and values.max() <= 1
    # Same seed, same depth; the default seed is upstream's 0.
    np.testing.assert_array_equal(np.asarray(model(str(photo), steps=3, seed=0).depth_map.data), values)
    monkeypatch.chdir(tmp_path)
    twin = tmp_path / "other" / photo.name
    twin.parent.mkdir()
    twin.write_bytes(photo.read_bytes())
    ensemble = model([str(photo), str(twin), Image.open(photo)], steps=3, ensemble=2, save=True)
    assert len(ensemble) == 3
    saved = sorted((tmp_path / "runs" / "depth" / "predict").glob("*.png"))
    assert [p.name for p in saved] == ["000_street.png", "001_street.png", "002_image.png"]


def test_depth_argument_validation(tiny_runtime, snapshot, photo):
    model = LibreSperidlabsIrisZ(device="cpu", task="depth", checkpoint_dir=str(snapshot))
    with pytest.raises(ValueError, match="512, 1024"):
        model(str(photo), imgsz=640)
    with pytest.raises(ValueError, match="ensemble"):
        model(str(photo), ensemble=0)
    with pytest.raises(ValueError, match="more than 2 steps"):
        model(str(photo), steps=2)
    with pytest.raises(TypeError, match="guidance"):
        model(str(photo), guidance=3.0)
    with pytest.raises(ValueError, match="source"):
        model()
    with pytest.raises(ValueError, match="offload"):
        LibreSperidlabsIrisZ(device="cpu", task="depth", offload=True)


def test_text_to_image_end_to_end(tiny_runtime, snapshot, tmp_path):
    model = LibreSperidlabsIrisZ(device="cpu", checkpoint_dir=str(snapshot))
    results = model("a lighthouse at dusk", imgsz=(64, 48), seed=5, n=2, steps=2)
    assert tiny_runtime == ["cpu"]
    encoder = model.pipeline.text_encoder
    assert encoder.calls == [["", "a lighthouse at dusk"]]
    assert [r.generated.data.shape for r in results] == [(64, 48, 3)] * 2
    assert [r.generated.seed for r in results] == [5, 6]
    assert results[0].generated.params["guidance"] == 4.0
    assert results[0].generated.params["model"] == str(snapshot)
    first = np.asarray(results[0].generated.data)
    repeat = model("a lighthouse at dusk", imgsz=(64, 48), seed=5, steps=2)
    np.testing.assert_array_equal(np.asarray(repeat.generated.data), first)
    with pytest.raises(ValueError, match="needs a prompt"):
        model(source="photo.jpg")
    with pytest.raises(TypeError, match="depth_format"):
        model("x", depth_format="depth")
    with pytest.raises(TypeError, match="ensemble"):
        model("x", ensemble=2)
    with pytest.raises(ValueError, match="guidance"):
        model("x", guidance=-1.0, imgsz=32, steps=1)


def test_image_and_depth_from_a_prompt(tiny_runtime, snapshot):
    model = LibreSperidlabsIrisZ(device="cpu", checkpoint_dir=str(snapshot))
    result = model("a grand old library", imgsz=(64, 48), seed=2, steps=2, depth=True)
    assert result.generated.data.shape == (64, 48, 3)
    assert result.generated.params["depth"] is True
    depth = result.depth_map
    assert depth.encoding == "log_depth" and tuple(depth.data.shape) == (64, 48)
    assert np.isfinite(np.asarray(depth.data)).all()
    repeat = model("a grand old library", imgsz=(64, 48), seed=2, steps=2, depth=True)
    np.testing.assert_array_equal(np.asarray(repeat.depth_map.data), np.asarray(depth.data))
    assert model("a grand old library", imgsz=(64, 48), seed=2, steps=2).depth_map is None
    with pytest.raises(ValueError, match="True or False"):
        model("x", depth="yes", imgsz=32, steps=1)


def test_depth_to_image_from_every_source(tiny_runtime, snapshot, photo, tmp_path, monkeypatch):
    model = LibreSperidlabsIrisZ(device="cpu", checkpoint_dir=str(snapshot))
    # A photo: its depth is estimated first, then the prompt is painted on it.
    result = model("a neon city in the rain", source=str(photo), imgsz=512, seed=1, steps=2)
    image = np.asarray(result.generated.data)
    assert image.shape[2] == 3 and image.shape[0] % 16 == 0 and image.shape[1] % 16 == 0
    assert result.depth_map.data.shape == image.shape[:2]
    assert result.generated.params["source"] == str(photo)
    assert model.pipeline.text_encoder.calls[-1] == ["", "a neon city in the rain"]
    repeat = model("a neon city in the rain", source=str(photo), imgsz=512, seed=1, steps=2)
    np.testing.assert_array_equal(np.asarray(repeat.generated.data), image)
    # A depth result, a DepthMap of another encoding, and a depth map image.
    depth_model = LibreSperidlabsIrisZ(device="cpu", task="depth", checkpoint_dir=str(snapshot))
    estimated = depth_model(str(photo), steps=3)
    from_result = model("a sandcastle", source=estimated, imgsz=64, seed=1, steps=2)
    assert from_result.generated.params["source"] == "Results"
    inverse = type(estimated.depth_map)(-np.asarray(estimated.depth_map.data), (64, 88))
    model("a sandcastle", source=inverse, imgsz=64, seed=1, steps=2)
    depth_png = tmp_path / "depth.png"
    Image.fromarray(np.tile(np.linspace(0, 255, 88, dtype=np.uint8), (64, 1))).save(depth_png)
    monkeypatch.chdir(tmp_path)
    many = model(["jungle", "sandcastle"], source=str(depth_png), depth_format="disparity", imgsz=64, n=2, steps=2,
                 seed=4, save=True)
    assert [r.generated.seed for r in many] == [4, 5, 4, 5]
    assert len(list((tmp_path / "runs" / "generate" / "predict").glob("*.png"))) == 4
    with pytest.raises(ValueError, match="512 or 1024"):
        model("x", source=str(photo), imgsz=640)
    with pytest.raises(ValueError, match="depth_format"):
        model("x", source=str(depth_png), depth_format="colours", imgsz=64, steps=1)
    with pytest.raises(ValueError, match="depth is given"):
        model("x", source=str(photo), depth=True)
    with pytest.raises(ValueError, match="one side"):
        model("x", source=str(photo), imgsz=(64, 48))
    with pytest.raises(TypeError, match="not a list"):
        model("x", source=[str(photo)], steps=1)
    with pytest.raises(TypeError, match="DepthMap"):
        model("x", source=estimated, depth_format="depth", steps=1)
    with pytest.raises(TypeError, match="task='depth'"):
        depth_model(str(photo), depth_format="depth")


def test_depth_map_sources_keep_their_values(tiny_runtime, snapshot, tmp_path):
    from libreyolo.models.gen.speridlabs_irisz.model import _load_depth_map_image

    # A 16-bit depth map as a NumPy array is not clipped to 8 bits.
    ramp = np.tile(np.linspace(1000, 60000, 88).astype(np.uint16), (64, 1))
    image, path = _load_depth_map_image(ramp)
    assert path is None and np.asarray(image).max() == 60000
    model = LibreSperidlabsIrisZ(device="cpu", checkpoint_dir=str(snapshot))
    from_array = model("x", source=ramp, depth_format="depth", imgsz=64, seed=1, steps=1)
    png = tmp_path / "depth16.png"
    Image.fromarray(ramp).save(png)
    from_file = model("x", source=str(png), depth_format="depth", imgsz=64, seed=1, steps=1)
    np.testing.assert_array_equal(np.asarray(from_array.generated.data), np.asarray(from_file.generated.data))
    assert from_file.generated.params["depth_format"] == "depth"
    assert "depth_format" not in model("x", source=from_file, imgsz=64, seed=1, steps=1).generated.params


def test_bad_output_path_fails_before_loading(tiny_runtime, snapshot, photo):
    model = LibreSperidlabsIrisZ(device="cpu", checkpoint_dir=str(snapshot))
    with pytest.raises(ValueError, match="image extension"):
        model("x", source=str(photo), save=True, output_path="out.pngg")
    assert model._pipeline is None and tiny_runtime == []


def test_big_variant_loads_with_its_guidance_mode(tiny_runtime, tmp_path):
    root = _write_snapshot(
        tmp_path / "big",
        variant="big",
        base_model="flux.2-dev",
        params=TinyParams,  # guidance_mode "remove" drops the guidance embedder
        guidance_mode="remove",
    )
    model = LibreSperidlabsIrisZ("32b", device="cpu", checkpoint_dir=str(root))
    assert model.DEFAULTS["guidance"] == 3.5
    result = model("a red fox", imgsz=32, seed=0, steps=1)
    assert result.generated.data.shape == (32, 32, 3)
    assert not model.pipeline.transformer.use_guidance_embed


def test_wrong_variant_folder_is_refused(tiny_runtime, tmp_path):
    root = tmp_path / "mixed"
    _write_snapshot(root)
    config_path = root / "Iris-Z-small" / "config.json"
    config = json.loads(config_path.read_text())
    config["variant"] = "big"
    config_path.write_text(json.dumps(config))
    model = LibreSperidlabsIrisZ("9b", device="cpu", task="depth", checkpoint_dir=str(root))
    with pytest.raises(ValueError, match="not Iris-Z-small"):
        model(np.zeros((32, 32, 3), dtype=np.uint8))


def test_qwen3_falls_back_to_dequantized_without_the_fp8_kernel(monkeypatch, caplog):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    loads = []

    class FakeModel:
        def __init__(self, dequantize):
            self.dequantize = dequantize
            self.device = torch.device("cpu")

        def __call__(self, **kwargs):
            if not self.dequantize:
                raise ImportError("finegrained-fp8 kernel unavailable")

    def fake_load(device, dequantize):
        loads.append((str(device), dequantize))
        return FakeModel(dequantize)

    monkeypatch.setattr(irisz_text.Qwen3Embedder, "_load", staticmethod(fake_load))
    with caplog.at_level("WARNING"):
        encoder = irisz_text.Qwen3Embedder(device="cuda")
    assert loads == [("cuda", False), ("cuda", True)]
    assert not encoder.fp8 and encoder.model.dequantize
    assert "dequantized" in caplog.text
    loads.clear()
    encoder = irisz_text.Qwen3Embedder(device="cpu")
    assert loads == [("cpu", True)] and not encoder.fp8
