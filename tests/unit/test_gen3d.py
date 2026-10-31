"""Offline unit tests for image-to-3D generation (ADR 0031): no weights, no network.

Covers the ``generate3d`` task, the ``Gaussians`` payload and its files, the
``Results`` integration, the shared image-to-3D call contract through a fake
family, and the Speridlabs Idol networks and pipeline at a tiny size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo import Gaussians, LibreGen, LibreSperidlabsIdol, Results
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.base3d import LibreGen3DModel
from libreyolo.tasks import TASKS, normalize_task, suffix_to_task, task_to_suffix

pytestmark = [pytest.mark.unit, pytest.mark.gen]


def _gaussians(n=6, seed=0, **kwargs) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    rotations = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    return Gaussians(
        torch.rand(n, 3, generator=g) - 0.5,
        torch.rand(n, 3, generator=g) * 0.02 + 1e-3,
        rotations,
        torch.rand(n, generator=g),
        torch.randn(n, 1, 3, generator=g),
        camera={"yaw": 0.3, "pitch": 0.2, "radius": 2.5, "fov": 30.0},
        seed=7,
        params={"model": "fake", "steps": 2},
        **kwargs,
    )


def _read_ply(path):
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    names = [line.split()[-1] for line in header if line.startswith("property")]
    data = np.frombuffer(raw[end:], dtype=np.dtype([(name, "<f4") for name in names]))
    return header, names, data


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------


def test_generate3d_task_is_canonical():
    assert "generate3d" in TASKS
    for alias in ("generate3d", "image-to-3d", "image23d", "3d-generation", "i23d"):
        assert normalize_task(alias) == "generate3d"
    assert task_to_suffix("generate3d") == "generate3d"
    assert suffix_to_task("-generate3d") == "generate3d"


# ---------------------------------------------------------------------------
# Gaussians payload
# ---------------------------------------------------------------------------


def test_payload_validates_shapes():
    good = _gaussians()
    assert len(good) == 6
    with pytest.raises(ValueError, match="scales"):
        Gaussians(good.xyz, good.scales[:, :2], good.rotations, good.opacities, good.sh)
    with pytest.raises(ValueError, match="sh"):
        Gaussians(good.xyz, good.scales, good.rotations, good.opacities, good.sh[:, 0])


def test_payload_conversions_keep_metadata():
    g = _gaussians(input_image=Image.new("RGB", (4, 4)))
    for converted in (g.cpu(), g.numpy(), g.to(torch.float64)):
        assert converted.seed == 7
        assert converted.camera == g.camera
        assert converted.params == g.params
        assert converted.input_image is g.input_image
    assert isinstance(g.numpy().xyz, np.ndarray)
    assert g.to(torch.float64).xyz.dtype == torch.float64


def test_payload_selects_gaussians():
    g = _gaussians(n=10)
    kept = g[g.opacities > 0.5]
    assert len(kept) == int((g.opacities > 0.5).sum())
    assert len(g[0]) == 1
    assert len(g[-1]) == 1
    assert torch.equal(g[-1].xyz, g.xyz[-1:])
    # Scalar indices of any kind keep one Gaussian with its row dimension.
    for index in (np.int64(2), g.opacities.argmax(), torch.tensor(3)):
        one = g[index]
        assert len(one) == 1 and one.sh.shape == (1, 1, 3)
    assert len(g.numpy()[np.int64(1)]) == 1
    # Iteration stops after the last Gaussian (an index past the end raises).
    assert [len(one) for one in g] == [1] * 10
    assert len(list(g.numpy())) == 10


def test_colors_decode_the_degree_zero_term():
    g = _gaussians()
    expected = (g.sh[:, 0] * 0.28209479177387814 + 0.5).clamp(0, 1)
    assert torch.allclose(g.colors, expected)
    assert np.allclose(g.numpy().colors, expected.numpy())


def test_ply_is_the_standard_layout_turned_y_up(tmp_path):
    g = _gaussians()
    g.save(tmp_path / "object.ply")
    header, names, data = _read_ply(tmp_path / "object.ply")
    assert header[:2] == ["ply", "format binary_little_endian 1.0"]
    comment = next(line for line in header if line.startswith("comment libreyolo "))
    record = json.loads(comment[len("comment libreyolo ") :])
    assert record["seed"] == 7 and record["steps"] == 2 and record["count"] == 6
    assert names == [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]  # fmt: skip
    xyz = g.xyz.numpy()
    assert np.array_equal(data["x"], xyz[:, 0])
    assert np.array_equal(data["y"], -xyz[:, 2])
    assert np.array_equal(data["z"], xyz[:, 1])
    opacity = g.opacities.clamp(1e-6, 1 - 1e-6)
    assert np.allclose(
        data["opacity"], torch.log(opacity / (1 - opacity)).numpy(), atol=1e-6
    )
    assert np.allclose(data["scale_0"], torch.log(g.scales[:, 0]).numpy())
    # +90 degrees about x: the object's z axis becomes the viewer's y axis.
    q = np.stack([data[f"rot_{i}"] for i in range(4)], axis=1)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)


def test_higher_order_sh_is_refused_in_ply_and_dropped_in_splat(tmp_path):
    base = _gaussians(n=3)
    g = Gaussians(
        base.xyz, base.scales, base.rotations, base.opacities, torch.randn(3, 4, 3)
    )
    with pytest.raises(NotImplementedError, match="higher-order"):
        g.save(tmp_path / "sh.ply")
    assert not (tmp_path / "sh.ply").exists()
    g.save(tmp_path / "sh.splat")
    assert (tmp_path / "sh.splat").stat().st_size == 32 * 3


def test_splat_is_32_bytes_per_gaussian_sorted(tmp_path):
    g = _gaussians(n=8)
    g.save(tmp_path / "object.splat")
    raw = (tmp_path / "object.splat").read_bytes()
    assert len(raw) == 32 * 8
    record = np.frombuffer(
        raw,
        dtype=[
            ("xyz", "<f4", 3),
            ("scale", "<f4", 3),
            ("rgba", "u1", 4),
            ("rot", "u1", 4),
        ],
    )
    keys = np.prod(record["scale"], axis=1) * record["rgba"][:, 3]
    assert np.all(np.diff(keys) <= 1e-6 * keys.max())
    assert set(np.round(record["xyz"][:, 1], 5)) == set(
        np.round(-g.xyz[:, 2].numpy(), 5)
    )


def test_save_rejects_other_formats(tmp_path):
    with pytest.raises(ValueError, match=r"\.ply or \.splat"):
        _gaussians().save(tmp_path / "object.obj")


def test_render_draws_the_object_in_view():
    # A dense ball at the origin seen from the camera fills the image centre.
    n = 4000
    g = torch.Generator().manual_seed(1)
    direction = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=1)
    xyz = direction * 0.2 * torch.rand(n, 1, generator=g) ** (1 / 3)
    ball = Gaussians(
        xyz,
        torch.full((n, 3), 0.004),
        torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        torch.ones(n),
        torch.full((n, 1, 3), 1.5),
        camera={"yaw": 0.7, "pitch": 0.4, "radius": 3.0, "fov": 30.0, "margin": 1.6},
    )
    image = np.array(ball.render(size=128))
    assert image.shape == (128, 128, 3)
    assert image[64, 64].max() > 200
    assert image[2, 2].max() == 0
    filled = image.max(axis=2) > 0
    ys, xs = np.nonzero(filled)
    # Framed like the input crop: centred, spanning 1 / margin of the image.
    assert abs(xs.mean() - 64) < 3 and abs(ys.mean() - 64) < 3
    assert abs((xs.max() - xs.min()) - 128 / 1.6) < 4


# ---------------------------------------------------------------------------
# Results integration
# ---------------------------------------------------------------------------


def test_results_integration(tmp_path):
    g = _gaussians()
    result = Results(None, (20, 30), path="chair.jpg", gaussians=g)
    assert len(result) == 1
    assert result.gaussians is g
    assert result[0].gaussians is g  # result slicing keeps the whole object
    assert result.cpu().gaussians.seed == 7
    assert result.summary() == [g.to_dict()]
    assert json.loads(result.to_json())[0]["name"] == "gaussians"
    assert result.save(str(tmp_path / "a.ply")) == str(tmp_path / "a.ply")
    assert (tmp_path / "a.ply").is_file()
    preview = result.plot()
    assert isinstance(preview, Image.Image)
    assert "gaussians=Gaussians(n=6, seed=7)" in repr(result)


# ---------------------------------------------------------------------------
# Call contract through a fake family
# ---------------------------------------------------------------------------


class Fake3D(LibreGen3DModel):
    """Encodes the seed and the input size in the object so tests can read them back."""

    FAMILY = "fake_3d"
    FILENAME_PREFIX = "LibreFake3D"
    INPUT_SIZES = {"s": 64}
    KNOBS = ("steps", "gaussians")
    DEFAULTS = {"steps": 2, "gaussians": 4}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def _ensure_weights(self) -> str:
        return "unused"

    def _load(self, snapshot_dir: str) -> None:
        self._pipeline = nn.Linear(1, 1)

    def _generate_3d(self, image, seeds, knobs):
        self.calls.append((image.mode, image.size, list(seeds), dict(knobs)))
        out = []
        for seed in seeds:
            n = knobs["gaussians"]
            out.append(
                Gaussians(
                    torch.full((n, 3), float(seed)),
                    torch.full((n, 3), 0.01),
                    torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                    torch.ones(n),
                    torch.zeros(n, 1, 3),
                    seed=seed,
                    params=dict(knobs),
                )
            )
        return out


@pytest.fixture
def model():
    return Fake3D("s", device="cpu")


@pytest.fixture
def chair(tmp_path):
    path = tmp_path / "chair.png"
    Image.new("RGB", (40, 30), (200, 10, 10)).save(path)
    return path


def test_one_source_returns_one_result(model, chair):
    result = model(chair, seed=5)
    assert isinstance(result, Results)
    assert result.path == str(chair)
    assert result.orig_shape == (30, 40)
    assert result.orig_img.shape == (30, 40, 3)
    assert result.gaussians.seed == 5
    assert model.calls[0][2] == [5]
    assert model.predict is not None and model.generate.__name__ == "generate"


def test_sources_like_predict(model, chair, tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("b.png", "a.jpg"):
        Image.new("RGB", (8, 8)).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    results = model(folder, seed=1)
    assert [Path(r.path).name for r in results] == ["a.jpg", "b.png"]
    results = model(
        [chair, Image.new("RGB", (8, 8)), np.zeros((6, 9, 3), np.uint8)], seed=1
    )
    assert [r.orig_shape for r in results] == [(30, 40), (8, 8), (6, 9)]
    assert model(source=chair, seed=1).path == str(chair)


def test_batches_split_like_predict(model):
    results = model(np.zeros((3, 12, 10, 3), np.uint8), seed=0)
    assert [r.orig_shape for r in results] == [(12, 10)] * 3
    assert len(model(torch.zeros(2, 3, 8, 8), seed=0)) == 2
    assert isinstance(model(np.zeros((1, 8, 8, 3), np.uint8), seed=0), Results)


def test_n_and_seeds(model, chair):
    results = model([chair, chair], seed=2**32 - 1, n=2)
    assert [r.gaussians.seed for r in results] == [2**32 - 1, 0, 2**32 - 1, 0]
    drawn = model(chair)
    assert 0 <= drawn.gaussians.seed < 2**32


def test_real_alpha_reaches_the_family(model, tmp_path):
    rgba = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    rgba.putpixel((8, 8), (255, 0, 0, 255))
    rgba.save(tmp_path / "cutout.png")
    model(tmp_path / "cutout.png")
    model(rgba)
    assert [call[0] for call in model.calls] == ["RGBA", "RGBA"]


def test_knobs_and_rejected_arguments(model, chair):
    model(chair, gaussians=8)
    assert model.calls[-1][3] == {"steps": 2, "gaussians": 8}
    with pytest.raises(TypeError, match="does not accept 'guidance'"):
        model(chair, guidance=3.0)
    with pytest.raises(TypeError, match="no text input"):
        model(chair, prompt="a chair")
    with pytest.raises(TypeError, match="fixed input size"):
        model(chair, imgsz=512)
    with pytest.raises(ValueError, match="needs a source"):
        model()
    with pytest.raises(ValueError, match="n must be"):
        model(chair, n=0)
    with pytest.raises(ValueError, match="seed must be"):
        model(chair, seed=-1)
    with pytest.raises(ValueError, match=r"\.ply or \.splat"):
        model(chair, save=True, output_path="x.png")
    with pytest.raises(ValueError, match="one source"):
        model([chair, chair], save=True, output_path="x.ply")
    with pytest.raises(FileNotFoundError):
        model(Path("missing.png"))
    assert len(model.calls) == 1


def test_constructor_validation():
    with pytest.raises(ValueError, match="mixed precision"):
        Fake3D("s", device="cpu", dtype="fp16")
    with pytest.raises(ValueError, match="offload"):
        Fake3D("s", device="cpu", offload=True)
    with pytest.raises(ValueError, match="compile=True needs CUDA"):
        Fake3D("s", device="cpu", compile=True)
    with pytest.raises(ValueError, match="generate3d"):
        Fake3D("s", device="cpu", task="generate")
    assert Fake3D("s", device="cpu", task="image-to-3d").task == "generate3d"


def test_save_writes_ply_files(model, chair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model([chair, chair], seed=3, save=True)
    saved = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.ply"))
    assert saved == [
        "runs/generate/predict/000_chair_3.ply",
        "runs/generate/predict/001_chair_3.ply",
    ]
    model(chair, seed=3, save=True, output_path=str(tmp_path / "out" / "chair.splat"))
    assert (tmp_path / "out" / "chair.splat").stat().st_size == 32 * 4


def test_stream_is_lazy(model, chair):
    stream = model([chair, chair], stream=True)
    assert model.calls == []
    assert len(list(stream)) == 2


def test_unsupported_surfaces_say_why(model):
    for call, match in (
        (model.train, "image and 3D object pairs"),
        (model.val, "ADR 0031"),
        (model.export, "ADR 0031"),
        (model.track, "tracking"),
    ):
        with pytest.raises(NotImplementedError, match=match):
            call()


# ---------------------------------------------------------------------------
# Speridlabs Idol: factory, knobs, encoder access
# ---------------------------------------------------------------------------


def test_idol_alias_resolves_without_loading():
    assert _ALIASES["speridlabs-idol"] == (LibreSperidlabsIdol, "0.6b")
    model = LibreGen("speridlabs-idol", device="cpu")
    assert isinstance(model, LibreSperidlabsIdol)
    assert model.task == "generate3d"
    assert model.model_name == "speridlabs-idol-0.6b"
    assert model._pipeline is None
    assert "idol" not in _ALIASES  # the model's name carries its maker


def test_idol_pins_every_download():
    from libreyolo.models.gen.speridlabs_idol import model as idol

    assert (
        LibreSperidlabsIdol.HF_REVISIONS["0.6b"]
        == "2731c2291fc65a797c5cb627a5e5526fc1490902"
    )
    for revision in (idol._DINOV3_REVISION, idol._FLUX2_VAE_REVISION):
        assert len(revision) == 40
    # Nothing gated: the VAE comes from the Apache-2.0 klein repo LibreGen
    # already uses, DINOv3 from LibreYOLO's ungated mirror.
    assert idol._FLUX2_VAE_REPO == "black-forest-labs/FLUX.2-klein-4B"
    assert idol._DINOV3_REPO == "LibreYOLO/LibreDINOv3vith16plus"


def test_idol_needs_no_hugging_face_login(monkeypatch):
    import huggingface_hub

    def online(*args, **kwargs):
        raise AssertionError("checking Hub access is not needed")

    monkeypatch.setattr(huggingface_hub, "auth_check", online)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    LibreSperidlabsIdol(device="cpu")._check_runtime()


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"shift": 0}, "shift must be"),
        ({"shift": float("nan")}, "shift must be"),
        ({"guidance": -1.0}, "guidance must be"),
        ({"guidance": float("inf")}, "guidance must be"),
        ({"margin": float("inf")}, "margin must be"),
        ({"mode": "photo"}, "mode must be"),
        ({"gaussians": 1000}, "gaussians must be"),
        ({"gaussians": 2_000_000}, "gaussians must be"),
        ({"steps": 0}, "steps must be"),
        ({"margin": 0.5}, "margin must be"),
        ({"cfg_strength": 3.0}, "does not accept"),
    ],
)
def test_idol_knob_validation(knobs, match):
    model = LibreSperidlabsIdol(device="cpu")
    with pytest.raises((ValueError, TypeError), match=match):
        model._resolve_knobs(knobs)


def test_idol_matches_its_snapshot_directory(tmp_path):
    from libreyolo.models.gen.speridlabs_idol.model import _WEIGHT_FILES

    assert not LibreSperidlabsIdol.matches_directory(tmp_path)
    for name in _WEIGHT_FILES:
        (tmp_path / name).write_text("{}")
    assert not LibreSperidlabsIdol.matches_directory(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"image_encoders": {}, "denoiser": {}, "decoder_gs": {}})
    )
    assert LibreSperidlabsIdol.matches_directory(tmp_path)
    assert isinstance(LibreGen(str(tmp_path), device="cpu"), LibreSperidlabsIdol)


# ---------------------------------------------------------------------------
# Speridlabs Idol networks and pipeline at a tiny size
# ---------------------------------------------------------------------------

TINY = {
    "denoiser": {
        "args": {
            "q_token_length": 16,
            "in_channels": 4,
            "cam_channels": 5,
            "model_channels": 24,
            "cond_channels": 8,
            "cond2_channels": 6,
            "num_blocks": 1,
            "num_refiner_blocks": 2,
            "num_heads": 2,
            "num_head_channels": 12,
            "mlp_ratio": 2,
        }
    },
    "decoder": {
        "args": {
            "model_channels": 24,
            "cond_channels": 4,
            "num_blocks": 1,
            "num_heads": 2,
        }
    },
    "decoder_gs": {
        "args": {
            "in_channels": 3,
            "model_channels": 24,
            "cond_channels": 4,
            "num_blocks": 1,
            "num_heads": 2,
            "representation_config": {
                "num_gaussians": 4,
                "lr": {
                    "_xyz": 1.0,
                    "_features_dc": 1.0,
                    "_opacity": 1.0,
                    "_scaling": 1.0,
                    "_rotation": 0.1,
                },
                "perturbe_size": 1.5,
                "3d_filter_kernel_size": 0.0009,
                "scaling_bias": 0.004,
                "opacity_bias": 0.1,
            },
        }
    },
    "image_size": 64,
    "max_voxel_level": 3,
}


def _tiny_networks():
    from libreyolo.models.gen.speridlabs_idol.nn import (
        GaussianFixedlenDecoder,
        LatentSeqMMFlowModel,
        OctreeProbabilityFixedlenDecoder,
    )

    torch.manual_seed(0)
    denoiser = LatentSeqMMFlowModel(**TINY["denoiser"]["args"]).eval()
    denoiser.pos_pe.uniform_()
    decoder = OctreeProbabilityFixedlenDecoder(**TINY["decoder"]["args"]).eval()
    decoder_gs = GaussianFixedlenDecoder(**TINY["decoder_gs"]["args"]).eval()
    return denoiser, decoder, decoder_gs


class _CentreBlob(nn.Module):
    """Stands in for BiRefNet: logits of a centred square."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.full((x.shape[0], 1, *x.shape[2:]), -20.0)
        h, w = x.shape[2:]
        logits[:, :, h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = 20.0
        return logits + self.anchor


class _FakeEncoder:
    def __call__(self, pixels, generator=None):
        return {
            "feature1": torch.randn(1, 7, 8, generator=generator),
            "feature2": torch.randn(1, 7, 6, generator=generator),
        }


def _tiny_idol():
    from libreyolo.models.gen.speridlabs_idol.model import _IdolModules

    model = LibreSperidlabsIdol(device="cpu")
    denoiser, decoder, decoder_gs = _tiny_networks()
    model._pipeline = _IdolModules(
        dict(TINY), denoiser, decoder, decoder_gs, _CentreBlob(), _FakeEncoder()
    )
    return model


def test_preprocess_crops_the_object(tmp_path):
    from libreyolo.models.gen.speridlabs_idol.pipeline import preprocess_image

    image = Image.new("RGB", (120, 80), (40, 200, 40))
    prepared = preprocess_image(image, _CentreBlob(), image_size=64, crop_margin=1.2)
    assert prepared.size == (64, 64) and prepared.mode == "RGB"
    pixels = np.array(prepared)
    assert pixels[32, 32].tolist() == [40, 200, 40]
    assert pixels[0, 0].tolist() == [0, 0, 0]
    rgba = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="No object found"):
        preprocess_image(rgba, _CentreBlob(), image_size=64)


def test_systematic_sample_keeps_every_point():
    from libreyolo.models.gen.speridlabs_idol.nn import systematic_sample

    probs = torch.softmax(torch.randn(2, 5, 8), dim=-1)
    counts = torch.tensor([[7, 0, 3, 3, 100], [1, 2, 3, 4, 5]])
    first = systematic_sample(probs, counts, torch.Generator().manual_seed(4))
    again = systematic_sample(probs, counts, torch.Generator().manual_seed(4))
    assert torch.equal(first, again)
    assert torch.equal(first.sum(-1), counts)


def test_octree_places_exactly_the_requested_points():
    denoiser, decoder, decoder_gs = _tiny_networks()
    latent = torch.randn(1, 16, 4)
    anchors = decoder.sample(
        latent, num_points=37, level=3, generator=torch.Generator().manual_seed(0)
    )
    assert anchors.shape == (1, 37, 3)
    assert anchors.min() >= 0 and anchors.max() <= 1
    splats = decoder_gs.to_gaussians(anchors, decoder_gs(anchors, latent))
    assert splats["xyz"].shape == (37 * 4, 3)
    assert torch.allclose(splats["rotation"].norm(dim=1), torch.ones(37 * 4), atol=1e-5)
    assert (splats["opacity"] > 0).all() and (splats["opacity"] < 1).all()
    assert (splats["scaling"] > 0).all()


def test_batched_guidance_matches_two_passes():
    from libreyolo.models.gen.speridlabs_idol.pipeline import sample_flow

    denoiser, _, _ = _tiny_networks()
    denoiser = denoiser.float()
    denoiser.dtype = torch.float32
    g = torch.Generator().manual_seed(0)
    latent, camera = (
        torch.randn(1, 16, 4, generator=g),
        torch.randn(1, 1, 5, generator=g),
    )
    cond = {
        "feature1": torch.randn(1, 7, 8, generator=g),
        "feature2": torch.randn(1, 7, 6, generator=g),
    }
    two = sample_flow(denoiser, latent, camera, cond, steps=3, batch_guidance=False)
    one = sample_flow(denoiser, latent, camera, cond, steps=3, batch_guidance=True)
    for a, b in zip(two, one):
        assert torch.allclose(a, b, atol=1e-5)


def test_idol_end_to_end_at_a_tiny_size(tmp_path):
    model = _tiny_idol()
    image = Image.new("RGB", (90, 60), (10, 20, 230))
    result = model(image, seed=11, steps=2, gaussians=32_768)
    g = result.gaussians
    assert len(g) == 32_768
    assert g.seed == 11
    assert g.params["steps"] == 2 and g.params["model"] == "speridlabs-idol-0.6b"
    assert set(g.camera) == {"yaw", "pitch", "radius", "fov", "margin"}
    assert g.camera["margin"] == 1.2
    assert g.input_image.size == (64, 64)
    assert g.xyz.dtype == torch.float32
    again = model(image, seed=11, steps=2, gaussians=32_768).gaussians
    assert torch.equal(g.xyz, again.xyz) and g.camera == again.camera
    other = model(image, seed=12, steps=2, gaussians=32_768).gaussians
    assert not torch.equal(g.xyz, other.xyz)
    pair = model(image, seed=11, n=2, steps=2, gaussians=32_768)
    assert torch.equal(pair[0].gaussians.xyz, g.xyz)
    result.save(tmp_path / "tiny.ply")
    _, _, data = _read_ply(tmp_path / "tiny.ply")
    assert len(data) == 32_768


def test_idol_loads_its_training_free_birefnet(tmp_path):
    """Idol ships BiRefNet without the training-only heads; LibreYOLO's network loads it."""
    from libreyolo.models.birefnet.nn import LibreBiRefNetModel
    from libreyolo.models.gen.speridlabs_idol.model import _BIREFNET_TRAINING_ONLY

    state = LibreBiRefNetModel(size="t").state_dict()
    dropped = [key for key in state if key.startswith(_BIREFNET_TRAINING_ONLY)]
    assert dropped
    module = LibreBiRefNetModel(size="t")
    missing, unexpected = module.load_state_dict(
        {k: v for k, v in state.items() if k not in dropped}, strict=False
    )
    assert sorted(missing) == sorted(dropped) and not unexpected


def test_idol_knobs_accept_numpy_integers():
    settings = LibreSperidlabsIdol(device="cpu")._resolve_knobs(
        {"gaussians": np.int64(65536), "steps": np.int32(3)}
    )
    assert settings["gaussians"] == 65536 and type(settings["gaussians"]) is int
    assert settings["steps"] == 3 and type(settings["steps"]) is int


@pytest.mark.parametrize("mode", ["LA", "P"])
def test_cutouts_in_other_alpha_modes_are_kept(mode):
    from libreyolo.models.gen.speridlabs_idol.pipeline import preprocess_image

    class Untouchable(_CentreBlob):
        def forward(self, x):
            raise AssertionError("a cutout must not go through background removal")

    rgba = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    for x in range(30, 50):
        for y in range(20, 60):
            rgba.putpixel((x, y), (120, 120, 120, 255))
    cutout = (
        rgba.convert("LA")
        if mode == "LA"
        else rgba.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE)
    )
    if mode == "P":
        # Mark the background colour transparent, as a palette PNG stores it.
        background = cutout.getpixel((0, 0))
        cutout.info["transparency"] = background
    prepared = preprocess_image(cutout, Untouchable(), image_size=64, crop_margin=1.2)
    assert prepared.size == (64, 64)
    assert np.array(prepared)[32, 32].max() > 0


def test_compiled_sampling_settings_stay_per_model():
    from libreyolo.models.gen.speridlabs_idol.model import _tf32_matmuls
    from libreyolo.models.gen.speridlabs_idol.nn import (
        MultiHeadAttention,
        RopeMultiHeadAttention,
        compile_for_sampling,
    )

    denoiser, _, _ = _tiny_networks()
    other, _, decoder_gs = _tiny_networks()
    compile_for_sampling(denoiser)
    assert all(
        m.cudnn for m in denoiser.modules() if isinstance(m, RopeMultiHeadAttention)
    )
    assert not any(
        m.cudnn for m in other.modules() if isinstance(m, RopeMultiHeadAttention)
    )
    assert not any(
        hasattr(m, "cudnn")
        for m in decoder_gs.modules()
        if isinstance(m, MultiHeadAttention)
    )

    flags = torch.backends.cuda.matmul
    flags.allow_tf32 = False
    with _tf32_matmuls():
        assert flags.allow_tf32 is True
    assert flags.allow_tf32 is False


def test_idol_generations_run_one_at_a_time():
    """Compiled sampling flips a process-wide flag, so no other generation may overlap it."""
    import threading
    import time

    from libreyolo.models.gen.speridlabs_idol.model import _GENERATION_LOCK

    model = _tiny_idol()
    image = Image.new("RGB", (90, 60), (10, 20, 230))
    done = threading.Event()
    worker = threading.Thread(
        target=lambda: (model(image, seed=1, steps=1, gaussians=32_768), done.set())
    )
    with _GENERATION_LOCK:
        worker.start()
        time.sleep(0.5)
        assert not done.is_set()
    worker.join(timeout=60)
    assert done.is_set()
