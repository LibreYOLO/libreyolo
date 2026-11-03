"""Offline unit tests for the TripoSplat image-to-3D family: no weights, no network.

Covers the factory and pins, the knobs, the networks at a tiny size (their
state-dict schema at full size), the sampler's guidance convention and an
end-to-end run through the shared ``generate3d`` contract on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGen, LibreTripoSplat
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.triposplat.nn import (
    FLOW_MODEL_ARGS,
    GS_DECODER_ARGS,
    OCTREE_DECODER_ARGS,
    LatentSeqMMFlowModel,
    OctreeGaussianDecoder,
    sample_probs,
)
from libreyolo.models.gen.triposplat.pipeline import sample_flow

pytestmark = [pytest.mark.unit, pytest.mark.gen]


# ---------------------------------------------------------------------------
# Factory, pins, knobs
# ---------------------------------------------------------------------------


def test_alias_resolves_without_loading():
    assert _ALIASES["triposplat"] == (LibreTripoSplat, "0.6b")
    assert _ALIASES["triposplat-0.6b"] == (LibreTripoSplat, "0.6b")
    model = LibreGen("triposplat", device="cpu")
    assert isinstance(model, LibreTripoSplat)
    assert model.task == "generate3d"
    assert model.model_name == "triposplat-0.6b"
    assert model._pipeline is None
    assert LibreGen("TripoSplat", device="cpu").model_name == "triposplat-0.6b"


def test_pins_every_download():
    from libreyolo.models.gen import image3d

    assert LibreTripoSplat.HF_REPOS["0.6b"] == "VAST-AI/TripoSplat"
    assert (
        LibreTripoSplat.HF_REVISIONS["0.6b"]
        == "56a96e603204ec410c4da60c13ea4fa09a2169a9"
    )
    for revision in (image3d.DINOV3_REVISION, image3d.FLUX2_VAE_REVISION):
        assert len(revision) == 40
    # Only TripoSplat's own networks come from its repo: the DINOv3 and VAE
    # copies there are not downloaded (the encoders are shared with Idol).
    assert set(LibreTripoSplat.SNAPSHOT_ALLOW_PATTERNS) == {
        "diffusion_models/triposplat_fp16.safetensors",
        "vae/triposplat_vae_decoder_fp16.safetensors",
        "background_removal/birefnet.safetensors",
    }
    assert LibreTripoSplat.get_download_url("LibreTripoSplat0.6b") == (
        "https://huggingface.co/VAST-AI/TripoSplat/tree/56a96e603204ec410c4da60c13ea4fa09a2169a9"
    )


def test_matches_its_snapshot_directory(tmp_path):
    assert not LibreTripoSplat.matches_directory(tmp_path)
    for name in LibreTripoSplat.SNAPSHOT_ALLOW_PATTERNS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    assert LibreTripoSplat.matches_directory(tmp_path)
    assert LibreTripoSplat.matches_directory(str(tmp_path))
    assert not LibreTripoSplat.matches_directory(tmp_path / "missing")
    model = LibreGen(str(tmp_path), device="cpu")
    assert isinstance(model, LibreTripoSplat) and model.model_path == str(tmp_path)


def test_defaults_follow_upstream():
    model = LibreTripoSplat(device="cpu")
    assert model._resolve_knobs({}) == {
        "steps": 20,
        "guidance": 3.0,
        "shift": 3.0,
        "gaussians": 262144,
        "margin": 1.2,
    }
    settings = model._resolve_knobs(
        {"gaussians": np.int64(65536), "steps": np.int32(3)}
    )
    assert settings["gaussians"] == 65536 and type(settings["gaussians"]) is int
    assert settings["steps"] == 3 and type(settings["steps"]) is int


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"shift": 0}, "shift must be"),
        ({"shift": float("nan")}, "shift must be"),
        ({"guidance": -1.0}, "guidance must be"),
        ({"guidance": float("inf")}, "guidance must be"),
        ({"margin": float("inf")}, "margin must be"),
        ({"margin": 0.5}, "margin must be"),
        ({"gaussians": 1000}, "gaussians must be"),
        ({"gaussians": 1_048_576}, "gaussians must be"),
        ({"gaussians": 65536.0}, "gaussians must be"),
        ({"steps": 0}, "steps must be"),
        ({"mode": "real"}, "does not accept"),
        ({"erode_radius": 2}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    model = LibreTripoSplat(device="cpu")
    with pytest.raises((ValueError, TypeError), match=match):
        model._resolve_knobs(knobs)


def test_constructor_rejects_idol_only_options():
    with pytest.raises(ValueError, match="compile=True"):
        LibreTripoSplat(device="cpu", compile=True)
    with pytest.raises(ValueError, match="dtype must be 'auto'"):
        LibreTripoSplat(device="cpu", dtype="fp16")
    with pytest.raises(ValueError, match="offload=True"):
        LibreTripoSplat(device="cpu", offload=True)


def test_network_dtype_per_device():
    assert LibreTripoSplat._network_dtype("cpu") is torch.float32
    assert LibreTripoSplat._network_dtype("cuda:0") is torch.float16
    assert LibreTripoSplat._network_dtype("mps") is torch.float16


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_full_size_networks_have_the_published_schema():
    """Tensor counts of the released safetensors (VAST-AI/TripoSplat @ 56a96e6); the names and shapes were
    checked against the files' headers when the port was written."""
    with torch.device("meta"):
        denoiser = LatentSeqMMFlowModel(**FLOW_MODEL_ARGS)
        decoder = OctreeGaussianDecoder(OCTREE_DECODER_ARGS, GS_DECODER_ARGS)
    assert len(denoiser.state_dict()) == 559
    assert sum(p.numel() for p in denoiser.parameters()) == 370_526_101
    assert len(decoder.state_dict()) == 388
    assert sum(p.numel() for p in decoder.parameters()) == 288_053_736
    assert (
        "pos_pe" not in denoiser.state_dict()
    )  # upstream recomputes it from Sobol seed 123
    assert decoder.gs.out_proj.out_features == 32 * (3 + 3 + 3 + 4 + 1 + 1)
    assert decoder.gaussians_per_point == 32


TINY_FLOW = dict(
    q_token_length=16,
    in_channels=4,
    cam_channels=5,
    out_channels=4,
    model_channels=24,
    cond_channels=8,
    cond2_channels=6,
    num_refiner_blocks=2,
    num_blocks=2,
    num_heads=2,
    num_head_channels=12,
    mlp_ratio=2,
    qk_rms_norm=True,
    share_mod=True,
    use_shift_table=True,
)
TINY_OCTREE = dict(
    model_channels=24,
    cond_channels=4,
    num_blocks=1,
    num_heads=2,
    mlp_ratio=2,
    share_mod=True,
)
TINY_GS = dict(
    in_channels=3,
    model_channels=24,
    cond_channels=4,
    num_blocks=1,
    num_heads=2,
    mlp_ratio=2,
    use_learned_offset_scale=True,
    use_per_offset=True,
    representation_config=dict(
        lr=dict(_xyz=1.0, _features_dc=1.0, _opacity=1.0, _scaling=1.0, _rotation=0.1),
        perturb_offset=True,
        perturbe_size=1.5,
        offset_scale=0.05,
        num_gaussians=4,
        filter_kernel_size_3d=0.0009,
        scaling_bias=0.004,
        opacity_bias=0.1,
        scaling_activation="softplus",
    ),
)


def _tiny_networks():
    torch.manual_seed(0)
    denoiser = LatentSeqMMFlowModel(**TINY_FLOW).eval()
    decoder = OctreeGaussianDecoder(TINY_OCTREE, TINY_GS).eval()
    return denoiser, decoder


def test_sobol_positions_are_deterministic():
    a = LatentSeqMMFlowModel(**TINY_FLOW).pos_pe
    b = LatentSeqMMFlowModel(**TINY_FLOW).pos_pe
    assert a.shape == (1, 16, 3) and torch.equal(a, b)
    assert a.min() >= 0 and a.max() <= 1


def test_sample_probs_keeps_every_point():
    probs = torch.softmax(torch.randn(2, 5, 8), dim=-1)
    counts = torch.tensor([[7, 0, 3, 3, 100], [1, 2, 3, 4, 5]])
    first = sample_probs(probs, counts, torch.Generator().manual_seed(4))
    again = sample_probs(probs, counts, torch.Generator().manual_seed(4))
    assert torch.equal(first, again)
    assert torch.equal(first.sum(-1), counts)


def test_decoder_places_exactly_the_requested_gaussians():
    _, decoder = _tiny_networks()
    latent = torch.randn(1, 16, 4)
    splats = decoder.decode(
        latent,
        num_gaussians=37 * 4,
        level=3,
        generator=torch.Generator().manual_seed(0),
    )
    assert splats["xyz"].shape == (37 * 4, 3)
    assert splats["xyz"].dtype == torch.float32
    # anchors lie in the cube; the learned offsets may push a Gaussian a little past it
    assert torch.isfinite(splats["xyz"]).all() and splats["xyz"].abs().max() < 1.0
    assert splats["features_dc"].shape == (37 * 4, 1, 3)
    assert torch.allclose(splats["rotation"].norm(dim=1), torch.ones(37 * 4), atol=1e-5)
    assert (splats["opacity"] > 0).all() and (splats["opacity"] < 1).all()
    assert (splats["scaling"] > 0).all()
    again = decoder.decode(
        latent,
        num_gaussians=37 * 4,
        level=3,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(splats["xyz"], again["xyz"])


def test_decoder_rounds_the_count_down_to_the_anchors():
    _, decoder = _tiny_networks()
    latent = torch.randn(1, 16, 4)
    splats = decoder.decode(
        latent, num_gaussians=10, level=2, generator=torch.Generator().manual_seed(0)
    )
    assert splats["xyz"].shape == (2 * 4, 3)


def test_guidance_at_or_below_one_runs_the_conditional_pass_only():
    denoiser, _ = _tiny_networks()
    calls = []
    forward = denoiser.forward

    def counting(z, cam, t, cond):
        calls.append(float(cond["feature1"].abs().sum()))
        return forward(z, cam, t, cond)

    denoiser.forward = counting
    g = torch.Generator().manual_seed(0)
    latent, camera = (
        torch.randn(1, 16, 4, generator=g),
        torch.randn(1, 1, 5, generator=g),
    )
    cond = {
        "feature1": torch.randn(1, 7, 8, generator=g),
        "feature2": torch.randn(1, 7, 6, generator=g),
    }
    sample_flow(denoiser, latent, camera, cond, steps=3, guidance=1.0)
    assert len(calls) == 3 and all(call > 0 for call in calls)
    calls.clear()
    sample_flow(denoiser, latent, camera, cond, steps=3, guidance=3.0)
    assert len(calls) == 6 and calls[1::2] == [0.0, 0.0, 0.0]  # the zeroed condition


def test_guidance_blends_like_upstream():
    """guided = s * v(cond) - (s - 1) * v(uncond), diffusers' convention, over one Euler step."""
    denoiser, _ = _tiny_networks()
    g = torch.Generator().manual_seed(1)
    latent, camera = (
        torch.randn(1, 16, 4, generator=g),
        torch.randn(1, 1, 5, generator=g),
    )
    cond = {
        "feature1": torch.randn(1, 7, 8, generator=g),
        "feature2": torch.randn(1, 7, 6, generator=g),
    }
    neg = {name: torch.zeros_like(value) for name, value in cond.items()}
    times = torch.tensor([1000.0])
    with torch.no_grad():
        v_c, _ = denoiser(latent, camera, times, cond)
        v_u, _ = denoiser(latent, camera, times, neg)
    expected = (
        latent - (3.0 * v_c - 2.0 * v_u) * 1.0
    )  # shift=1: one step from t=1 to t=0
    out, _ = sample_flow(
        denoiser, latent, camera, cond, steps=1, shift=1.0, guidance=3.0
    )
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# End to end at a tiny size
# ---------------------------------------------------------------------------


class _CentreBlob(torch.nn.Module):
    """Stands in for BiRefNet: logits of a centred square."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

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

    def to(self, device):
        return self


def _tiny_model():
    from libreyolo.models.gen.triposplat.model import _TripoSplatModules

    model = LibreTripoSplat(device="cpu")
    model.input_size = 64
    denoiser, decoder = _tiny_networks()
    model._pipeline = _TripoSplatModules(
        denoiser, decoder, _CentreBlob(), _FakeEncoder()
    )
    return model


def test_end_to_end_at_a_tiny_size(tmp_path, monkeypatch):
    from libreyolo.models.gen.triposplat import model as module

    monkeypatch.setattr(module, "MAX_VOXEL_LEVEL", 3, raising=False)
    model = _tiny_model()
    image = Image.new("RGB", (90, 60), (10, 20, 230))
    result = model(image, seed=11, steps=2, gaussians=32_768)
    g = result.gaussians
    assert len(g) == 32_768
    assert g.seed == 11
    assert g.params["steps"] == 2 and g.params["model"] == "triposplat-0.6b"
    assert g.params["guidance"] == 3.0 and g.params["margin"] == 1.2
    assert set(g.camera) == {"yaw", "pitch", "radius", "fov", "margin"}
    assert g.input_image.size == (64, 64)
    assert g.xyz.dtype == torch.float32 and g.rotations.dtype == torch.float32
    assert torch.allclose(g.rotations.norm(dim=1), torch.ones(len(g)), atol=1e-5)
    again = model(image, seed=11, steps=2, gaussians=32_768).gaussians
    assert torch.equal(g.xyz, again.xyz) and g.camera == again.camera
    other = model(image, seed=12, steps=2, gaussians=32_768).gaussians
    assert not torch.equal(g.xyz, other.xyz)
    pair = model(image, seed=11, n=2, steps=2, gaussians=32_768)
    assert torch.equal(pair[0].gaussians.xyz, g.xyz)
    result.save(tmp_path / "tiny.ply")
    assert (tmp_path / "tiny.ply").stat().st_size > 32_768 * 4 * 14
    (tmp_path / "cut.png").parent.mkdir(exist_ok=True)
    rgba = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    for x in range(30, 50):
        for y in range(20, 60):
            rgba.putpixel((x, y), (120, 120, 120, 255))
    cutout = model(rgba, seed=1, steps=1, gaussians=32_768, margin=2.0).gaussians
    assert cutout.camera["margin"] == 2.0


def test_loads_birefnet_with_or_without_training_heads(tmp_path):
    from libreyolo.models.birefnet.nn import LibreBiRefNetModel
    from libreyolo.models.gen.triposplat.model import _BIREFNET_TRAINING_ONLY

    state = LibreBiRefNetModel(size="t").state_dict()
    dropped = [key for key in state if key.startswith(_BIREFNET_TRAINING_ONLY)]
    assert dropped
    module = LibreBiRefNetModel(size="t")
    missing, unexpected = module.load_state_dict(
        {k: v for k, v in state.items() if k not in dropped}, strict=False
    )
    assert sorted(missing) == sorted(dropped) and not unexpected
    missing, unexpected = LibreBiRefNetModel(size="t").load_state_dict(
        state, strict=True
    )
    assert not missing and not unexpected
