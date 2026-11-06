"""Offline unit tests for the LGM image-to-3D family: no weights, no network.

The cameras, rays, recentring and frame conversion are checked against their
definitions, the attention against a written-out softmax, and the two stages
run end to end with tiny random networks on CPU and, when available, on
Apple MPS.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGen, LibreLGM
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.lgm import nn as lnn
from libreyolo.models.gen.lgm import pipeline as lpipe

pytestmark = [pytest.mark.unit, pytest.mark.gen]

# ---------------------------------------------------------------------------
# Factory, knobs, snapshot
# ---------------------------------------------------------------------------


def test_alias_resolves_without_loading():
    assert _ALIASES["lgm"] == (LibreLGM, "0.4b")
    assert _ALIASES["lgm-0.4b"] == (LibreLGM, "0.4b")
    model = LibreGen("lgm", device="cpu")
    assert isinstance(model, LibreLGM)
    assert model.task == "generate3d" and model.model_name == "lgm-0.4b"
    assert model._pipeline is None


def test_defaults_follow_upstream():
    model = LibreLGM(device="cpu")
    assert model._resolve_knobs({}) == {"steps": 30, "guidance": 5.0, "elevation": 0.0}
    assert model._resolve_knobs({"steps": np.int64(12)})["steps"] == 12


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"steps": 0}, "steps must be"),
        ({"steps": 2.5}, "steps must be"),
        ({"guidance": 0.5}, "guidance must be"),
        ({"guidance": float("nan")}, "guidance must be"),
        ({"elevation": 90}, "elevation must be"),
        ({"gaussians": 1000}, "does not accept"),
        ({"margin": 1.2}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        LibreLGM(device="cpu")._resolve_knobs(knobs)


def test_constructor_rejects_unsupported_options():
    with pytest.raises(ValueError, match="compile=True"):
        LibreLGM(device="cpu", compile=True)
    with pytest.raises(ValueError, match="dtype must be 'auto'"):
        LibreLGM(device="cpu", dtype="fp16")


def test_matches_its_snapshot_directory(tmp_path):
    assert not LibreLGM.matches_directory(tmp_path)
    (tmp_path / "model_fp16_fixrot.safetensors").write_bytes(b"")
    assert LibreLGM.matches_directory(tmp_path)
    assert isinstance(LibreGen(str(tmp_path), device="cpu"), LibreLGM)


def test_pins_every_download():
    from libreyolo.models.gen.lgm import model as module

    assert LibreLGM.HF_REPOS["0.4b"] == "LibreYOLO/LibreLGM0.4b"
    assert LibreLGM.HF_REVISIONS["0.4b"] == "492f8ab3a45e79073df505ebc28f5c3beb479303"
    assert set(LibreLGM.SNAPSHOT_ALLOW_PATTERNS) == {
        "model_fp16_fixrot.safetensors",
        "birefnet.safetensors",
    }
    assert module.IMAGEDREAM_REPO == "LibreYOLO/LibreImageDream"
    assert module.IMAGEDREAM_REVISION == "9076f3a181027f8172ec1abed27faddfeca7f7d5"


# ---------------------------------------------------------------------------
# Cameras, rays, recentring, frame
# ---------------------------------------------------------------------------


def test_orbit_camera_follows_its_convention():
    front = lpipe.orbit_camera(0, 0, radius=1.5)
    assert np.allclose(front[:3, 3], [0, 0, 1.5], atol=1e-6)
    # OpenGL: the camera looks down its -z axis, at the origin.
    assert np.allclose(front[:3, 2], [0, 0, 1], atol=1e-6)
    assert np.allclose(lpipe.orbit_camera(0, 90)[:3, 3], [1, 0, 0], atol=1e-6)
    above = lpipe.orbit_camera(-30, 0)
    assert above[1, 3] > 0  # negative elevation sits above the object
    rotation = front[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)


def test_imagedream_cameras_have_an_empty_input_slot():
    cameras = lpipe.imagedream_cameras(0.0)
    assert cameras.shape == (5, 16) and cameras.dtype == torch.float32
    assert torch.count_nonzero(cameras[4]) == 0
    assert not torch.equal(cameras[0], cameras[1])


def test_rays_start_on_the_orbit_and_are_unit():
    rays = lpipe.ray_embeddings(0.0, size=16)
    assert rays.shape == (4, 6, 16, 16)
    directions = rays[:, 3:]
    assert torch.allclose(directions.norm(dim=1), torch.ones(4, 16, 16), atol=1e-5)
    # The centre ray of view 0 points from +z towards the origin.
    centre = directions[0, :, 7:9, 7:9].mean(dim=(1, 2))
    assert centre[2] < -0.99
    # Plücker moment o x d is orthogonal to d.
    moment = rays[:, :3]
    assert (moment * directions).sum(dim=1).abs().max() < 1e-5


def test_recenter_fits_the_object_into_eighty_percent():
    image = np.zeros((60, 100, 4), dtype=np.uint8)
    image[10:30, 20:40] = (200, 50, 25, 255)
    result = lpipe.recenter(image, image[..., 3] > 0)
    assert result.shape == (100, 100, 4) and result.dtype == np.uint8
    rows, _ = np.nonzero(result[..., 3])
    assert 76 <= rows.max() - rows.min() + 1 <= 81
    assert abs((rows.min() + rows.max()) / 2 - 49.5) <= 1
    assert result[50, 50, :3].tolist() == [200, 50, 25]


class _CentreBlob(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.full((x.shape[0], 1, *x.shape[2:]), -20.0, device=x.device)
        h, w = x.shape[2:]
        logits[:, :, h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = 20.0
        return logits + self.anchor


def _image():
    return Image.new("RGB", (90, 60), (10, 200, 30))


def test_preprocess_recentres_on_white():
    prepared = lpipe.preprocess_image(_image(), _CentreBlob())
    assert prepared.shape == (90, 90, 3) and prepared.dtype == np.float32
    assert np.allclose(prepared[45, 45], np.array([10, 200, 30]) / 255, atol=2e-3)
    assert np.allclose(prepared[0, 0], 1.0)
    cutout = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="No object found"):
        lpipe.preprocess_image(cutout, _CentreBlob())


def test_object_frame_is_z_up_half_size_and_keeps_upstream_pruning():
    g = torch.zeros(1, 3, 14)
    g[0, :, 0:3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    g[0, :, 3] = torch.tensor(
        [0.9, 0.9, 0.001]
    )  # the last is pruned like upstream's ply
    g[0, :, 4:7] = 0.02
    g[0, :, 7] = 1.0  # identity rotation
    g[0, :, 11:14] = 0.5
    out = lpipe.to_object_frame(g)
    assert out["xyz"].tolist() == [[0.5, 0.0, 0.0], [0.0, 0.0, 0.5]]  # LGM's +y is up
    assert torch.allclose(out["scales"], torch.full((2, 3), 0.01))
    assert torch.allclose(out["sh"], torch.zeros(2, 1, 3))
    # A Gaussian stretched along LGM's y stays stretched along the vertical.
    q = out["rotations"][0]
    w, x, y, z = q.tolist()
    local_y = np.array(
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    )
    assert np.allclose(local_y, [0, 0, 1], atol=1e-6)
    camera = lpipe.input_camera(-20.0)
    assert camera["yaw"] == pytest.approx(math.pi)
    assert camera["pitch"] == pytest.approx(math.radians(20))
    assert camera["radius"] == 0.75 and camera["margin"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _reference_attention(q, k, v):
    return ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).softmax(-1) @ v


def test_cross_attention_with_image_tokens_matches_a_written_out_softmax():
    torch.manual_seed(0)
    attn = lnn.CrossAttention(
        32, 24, heads=2, dim_head=16, ip_dim=3, ip_weight=1.0
    ).eval()
    x, context = torch.randn(2, 10, 32), torch.randn(2, 7 + 3, 24)

    def split(t):
        return t.reshape(2, t.shape[1], 2, 16).transpose(1, 2)

    q = split(attn.to_q(x))
    text, image = context[:, :7], context[:, 7:]
    out = _reference_attention(q, split(attn.to_k(text)), split(attn.to_v(text)))
    out = out + _reference_attention(
        q, split(attn.to_k_ip(image)), split(attn.to_v_ip(image))
    )
    expected = attn.to_out(out.transpose(1, 2).reshape(2, 10, 32))
    assert torch.allclose(attn(x, context), expected, atol=1e-5)


def test_multiview_self_attention_mixes_the_views_of_one_object_only():
    torch.manual_seed(0)
    block = lnn.BasicTransformerBlock3D(32, 2, 16, context_dim=24).eval()
    x, context = torch.randn(10, 6, 32), torch.randn(10, 4, 24)
    out = block(x, context, num_frames=5)
    changed = x.clone()
    changed[7] += 1.0  # a view of the second object
    again = block(changed, context, num_frames=5)
    assert torch.equal(out[:5], again[:5])
    assert not torch.allclose(out[5:7], again[5:7])


TINY_UNET = {
    "image_size": 32,
    "in_channels": 4,
    "model_channels": 32,
    "out_channels": 4,
    "num_res_blocks": 1,
    "attention_resolutions": [1, 2],
    "channel_mult": [1, 2],
    "num_head_channels": 16,
    "context_dim": 64,
    "camera_dim": 16,
    "ip_dim": 4,
}
TINY_LGM = {
    "down_channels": (32, 32, 32, 32),
    "down_attention": (False, False, False, True),
    "mid_attention": True,
    "up_channels": (32, 32, 32),
    "up_attention": (True, False, False),
}
SCHEDULER = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "set_alpha_to_one": False,
    "steps_offset": 1,
    "timestep_spacing": "leading",
}


class _FakeImageEncoder(torch.nn.Module):
    """224 px pixels -> ``hidden_states`` whose penultimate entry is ``[B, 49, 1280]``."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 1280)

    def forward(self, pixel_values, output_hidden_states=True):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values, 7)
        tokens = self.proj(pooled.flatten(2).transpose(1, 2))
        return SimpleNamespace(hidden_states=[tokens, tokens + 1])


def _tiny_modules(device="cpu", half=False):
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPImageProcessor

    from libreyolo.models.gen.lgm.model import _LGMModules

    torch.manual_seed(0)
    dtype = torch.float16 if half else torch.float32
    lgm = lnn.LGMNetwork(splat_size=128, **TINY_LGM).eval().to(device)
    if half:
        lgm.half()
    vae = AutoencoderKL(
        block_out_channels=(32, 32, 32, 32),
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        latent_channels=4,
        layers_per_block=1,
        sample_size=256,
    )
    return _LGMModules(
        device=torch.device(device),
        lgm=lgm,
        autocast=half,
        birefnet=_CentreBlob().to(device),
        unet=lnn.MultiViewUNet(**TINY_UNET).eval().to(device, dtype),
        vae=vae.eval().to(device, dtype),
        image_encoder=_FakeImageEncoder().eval().to(device, dtype),
        feature_extractor=CLIPImageProcessor(),
        scheduler=DDIMScheduler(**SCHEDULER),
        text_tokens=torch.randn(1, 6, 64).to(device, dtype),
    )


def test_tiny_pipeline_end_to_end():
    modules = _tiny_modules()
    prepared = lpipe.preprocess_image(_image(), modules.birefnet)
    out = lpipe.run_stages(modules, prepared, 7, steps=2)
    assert out["views"].shape == (4, 256, 256, 3)
    assert out["lgm"].shape == (1, 4 * 128 * 128, 14)
    g = out["gaussians"]
    n = g["xyz"].shape[0]
    assert 0 < n <= 4 * 128 * 128
    assert torch.allclose(g["rotations"].norm(dim=1), torch.ones(n), atol=1e-5)
    assert (g["opacities"] >= lpipe.MIN_OPACITY).all() and (g["scales"] > 0).all()
    assert g["xyz"].abs().max() <= 0.5
    again = lpipe.run_stages(modules, prepared, 7, steps=2)
    assert np.array_equal(out["views"], again["views"])
    other = lpipe.run_stages(modules, prepared, 8, steps=2)
    assert not np.array_equal(out["views"], other["views"])


def test_guidance_one_runs_the_conditional_pass_only():
    modules = _tiny_modules()
    batches = []
    forward = modules.unet.forward
    modules.unet.forward = lambda x, *a, **k: (
        batches.append(x.shape[0]) or forward(x, *a, **k)
    )
    prepared = lpipe.preprocess_image(_image(), modules.birefnet)
    lpipe.sample_views(
        modules,
        prepared,
        torch.Generator().manual_seed(0),
        steps=2,
        guidance=1.0,
        elevation=0.0,
    )
    lpipe.sample_views(
        modules,
        prepared,
        torch.Generator().manual_seed(0),
        steps=2,
        guidance=5.0,
        elevation=0.0,
    )
    assert batches == [5, 5, 10, 10]


def test_generate_through_the_public_call(tmp_path):
    model = LibreLGM(device="cpu")
    model._pipeline = _tiny_modules()
    calls = []
    forward = model._pipeline.image_encoder.forward
    model._pipeline.image_encoder.forward = lambda *a, **k: (
        calls.append(1) or forward(*a, **k)
    )
    pair = model(_image(), seed=3, n=2, steps=2)
    assert len(pair) == 2 and len(calls) == 1  # one CLIP encoding for both seeds
    result = model(_image(), seed=3, steps=2)
    assert torch.equal(result.gaussians.xyz, pair[0].gaussians.xyz)
    g = result.gaussians
    assert g.seed == 3 and g.params["model"] == "lgm-0.4b" and g.params["steps"] == 2
    assert g.camera["fov"] == pytest.approx(49.1)
    assert g.input_image.size == (90, 90)
    result.save(tmp_path / "object.ply")
    assert (tmp_path / "object.ply").stat().st_size > len(g) * 14 * 4
    assert result.plot() is not None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_tiny_pipeline_runs_on_mps():
    prepared = lpipe.preprocess_image(_image(), _CentreBlob())
    mps = lpipe.run_stages(_tiny_modules("mps", half=True), prepared, 5, steps=2)
    g = mps["gaussians"]
    assert g["xyz"].device.type == "mps" and torch.isfinite(g["xyz"]).all()
    assert mps["views"].shape == (4, 256, 256, 3) and np.isfinite(mps["views"]).all()
    # In fp32 the network stage on MPS matches CPU on the same views.
    cpu_modules, mps_modules = _tiny_modules("cpu"), _tiny_modules("mps")
    views = mps["views"]
    a = lpipe.views_to_gaussians(cpu_modules, views, 0.0)
    b = lpipe.views_to_gaussians(mps_modules, views, 0.0).cpu()
    assert torch.allclose(a, b, atol=1e-3)
