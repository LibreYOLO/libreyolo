"""Offline unit tests for the TRELLIS.2 image-to-3D family: no weights, no network.

The new sparse pieces are checked against plain references (a brute-force
dictionary lookup for the dual-grid meshing and the neighbour search, an
explicit loop for the channel-to-space upsampling, a complex-free rotation
for RoPE); the networks and the pipeline run end to end at a tiny size, on
CPU and, when available, on Apple MPS.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGen, LibreTRELLIS2, Meshes
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.trellis import sparse as ts
from libreyolo.models.gen.trellis2 import mesh as tmesh
from libreyolo.models.gen.trellis2 import nn as t2nn
from libreyolo.models.gen.trellis2 import pipeline as t2pipe

pytestmark = [pytest.mark.unit, pytest.mark.gen]


# ---------------------------------------------------------------------------
# Factory, knobs, snapshot
# ---------------------------------------------------------------------------


def test_aliases_resolve_without_loading():
    for alias in ("trellis2", "trellis2-4b", "trellis.2", "trellis.2-4b"):
        assert _ALIASES[alias] == (LibreTRELLIS2, "4b")
    model = LibreGen("TRELLIS.2", device="cpu")
    assert isinstance(model, LibreTRELLIS2)
    assert model.task == "generate3d" and model.model_name == "trellis2-4b"
    assert model._pipeline is None


def test_defaults_follow_upstream():
    model = LibreTRELLIS2(device="cpu")
    assert model._resolve_knobs({}) == {
        "steps": 12,
        "guidance": 7.5,
        "resolution": 1024,
        "margin": 1.0,
        "max_tokens": 49152,
    }
    assert model._resolve_knobs({"steps": np.int64(6)})["steps"] == 6


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"steps": 0}, "steps must be"),
        ({"guidance": -1.0}, "guidance must be"),
        ({"guidance": float("nan")}, "guidance must be"),
        ({"resolution": 768}, "resolution must be"),
        ({"resolution": True}, "resolution must be"),
        ({"margin": 0.5}, "margin must be"),
        ({"max_tokens": 0}, "max_tokens must be"),
        ({"gaussians": 1000}, "does not accept"),
        ({"shift": 3.0}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        LibreTRELLIS2(device="cpu")._resolve_knobs(knobs)


def test_constructor_and_save_formats():
    with pytest.raises(ValueError, match="no compiled sampler"):
        LibreTRELLIS2(device="cpu", compile=True)
    model = LibreTRELLIS2(device="cpu")
    with pytest.raises(ValueError, match=r"needs a \.obj extension"):
        model(_image(), save=True, output_path="object.ply")


def test_matches_its_snapshot_directory(tmp_path):
    from libreyolo.models.gen.trellis2.model import _UPSTREAM_FILES

    for name in _UPSTREAM_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"name": "Trellis2ImageTo3DPipeline", "args": {}})
    )
    assert LibreTRELLIS2.matches_directory(tmp_path)
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"name": "TrellisImageTo3DPipeline", "args": {}})
    )
    assert not LibreTRELLIS2.matches_directory(tmp_path)


# ---------------------------------------------------------------------------
# Sparse operations and meshing
# ---------------------------------------------------------------------------


def _coords(batch=1, side=10, count=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(batch):
        xyz = torch.unique(torch.randint(0, side, (count, 3), generator=g), dim=0)
        rows.append(torch.cat([torch.full((xyz.shape[0], 1), b), xyz], dim=1))
    return torch.cat(rows).int()


def test_sorted_neighbour_lookup_equals_the_dense_grid(monkeypatch):
    coords = _coords(batch=2)
    dense = ts._neighbour_rows(coords, 2)
    monkeypatch.setattr(ts, "_DENSE_LOOKUP_CELLS", 0)
    assert torch.equal(ts._neighbour_rows(coords, 2), dense)


def _reference_mesh(coords, dual, intersected, weight, resolution):
    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    vertices = (coords.float() + dual) * (1.0 / resolution) - 0.5
    faces = []
    for i, c in enumerate(coords.tolist()):
        for axis in range(3):
            if not intersected[i, axis]:
                continue
            quad = []
            for offset in tmesh._EDGE_NEIGHBOURS[axis]:
                quad.append(index.get(tuple(a + b for a, b in zip(c, offset))))
            if None in quad:
                continue
            w = weight[quad, 0]
            split = tmesh._SPLIT_02 if w[0] * w[2] > w[1] * w[3] else tmesh._SPLIT_13
            faces += [quad[k] for k in split]
    return vertices, torch.tensor(faces, dtype=torch.long).reshape(-1, 3)


def test_dual_grid_mesh_matches_a_dictionary_lookup():
    g = torch.Generator().manual_seed(1)
    coords = _coords(side=8, count=400)[:, 1:]
    n = coords.shape[0]
    dual = torch.rand(n, 3, generator=g)
    intersected = torch.rand(n, 3, generator=g) > 0.4
    weight = torch.rand(n, 1, generator=g) + 0.1
    vertices, faces = tmesh.flexible_dual_grid_to_mesh(
        coords, dual, intersected, weight, 8
    )
    ref_vertices, ref_faces = _reference_mesh(coords, dual, intersected, weight, 8)
    assert torch.equal(vertices, ref_vertices)
    assert faces.shape[0] > 0 and torch.equal(faces, ref_faces)


def test_dual_grid_ignores_neighbours_outside_the_grid():
    # The x-edge quad of (0, 3, 3) would need (0, 4, 4) etc.: outside a 4^3 grid.
    coords = torch.tensor([[0, 3, 3], [0, 0, 0]], dtype=torch.int32)
    intersected = torch.tensor([[True, True, True], [False, False, False]])
    vertices, faces = tmesh.flexible_dual_grid_to_mesh(
        coords, torch.full((2, 3), 0.5), intersected, torch.ones(2, 1), 4
    )
    assert vertices.shape == (2, 3) and faces.shape == (0, 3)


def test_channel_to_spatial_matches_a_loop():
    coords = _coords(batch=2, side=5, count=20)
    x = ts.SparseTensor(torch.zeros(coords.shape[0], 1), coords)
    sub = torch.rand(coords.shape[0], 8, generator=torch.Generator().manual_seed(2)) > 0.5
    child_coords, rows = t2nn.channel_to_spatial(x, sub)
    expected_coords, expected_rows = [], []
    for i, c in enumerate(coords.tolist()):
        for k in range(8):
            if sub[i, k]:
                expected_coords.append(
                    [c[0], 2 * c[1] + (k & 1), 2 * c[2] + (k >> 1 & 1), 2 * c[3] + (k >> 2 & 1)]
                )
                expected_rows.append(i * 8 + k)
    assert child_coords.tolist() == expected_coords
    assert rows.tolist() == expected_rows


def test_rope_rotates_pairs_like_the_real_formula():
    positions = torch.randint(0, 16, (10, 3))
    phases = t2nn.rope_phases(positions, 28)
    assert phases.shape == (10, 14) and torch.allclose(phases.abs(), torch.ones(10, 14))
    # 3 axes x 4 frequencies rotate 12 pairs; the last 2 pairs stay put.
    x = torch.randn(10, 2, 28)
    rotated = t2nn.apply_rope(x, phases)
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = phases.real.unsqueeze(1), phases.imag.unsqueeze(1)
    expected = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).reshape(x.shape)
    assert torch.allclose(rotated, expected, atol=1e-6)
    assert torch.equal(rotated[..., 24:], x[..., 24:])


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class _Constant(torch.nn.Module):
    """Velocity cond * 2 for the conditional pass, x for the unconditional one."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, cond):
        zero = bool(cond.abs().sum() == 0)
        self.calls.append((float(t[0]), zero))
        feats = x.feats if isinstance(x, ts.SparseTensor) else x
        out = feats.clone() if zero else torch.full_like(feats, 2.0) + 0.1 * feats
        return x.replace(out) if isinstance(x, ts.SparseTensor) else out


def test_sampler_follows_upstream_guidance_and_rescale():
    model = _Constant()
    noise = torch.randn(1, 2, 3, generator=torch.Generator().manual_seed(0))
    cond = torch.ones(1, 4, 5)
    out = t2pipe.sample_flow(
        model,
        noise,
        cond,
        steps=3,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 1.0),
        rescale_t=5.0,
    )
    # Upstream's arithmetic, written out.
    x = noise.clone()
    t_seq = np.linspace(1, 0, 4)
    t_seq = (5.0 * t_seq / (1 + 4.0 * t_seq)).tolist()
    s = 1e-5
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        pos = torch.full_like(x, 2.0) + 0.1 * x
        if 0.6 <= t <= 1.0:
            neg = x.clone()
            pred = 7.5 * pos + (1 - 7.5) * neg
            x0_pos = (1 - s) * x - (s + (1 - s) * t) * pos
            x0_cfg = (1 - s) * x - (s + (1 - s) * t) * pred
            ratio = x0_pos.std(dim=[1, 2], keepdim=True) / x0_cfg.std(
                dim=[1, 2], keepdim=True
            )
            x0 = 0.7 * (x0_cfg * ratio) + 0.3 * x0_cfg
            pred = ((1 - s) * x - x0) / (s + (1 - s) * t)
        else:
            pred = pos
        x = x - (t - t_prev) * pred
    assert torch.equal(out, x)
    unconditional = [t for t, zero in model.calls if zero]
    assert len(unconditional) == sum(0.6 <= t <= 1.0 for t in t_seq[:-1]) >= 1


def test_sparse_std_is_the_population_std_per_object():
    coords = _coords(batch=2, side=6, count=40)
    feats = torch.randn(coords.shape[0], 3)
    std = t2pipe._std(ts.SparseTensor(feats, coords))
    x = ts.SparseTensor(feats, coords)
    for i, s in enumerate(x.layout):
        assert torch.allclose(std[i, 0], feats[s].std(unbiased=False), atol=1e-5)


# ---------------------------------------------------------------------------
# Tiny networks end to end
# ---------------------------------------------------------------------------

LATENT = 4
TINY = {
    "flow": {
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 48,
        "cond_channels": 16,
        "num_blocks": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "pe_mode": "rope",
        "share_mod": True,
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
    },
    "decoder": {
        "resolution": 32,
        "model_channels": [16, 16, 8, 8, 8],
        "latent_channels": LATENT,
        "num_blocks": [1, 1, 1, 1, 0],
        "block_type": ["SparseConvNeXtBlock3d"] * 5,
        "up_block_type": ["SparseResBlockC2S3d"] * 4,
        "block_args": [{}, {}, {}, {}, {}],
    },
}


class _Blob(torch.nn.Module):
    """Occupancy decoder stand-in: a 4^3 block of the 64^3 grid, so 8 voxels at 32^3."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, z):
        logits = torch.full((z.shape[0], 1, 64, 64, 64), -1.0, device=z.device)
        logits[:, :, 30:34, 30:34, 30:34] = 1.0
        return logits + self.anchor


class _CentreBlob(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.full((x.shape[0], 1, *x.shape[2:]), -20.0, device=x.device)
        h, w = x.shape[2:]
        logits[:, :, h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = 20.0
        return logits + self.anchor


class _FakeDINOv3(torch.nn.Module):
    """Patch features of the image: embeddings, rotary table, blocks, like transformers' DINOv3."""

    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 16, 64, 64)
        self.embeddings = _Embed(self.patch)
        self.rope_embeddings = lambda pixels: None
        self.layer = torch.nn.ModuleList([_Block()])


class _Embed(torch.nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.patch_embeddings = patch

    def forward(self, pixels, bool_masked_pos=None):
        return self.patch_embeddings(pixels).flatten(2).transpose(1, 2)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, hidden, position_embeddings=None):
        return hidden + self.proj(hidden)


def _tiny_modules(device="cpu", reduced=False):
    from libreyolo.models.gen.trellis2.model import _TRELLIS2Modules

    torch.manual_seed(0)
    dtype = "bfloat16" if reduced else "float32"
    flows = {
        "structure_flow": t2nn.SparseStructureFlowModel(
            resolution=16, dtype=dtype, **TINY["flow"]
        ),
        "shape_flow_512": t2nn.SLatFlowModel(
            resolution=32,
            dtype=dtype,
            **{**TINY["flow"], "in_channels": LATENT, "out_channels": LATENT},
        ),
        "shape_flow_1024": t2nn.SLatFlowModel(
            resolution=64,
            dtype=dtype,
            **{**TINY["flow"], "in_channels": LATENT, "out_channels": LATENT},
        ),
    }
    decoder = t2nn.FlexiDualGridVaeDecoder(**TINY["decoder"], use_fp16=reduced)
    with torch.no_grad():
        # Keep every child voxel, and cross every edge, so the tiny object has a surface.
        for stage in decoder.blocks[:-1]:
            stage[-1].to_subdiv.bias.fill_(5.0)
        decoder.output_layer.bias[3:6].fill_(5.0)
    nets = {**flows, "shape_decoder": decoder, "structure_decoder": _Blob()}
    nets = {name: net.eval().to(device) for name, net in nets.items()}
    sampler = {
        "guidance_rescale": 0.5,
        "guidance_interval": [0.6, 1.0],
        "rescale_t": 3.0,
    }
    return _TRELLIS2Modules(
        device=torch.device(device),
        dinov3=_FakeDINOv3().to(device),
        birefnet=_CentreBlob().to(device),
        samplers={"structure": sampler, "shape": sampler},
        shape_normalization={"mean": [0.0] * LATENT, "std": [1.0] * LATENT},
        **nets,
    )


def _stage(sampler):
    return t2pipe.sampler_settings(sampler, steps=2, guidance=3.0)


def _image():
    return Image.new("RGB", (90, 60), (10, 200, 30))


def test_preprocess_crops_and_premultiplies():
    prepared = t2pipe.preprocess_image(_image(), _CentreBlob())
    assert prepared.mode == "RGB" and prepared.size[0] == prepared.size[1]
    pixels = np.array(prepared)
    centre = prepared.size[0] // 2
    assert pixels[centre, centre].tolist() == [10, 200, 30]
    rgba = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="No object found"):
        t2pipe.preprocess_image(rgba, _CentreBlob())


def test_encode_image_takes_the_last_blocks_prenorm_tokens():
    transformers = pytest.importorskip("transformers")
    config = transformers.DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=16,
        num_register_tokens=4,
    )
    torch.manual_seed(0)
    dinov3 = transformers.DINOv3ViTModel(config).eval()
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    )
    tokens = t2pipe.encode_image(dinov3, image, 64, "cpu")
    assert tokens.shape == (1, 1 + 4 + 16, 32)
    pixels = np.array(image.resize((64, 64), Image.Resampling.LANCZOS)) / 255.0
    pixels = torch.from_numpy(pixels).float().permute(2, 0, 1)[None]
    mean = torch.tensor(t2pipe.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(t2pipe.IMAGENET_STD).view(1, 3, 1, 1)
    hidden = dinov3((pixels - mean) / std, output_hidden_states=True).hidden_states[-1]
    expected = torch.nn.functional.layer_norm(hidden, hidden.shape[-1:])
    assert torch.allclose(tokens, expected, atol=1e-5)


@pytest.mark.parametrize("resolution", [512, 1024])
def test_tiny_pipeline_end_to_end(resolution):
    modules = _tiny_modules()
    prepared = t2pipe.preprocess_image(_image(), modules.birefnet)
    structure = _stage(modules.samplers["structure"])
    out = t2pipe.run_stages(
        modules,
        prepared,
        7,
        resolution=resolution,
        structure=structure,
        shape=structure,
    )
    assert out["coords"].shape[0] == 8 and (out["coords"][:, 1:] < 32).all()
    assert out["resolution"] == resolution
    vertices, faces = out["vertices"], out["faces"]
    assert faces.shape[0] > 0 and faces.max() < vertices.shape[0]
    assert vertices.abs().max() <= 0.5 + 1 / resolution
    if resolution == 1024:
        # The fine stage runs on the 512 latent's predicted voxels, regridded at 64^3.
        assert out["cascade_coords"].shape[0] == 8 * 8
    again = t2pipe.run_stages(
        modules, prepared, 7, resolution=resolution, structure=structure, shape=structure
    )
    assert torch.equal(vertices, again["vertices"]) and torch.equal(faces, again["faces"])


def test_cascade_lowers_the_resolution_to_fit_the_token_budget():
    # A 512 x 512 slab: (resolution / 16)^2 voxels on the fine grid.
    xy = torch.stack(torch.meshgrid(*[torch.arange(512)] * 2, indexing="ij"), -1)
    xy = xy.reshape(-1, 2)
    hr = torch.cat([torch.zeros(xy.shape[0], 1, dtype=xy.dtype), xy, torch.zeros_like(xy[:, :1])], 1)
    coords, resolution = t2pipe._cascade_coords(hr.int(), 512, 1536, max_tokens=7000)
    assert resolution == 1280 and coords.shape[0] == 80 * 80
    coords, resolution = t2pipe._cascade_coords(hr.int(), 512, 1536, max_tokens=1)
    assert resolution == 1024  # never below the 1024 model's grid


def test_generate_through_the_public_call(tmp_path):
    model = LibreTRELLIS2(device="cpu")
    model._pipeline = _tiny_modules()
    calls = []
    encode = t2pipe.encode_image

    def counted(*args, **kwargs):
        calls.append(args[2])
        return encode(*args, **kwargs)

    t2pipe.encode_image = counted
    try:
        pair = model(_image(), seed=3, n=2, steps=2)
    finally:
        t2pipe.encode_image = encode
    assert len(pair) == 2 and sorted(calls) == [512, 1024]  # one encoding per size
    result = model(_image(), seed=3, steps=2)
    mesh = result.meshes
    assert result.gaussians is None and isinstance(mesh, Meshes)
    assert len(mesh) == 1 and mesh.body_model == "object"
    assert mesh.global_orient.shape == (1, 0) and mesh.transl.shape == (1, 3)
    assert torch.equal(mesh.vertices, pair[0].meshes.vertices)
    assert mesh.vertices.shape[0] == 1 and mesh.faces.shape[1] == 3
    assert mesh.seed == 3 and mesh.cpu().seed == 3
    assert mesh.generation_params["model"] == "trellis2-4b" and mesh.generation_params["resolution"] == 1024
    assert mesh.generation_params["output_resolution"] == 1024
    row = result.summary()[0]
    assert row["num_faces"] == mesh.faces.shape[0] and row["seed"] == 3
    result.save(tmp_path / "object.obj")
    lines = (tmp_path / "object.obj").read_text().splitlines()
    assert sum(line.startswith("f ") for line in lines) == mesh.faces.shape[0]
    saved = model(_image(), seed=3, steps=2, save=True)
    assert model._save_dir is not None and any(model._save_dir.glob("*_3.obj"))
    assert saved.meshes.seed == 3


def _mps_ok() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.ones(2, device="mps").to(torch.bfloat16).sum().item()
    except (RuntimeError, TypeError):
        return False
    return True


@pytest.mark.skipif(not _mps_ok(), reason="needs an Apple GPU with bf16 on MPS")
def test_tiny_pipeline_runs_on_mps_like_on_cpu():
    prepared = t2pipe.preprocess_image(_image(), _CentreBlob())
    stage = _stage(
        {"guidance_rescale": 0.5, "guidance_interval": [0.6, 1.0], "rescale_t": 3.0}
    )
    kwargs = dict(resolution=512, structure=stage, shape=stage, stop_after="latent")
    cpu = t2pipe.run_stages(_tiny_modules("cpu"), prepared, 5, **kwargs)
    mps = t2pipe.run_stages(_tiny_modules("mps"), prepared, 5, **kwargs)
    assert torch.allclose(
        cpu["structure_latent"], mps["structure_latent"].cpu(), atol=1e-4
    )
    assert torch.allclose(
        cpu["shape_latent"].feats, mps["shape_latent"].feats.cpu(), atol=1e-3
    )
    full = t2pipe.run_stages(
        _tiny_modules("mps", reduced=True),
        prepared,
        5,
        resolution=1024,
        structure=stage,
        shape=stage,
    )
    assert torch.isfinite(full["vertices"]).all() and full["faces"].shape[0] > 0
