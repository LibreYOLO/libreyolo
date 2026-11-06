"""Offline unit tests for the Direct3D-S2 image-to-mesh family: no weights, no network.

The sparse operations are checked against dense PyTorch references (a
zero-filled Conv3d grid for the torchsparse convolution, per-query loops for
the three spatial sparse attention branches, a brute-force distance field for
the mesh voxelisation); the networks and pipeline run end to end at a tiny
size, on CPU and, when available, on Apple MPS.
"""

from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from libreyolo import LibreGen
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.direct3d_s2 import LibreDirect3DS2
from libreyolo.models.gen.direct3d_s2 import model as dmodel
from libreyolo.models.gen.direct3d_s2 import nn as dnn
from libreyolo.models.gen.direct3d_s2 import pipeline as dpipe
from libreyolo.models.gen.direct3d_s2 import sparse as ds

pytestmark = [pytest.mark.unit, pytest.mark.gen]


needs_diffusers = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="the flow-matching scheduler comes from diffusers (gen extra)",
)


def _coords(side=12, count=300, seed=0, batch=1):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(batch):
        xyz = torch.unique(torch.randint(0, side, (count, 3), generator=g), dim=0)
        rows.append(torch.cat([torch.full((xyz.shape[0], 1), b), xyz], dim=1))
    return torch.cat(rows).int()


# ---------------------------------------------------------------------------
# Factory, knobs, snapshot
# ---------------------------------------------------------------------------


def test_alias_resolves_without_loading():
    assert _ALIASES["direct3d-s2"] == (LibreDirect3DS2, "v11")
    assert _ALIASES["direct3d-s2-v1.1"] == (LibreDirect3DS2, "v11")
    model = LibreGen("direct3d-s2", device="cpu")
    assert isinstance(model, LibreDirect3DS2)
    assert model.task == "generate3d" and model.model_name == "direct3d-s2-v11"
    assert model.SAVE_SUFFIXES == (".obj",)
    assert model._pipeline is None


def test_defaults_follow_upstream():
    model = LibreDirect3DS2(device="cpu")
    assert model._resolve_knobs({}) == {
        "resolution": 1024,
        "steps": (50, 30, 15),
        "guidance": 7.0,
    }
    assert model._resolve_knobs({"steps": np.int64(12)})["steps"] == (12, 12, 12)
    assert model._resolve_knobs({"steps": [4, 5, 6], "resolution": 512})["steps"] == (
        4,
        5,
        6,
    )


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"steps": 0}, "steps must be"),
        ({"steps": (1, 2)}, "steps must be"),
        ({"guidance": -1.0}, "guidance must be"),
        ({"guidance": float("nan")}, "guidance must be"),
        ({"resolution": 256}, "resolution must be"),
        ({"resolution": True}, "resolution must be"),
        ({"shift": 3.0}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        LibreDirect3DS2(device="cpu")._resolve_knobs(knobs)


def test_constructor_rejects_unsupported_options():
    with pytest.raises(ValueError, match="compile"):
        LibreDirect3DS2(device="cpu", compile=True)
    with pytest.raises(ValueError, match="dtype"):
        LibreDirect3DS2(device="cpu", dtype="fp16")


def test_matches_its_snapshot_directory(tmp_path):
    assert not LibreDirect3DS2.matches_directory(tmp_path)
    for name in dmodel._UPSTREAM_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    assert LibreDirect3DS2.matches_directory(tmp_path)
    model = LibreGen(str(tmp_path), device="cpu")
    assert isinstance(model, LibreDirect3DS2) and model._checkpoint_dir == tmp_path


def test_pins_every_download():
    import re

    from libreyolo.models.gen.trellis.model import _DINOV2_REVISION

    sha = re.compile(r"^[0-9a-f]{40}$")
    assert all(sha.match(r) for r in LibreDirect3DS2.HF_REVISIONS.values())
    assert sha.match(_DINOV2_REVISION)
    assert sha.match(dmodel._BIREFNET_REVISION)
    assert set(LibreDirect3DS2.SNAPSHOT_ALLOW_PATTERNS) == set(dmodel._UPSTREAM_FILES)


# ---------------------------------------------------------------------------
# Sparse operations
# ---------------------------------------------------------------------------


def test_torchsparse_conv_equals_a_dense_conv_read_at_the_active_voxels():
    torch.manual_seed(0)
    coords = _coords(side=10, batch=2)
    feats = torch.randn(coords.shape[0], 5)
    conv = ds.SparseConv3d(5, 7, 3)
    out = conv(ds.SparseTensor(feats, coords)).feats
    c = coords.long()
    dense = torch.zeros(2, 5, 10, 10, 10)
    dense[c[:, 0], :, c[:, 1], c[:, 2], c[:, 3]] = feats
    weight = torch.zeros(7, 5, 3, 3, 3)
    for j in range(27):  # torchsparse order: x fastest
        weight[:, :, j % 3, j // 3 % 3, j // 9] = conv.conv.kernel[j].t()
    ref = F.conv3d(dense, weight, conv.conv.bias, padding=1)
    ref = ref.permute(0, 2, 3, 4, 1)[c[:, 0], c[:, 1], c[:, 2], c[:, 3]]
    assert torch.allclose(out, ref, atol=1e-5)


def test_pointwise_conv_uses_the_in_out_kernel():
    conv = ds.SparseConv3d(4, 3, 1)
    assert conv.conv.kernel.shape == (4, 3)
    x = ds.SparseTensor(torch.randn(9, 4), _coords(count=9))
    assert torch.allclose(conv(x).feats, x.feats @ conv.conv.kernel + conv.conv.bias)


def test_conv_chunks_give_the_same_rows(monkeypatch):
    torch.manual_seed(1)
    coords = _coords(side=8)
    x = ds.SparseTensor(torch.randn(coords.shape[0], 4), coords)
    conv = ds.SparseConv3d(4, 6, 3)
    whole = conv(x).feats
    monkeypatch.setattr(ds, "_CONV_CHUNK_ROWS", 17)
    x.cache.clear()
    assert torch.allclose(conv(x).feats, whole, atol=1e-6)


def test_mean_pool_counts_the_zero_it_starts_from_like_upstream():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 0, 0], [0, 0, 0, 3]]
    ).int()
    feats = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    pooled = ds.sparse_mean_pool(ds.SparseTensor(feats, coords), 2)
    assert pooled.coords.tolist() == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
    # (0 + 1 + 3) / 3, (0 + 7) / 2, (0 + 5) / 2
    assert torch.allclose(pooled.feats[:, 0], torch.tensor([4 / 3, 3.5, 2.5]))


def test_subdivide_makes_eight_children_z_fastest():
    x = ds.SparseTensor(
        torch.tensor([[1.0], [2.0]]), torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]]).int()
    )
    out = ds.sparse_subdivide(x)
    assert out.coords[:8].tolist() == [
        [0, a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)
    ]
    assert out.coords[8].tolist() == [0, 2, 4, 6]
    assert out.feats[:, 0].tolist() == [1.0] * 8 + [2.0] * 8


def test_sort_block_orders_by_block_then_position():
    coords = torch.tensor([[0, 5, 0, 0], [0, 0, 0, 5], [0, 1, 0, 0], [0, 0, 0, 0]])
    assert ds.sort_block(coords, 4).tolist() == [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 5],
        [0, 5, 0, 0],
    ]


def test_spatial_sparse_attention_matches_per_query_loops():
    torch.manual_seed(0)
    block, heads, groups, channels, topk = 4, 8, 2, 16, 3
    coords = ds.sort_block(_coords(side=16, count=400).long(), block).int()
    n = coords.shape[0]
    x = ds.SparseTensor(torch.zeros(n, 1), coords)
    q = torch.randn(n, heads, channels)
    k = torch.randn(n, groups, channels)
    v = torch.randn(n, groups, channels)
    ck = ds.sparse_mean_pool(ds.SparseTensor(k.reshape(n, -1), coords), block)
    cv = ds.sparse_mean_pool(ds.SparseTensor(v.reshape(n, -1), coords), block)
    ck = ck.replace(ck.feats.view(-1, groups, channels))
    cv = cv.replace(cv.feats.view(-1, groups, channels))
    compressed, selected, windowed = ds.spatial_sparse_attention(
        x,
        q,
        k,
        v,
        ck,
        cv,
        topk=topk,
        selection_block_size=block,
        compression_block_size=block,
        window_size=4,
        shift=2,
    )
    scale = 1 / math.sqrt(channels)
    blocks = coords[:, 1:].long() // block
    block_id = (blocks[:, 0] * 16 + blocks[:, 1]) * 16 + blocks[:, 2]
    compressed_id = (ck.coords[:, 1] * 16 + ck.coords[:, 2]) * 16 + ck.coords[:, 3]
    windows = (coords[:, 1:].long() + 2) // 4
    window_id = (windows[:, 0] * 64 + windows[:, 1]) * 64 + windows[:, 2]
    per_group = heads // groups
    for i in range(0, n, 7):
        for h in range(heads):
            g = h // per_group
            p = (ck.feats[:, g] @ q[i, h] * scale).softmax(0)
            assert torch.allclose(p @ cv.feats[:, g], compressed[i, h], atol=1e-5)
            score = sum(
                (ck.feats[:, g] @ q[i, hh] * scale).softmax(0)
                for hh in range(g * per_group, (g + 1) * per_group)
            )
            chosen = torch.isin(
                block_id, compressed_id[score.topk(topk).indices].long()
            )
            p = (k[chosen, g] @ q[i, h] * scale).softmax(0)
            assert torch.allclose(p @ v[chosen, g], selected[i, h], atol=1e-5)
            mine = window_id == window_id[i]
            p = (k[mine, g] @ q[i, h] * scale).softmax(0)
            assert torch.allclose(p @ v[mine, g], windowed[i, h], atol=1e-5)


def test_attention_chunks_give_the_same_outputs(monkeypatch):
    torch.manual_seed(2)
    coords = ds.sort_block(_coords(side=12, count=200).long(), 4).int()
    n = coords.shape[0]
    q, k, v = torch.randn(n, 4, 8), torch.randn(n, 2, 8), torch.randn(n, 2, 8)
    ck = ds.sparse_mean_pool(ds.SparseTensor(k.reshape(n, -1), coords), 4)
    cv = ds.sparse_mean_pool(ds.SparseTensor(v.reshape(n, -1), coords), 4)
    ck = ck.replace(ck.feats.view(-1, 2, 8))
    cv = cv.replace(cv.feats.view(-1, 2, 8))

    def run():
        x = ds.SparseTensor(torch.zeros(n, 1), coords)
        return ds.spatial_sparse_attention(
            x,
            q,
            k,
            v,
            ck,
            cv,
            topk=2,
            selection_block_size=4,
            compression_block_size=4,
            window_size=4,
            shift=0,
        )

    whole = run()
    monkeypatch.setattr(ds, "_ATTENTION_CHUNK_ELEMENTS", 4096)
    for a, b in zip(whole, run()):
        assert torch.allclose(a, b, atol=1e-6)


def test_selection_needs_block_sorted_voxels():
    coords = torch.tensor([[0, 9, 0, 0], [0, 0, 0, 0]]).int()
    with pytest.raises(ValueError, match="sorted by block"):
        ds.selection_blocks(coords, 4)


# ---------------------------------------------------------------------------
# Meshing helpers
# ---------------------------------------------------------------------------


def test_mesh_voxelisation_matches_a_brute_force_distance_field():
    rng = np.random.default_rng(0)
    vertices = rng.random((30, 3)) * 1.6 - 0.8
    faces = rng.integers(0, 30, (20, 3))
    faces[0] = [3, 3, 7]  # a degenerate edge-triangle
    size = 48
    voxels = dpipe.mesh_to_voxels(
        vertices, faces, "cpu", size=size, threshold=4.0, factor=8
    )
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(size)] * 3, indexing="ij"), -1
    ).view(-1, 3)
    points = grid.float() / (size - 1) - 0.5
    corners = torch.tensor(vertices, dtype=torch.float32)[torch.tensor(faces)] * 0.5
    best = torch.full((points.shape[0],), 1e9)
    for tri in corners:
        d = dpipe._point_triangle_distance(
            points[None], tri[None, 0:1], tri[None, 1:2], tri[None, 2:3]
        )[0]
        best = torch.minimum(best, d)
    expected = torch.unique(grid[best < 4.0 / size] // 8, dim=0)
    assert torch.equal(voxels[:, 1:], expected) and (voxels[:, 0] == 0).all()


def test_point_triangle_distance_is_exact_on_simple_cases():
    tri = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    points = torch.tensor(
        [[[0.25, 0.25, 2.0], [-1.0, 0, 0], [0.5, -3, 0], [2.0, 2.0, 0]]]
    )
    d = dpipe._point_triangle_distance(
        points, tri[None, 0:1], tri[None, 1:2], tri[None, 2:3]
    )[0]
    assert torch.allclose(d, torch.tensor([2.0, 1.0, 3.0, math.sqrt(4.5)]))


def test_normalize_vertices_fits_the_longest_side():
    v = np.array([[0.0, 0, 0], [2, 1, 1]])
    out = dpipe.normalize_vertices(v)
    assert np.allclose(out.max(axis=0) - out.min(axis=0), [1.9, 0.95, 0.95])
    assert np.allclose(out.max(axis=0) + out.min(axis=0), 0)


def test_blend_patches_joins_three_patches_per_axis():
    res, patch = 40, 16  # stride 12, overlap 4
    patches = [torch.full((1, 1, patch, patch, patch), float(i)) for i in range(27)]
    out = dnn.blend_patches(patches, 1, res, patch, "cpu", torch.float32)
    assert out.shape == (1, 1, res, res, res)
    assert out[0, 0, 0, 0, 0] == 0 and out[0, 0, -1, -1, -1] == 26
    assert out[0, 0, 0, 0, 20] == 1  # middle patch along z, outside the overlaps
    assert torch.allclose(out[0, 0, 0, 0, 12:16], torch.linspace(0, 1, 4))
    assert out.min() >= 0 and out.max() <= 26


def test_object_mesh_frame_and_winding():
    vertices = np.array([[1.0, 0.5, -0.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = dmodel.object_mesh(vertices, faces)
    assert mesh.body_model == "object" and len(mesh) == 1
    assert mesh.vertices.shape == (1, 3, 3) and mesh.faces.tolist() == [[0, 1, 2]]
    assert mesh.vertices[0, 0].tolist() == [0.5, 0.25, 0.25]


# ---------------------------------------------------------------------------
# Tiny pipeline
# ---------------------------------------------------------------------------


TINY = {
    "dense_dit": dict(
        resolution=4,
        in_channels=2,
        model_channels=32,
        cond_channels=16,
        out_channels=2,
        num_blocks=1,
        num_heads=2,
        patch_size=1,
        qk_rms_norm=True,
        latent_shape=[2, 4, 4, 4],
    ),
    "dense_vae": dict(
        embed_dim=2,
        model_channels_decoder=[16, 8],
        num_res_blocks_decoder=1,
        num_res_blocks_middle_decoder=1,
    ),
    "sparse_dit_512": dict(
        resolution=8,
        in_channels=4,
        model_channels=32,
        cond_channels=16,
        out_channels=4,
        num_blocks=2,
        num_heads=4,
        num_kv_heads=2,
        compression_block_size=4,
        selection_block_size=4,
        topk=2,
        compression_version="v1",
        window_size=4,
    ),
    "sparse_vae_512": dict(
        embed_dim=4,
        resolution=8,
        model_channels_decoder=64,
        num_blocks_decoder=1,
        num_heads_decoder=4,
    ),
    "sparse_dit_1024": dict(
        resolution=16,
        in_channels=4,
        model_channels=32,
        cond_channels=16,
        out_channels=4,
        num_blocks=2,
        num_heads=4,
        num_kv_heads=2,
        compression_block_size=4,
        selection_block_size=4,
        topk=2,
        compression_version="v1",
        window_size=4,
        factor=0.5,
    ),
    "sparse_vae_1024": dict(
        embed_dim=4,
        resolution=16,
        model_channels_decoder=64,
        num_blocks_decoder=1,
        num_heads_decoder=4,
        chunk_size=2,
    ),
    "refiner": dict(
        res=64, patch_size=24, feat_channels=4, block_out_channels=(8, 8, 8, 8)
    ),
    "refiner_1024": dict(res=64, patch_size=32, block_out_channels=(8, 8, 8, 8)),
}

BUILDERS = {
    "dense_dit": dnn.DenseDiT,
    "dense_vae": dnn.DenseShapeVAE,
    "sparse_dit_512": dnn.SparseDiT,
    "sparse_vae_512": dnn.SparseSDFVAE,
    "sparse_dit_1024": dnn.SparseDiT,
    "sparse_vae_1024": dnn.SparseSDFVAE,
    "refiner": dnn.VoxelRefiner,
    "refiner_1024": dnn.VoxelSignRefiner,
}


class _FakeDINOv2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 16)

    def forward(self, pixel_values, output_hidden_states=False):
        pooled = F.adaptive_avg_pool2d(pixel_values, 7).flatten(2).transpose(1, 2)
        return SimpleNamespace(hidden_states=[self.proj(pooled)])


class _CentreBlob(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = torch.full((x.shape[0], 1, *x.shape[2:]), -20.0, device=x.device)
        h, w = x.shape[2:]
        logits[:, :, h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = 20.0
        return logits + self.anchor


def _sphere(side, radius=0.3):
    axis = torch.linspace(-1, 1, side)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return ((x**2 + y**2 + z**2).sqrt() - radius)[None, None]


def _tiny_modules(device="cpu", use_fp16=False, sphere=True):
    torch.manual_seed(0)
    nets = {
        name: BUILDERS[name](**TINY[name], use_fp16=use_fp16).eval().to(device)
        for name in BUILDERS
    }
    # Bias the occupancy so the random structure keeps a fraction of the grid.
    with torch.no_grad():
        nets["dense_vae"].decoder.out_layer[-1].bias.fill_(-0.5)
    if sphere:
        # Run the real refiners, then mesh a known SDF of the same size.
        for name in ("refiner", "refiner_1024"):
            refine = nets[name].refine

            def wrapped(*args, _refine=refine):
                volume = _refine(*args)
                return _sphere(volume.shape[-1]).to(volume)

            nets[name].refine = wrapped
    dtype = torch.float16 if use_fp16 else torch.float32
    return dmodel._Direct3DS2Modules(
        device=torch.device(device),
        dtype=dtype,
        dinov2=_FakeDINOv2().to(device),
        birefnet=_CentreBlob().to(device),
        shifts=(6.0, 6.0, 8.0),
        **nets,
    )


def _image():
    return Image.new("RGB", (90, 60), (10, 200, 30))


def test_preprocess_crops_and_pads_on_white():
    prepared = dpipe.preprocess_image(_image(), _CentreBlob())
    assert prepared.size == (518, 518) and prepared.mode == "RGB"
    pixels = np.array(prepared)
    assert pixels[259, 259].tolist() == [10, 200, 30]
    assert pixels[2, 2].tolist() == [255, 255, 255]
    cutout = Image.new("RGBA", (40, 80), (0, 0, 0, 0))
    cutout.paste((200, 0, 0, 255), (10, 10, 30, 70))
    pixels = np.array(dpipe.preprocess_image(cutout))
    assert pixels[259, 259].tolist() == [200, 0, 0]
    assert pixels[259, 60].tolist() == [255, 255, 255]  # square padding


@needs_diffusers
def test_sampler_uses_upstream_flow_matching_and_guidance():
    calls = []

    class Recorder(torch.nn.Module):
        def forward(self, x, t, cond):
            calls.append((float(t[0]), bool(cond.abs().sum() == 0)))
            return torch.ones_like(x) * (1.0 if cond.abs().sum() else 0.0)

    out = dpipe.sample(
        Recorder(),
        torch.zeros(1, 2),
        torch.ones(1, 3, 4),
        steps=4,
        guidance=2.0,
        shift=1.0,
    )
    # diffusers' FlowMatchEulerDiscreteScheduler: sigmas from 1 to 1/1000.
    assert [round(t) for t, zero in calls if not zero] == [1000, 667, 334, 1]
    assert len([1 for _, zero in calls if zero]) == 4
    # Each step moves by -dt * (0 + 2 * (1 - 0)); the steps span sigma 1 to 0.
    assert torch.allclose(out, torch.full((1, 2), -2.0))


@needs_diffusers
def test_tiny_pipeline_end_to_end():
    modules = _tiny_modules()
    prepared = dpipe.preprocess_image(_image(), modules.birefnet)
    out = dpipe.run_stages(
        modules,
        prepared,
        torch.Generator().manual_seed(7),
        steps=(2, 2, 2),
        guidance=3.0,
    )
    voxels = out["voxels512"]
    assert voxels.shape[0] > 0 and (voxels[:, 1:] < 8).all()
    assert torch.equal(voxels, ds.sort_block(voxels, 4))
    assert out["sparse512_sdf"].coords[:, 1:].max() < 64
    assert out["voxels1024"].shape[0] > 0 and (out["voxels1024"][:, 1:] < 16).all()
    assert out["sparse1024_sdf"].coords[:, 1:].max() < 128
    vertices, faces = out["mesh"]
    assert vertices.shape[1] == 3 and faces.shape[1] == 3 and len(faces) > 0
    assert np.abs(vertices).max() <= 1.0
    again = dpipe.run_stages(
        modules,
        prepared,
        torch.Generator().manual_seed(7),
        steps=(2, 2, 2),
        guidance=3.0,
    )
    assert torch.equal(again["sparse1024_latent"], out["sparse1024_latent"])
    other = dpipe.run_stages(
        modules,
        prepared,
        torch.Generator().manual_seed(8),
        steps=(2, 2, 2),
        guidance=3.0,
        stop_after="dense",
    )
    assert not torch.equal(other["dense_latent"], out["dense_latent"])


def test_chunked_decoding_covers_the_same_voxels_as_whole_decoding():
    torch.manual_seed(3)
    whole = dnn.SparseSDFVAE(**dict(TINY["sparse_vae_1024"], chunk_size=1)).eval()
    chunked = dnn.SparseSDFVAE(**TINY["sparse_vae_1024"]).eval()
    chunked.load_state_dict(whole.state_dict())
    coords = _coords(side=16, count=150)
    latent = ds.SparseTensor(torch.randn(coords.shape[0], 4), coords)
    with torch.no_grad():
        a = whole.decoder(latent, factor=1.0)
        b = chunked.decoder(ds.SparseTensor(latent.feats, coords), factor=1.0)
    key = lambda c: (c[:, 1].long() * 128 + c[:, 2]) * 128 + c[:, 3]  # noqa: E731
    # Upstream's crop bound (end * 8 + 7) repeats voxels at chunk borders.
    assert b.coords.shape[0] > a.coords.shape[0]
    assert torch.equal(torch.unique(key(a.coords)), torch.unique(key(b.coords)))


@needs_diffusers
def test_generate_through_the_public_call(tmp_path):
    model = LibreDirect3DS2(device="cpu")
    model._pipeline = _tiny_modules()
    calls = []
    forward = model._pipeline.dinov2.forward
    model._pipeline.dinov2.forward = lambda *a, **k: calls.append(1) or forward(*a, **k)
    pair = model(_image(), seed=3, n=2, steps=2, resolution=512)
    assert len(pair) == 2 and len(calls) == 1  # one encoding for both seeds
    result = model(_image(), seed=3, steps=2, resolution=512)
    mesh = result.meshes
    assert result.gaussians is None and mesh.body_model == "object" and len(mesh) == 1
    assert torch.equal(mesh.vertices, pair[0].meshes.vertices)
    assert mesh.vertices.abs().max() <= 0.5
    assert mesh.seed == 3 and mesh.generation_params["model"] == "direct3d-s2-v11"
    assert mesh.generation_params["steps"] == (2, 2, 2)
    result.save(tmp_path / "object.obj")
    text = (tmp_path / "object.obj").read_text()
    assert text.count("\nv ") == mesh.vertices.shape[1]
    with pytest.raises(ValueError, match=".obj"):
        model(
            _image(),
            seed=3,
            steps=2,
            resolution=512,
            save=True,
            output_path=str(tmp_path / "x.ply"),
        )


def _mps_runs_conv3d() -> bool:
    """Real Apple GPUs run Conv3d on MPS; GitHub's virtual macOS GPU cannot compile its Metal kernel."""
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.nn.functional.conv3d(
            torch.ones(1, 4, 4, 4, 4, device="mps", dtype=torch.float16),
            torch.ones(4, 4, 3, 3, 3, device="mps", dtype=torch.float16),
        )
        torch.mps.synchronize()
    except RuntimeError:
        return False
    return True


@pytest.mark.skipif(
    not _mps_runs_conv3d(), reason="needs an Apple GPU that runs Conv3d on MPS"
)
@needs_diffusers
def test_tiny_pipeline_runs_on_mps_like_on_cpu():
    prepared = dpipe.preprocess_image(_image(), _CentreBlob())
    noise = {
        "dense": torch.randn(1, 2, 4, 4, 4, generator=torch.Generator().manual_seed(0)),
    }
    kwargs = dict(steps=(2, 2, 2), guidance=3.0, stop_after="dense", noise=noise)
    cpu = dpipe.run_stages(
        _tiny_modules("cpu"), prepared, torch.Generator().manual_seed(0), **kwargs
    )
    mps = dpipe.run_stages(
        _tiny_modules("mps"),
        prepared,
        torch.Generator(device="mps").manual_seed(0),
        **kwargs,
    )
    assert torch.allclose(cpu["dense_latent"], mps["dense_latent"].cpu(), atol=1e-3)
    # The whole fp16 pipeline runs on MPS, as it does on CUDA.
    half = _tiny_modules("mps", use_fp16=True)
    out = dpipe.run_stages(
        half,
        prepared,
        torch.Generator(device="mps").manual_seed(0),
        steps=(2, 2, 2),
        guidance=3.0,
    )
    assert out["sparse1024_latent"].dtype == torch.float16
    assert torch.isfinite(out["sparse1024_latent"]).all() and len(out["mesh"][1]) > 0
