"""Offline unit tests for the TRELLIS image-to-3D family: no weights, no network.

The sparse operations are checked against dense PyTorch references (a
zero-filled Conv3d grid for the submanifold convolution, per-window loops for
windowed attention); the networks and pipeline run end to end at a tiny size,
on CPU and, when available, on Apple MPS.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from libreyolo import LibreGen, LibreTRELLIS
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.trellis import nn as tnn
from libreyolo.models.gen.trellis import pipeline as tpipe
from libreyolo.models.gen.trellis import sparse as ts

pytestmark = [pytest.mark.unit, pytest.mark.gen]


def _coords(batch=2, side=10, count=300, seed=0):
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
    assert _ALIASES["trellis"] == (LibreTRELLIS, "l")
    assert _ALIASES["trellis-image-large"] == (LibreTRELLIS, "l")
    model = LibreGen("trellis", device="cpu")
    assert isinstance(model, LibreTRELLIS)
    assert model.task == "generate3d" and model.model_name == "trellis-l"
    assert model._pipeline is None


def test_defaults_follow_upstream():
    model = LibreTRELLIS(device="cpu")
    assert model._resolve_knobs({}) == {
        "steps": 25,
        "guidance": 5.0,
        "shift": 3.0,
        "margin": 1.2,
    }
    assert model._resolve_knobs({"steps": np.int64(12)})["steps"] == 12


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"steps": 0}, "steps must be"),
        ({"guidance": -1.0}, "guidance must be"),
        ({"guidance": float("nan")}, "guidance must be"),
        ({"shift": 0}, "shift must be"),
        ({"margin": 0.5}, "margin must be"),
        ({"gaussians": 1000}, "does not accept"),
        ({"mode": "real"}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        LibreTRELLIS(device="cpu")._resolve_knobs(knobs)


def test_constructor_rejects_unsupported_options():
    with pytest.raises(ValueError, match="compile=True"):
        LibreTRELLIS(device="cpu", compile=True)
    with pytest.raises(ValueError, match="dtype must be 'auto'"):
        LibreTRELLIS(device="cpu", dtype="fp16")


def test_matches_its_snapshot_directory(tmp_path):
    from libreyolo.models.gen.trellis.model import _UPSTREAM_FILES, _WEIGHT_FILES

    assert not LibreTRELLIS.matches_directory(tmp_path)
    for name in _UPSTREAM_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}")
    assert not LibreTRELLIS.matches_directory(tmp_path)
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"name": "TrellisImageTo3DPipeline", "args": {}})
    )
    # An unchanged upstream folder (no birefnet.safetensors) is recognised.
    assert LibreTRELLIS.matches_directory(tmp_path)
    assert (
        "birefnet.safetensors" in _WEIGHT_FILES
        and "birefnet.safetensors" not in _UPSTREAM_FILES
    )
    assert isinstance(LibreGen(str(tmp_path), device="cpu"), LibreTRELLIS)


def test_pins_every_download():
    from libreyolo.models.gen.trellis import model as module

    assert LibreTRELLIS.HF_REPOS["l"] == "LibreYOLO/LibreTRELLISl"
    assert LibreTRELLIS.HF_REVISIONS["l"] == "75c784a8fa56ebfb76f46d9b9d830caec25826a7"
    assert module._DINOV2_REPO == "facebook/dinov2-with-registers-large"
    assert len(module._DINOV2_REVISION) == 40


# ---------------------------------------------------------------------------
# Sparse operations against dense references
# ---------------------------------------------------------------------------


def test_submanifold_conv_equals_a_dense_conv_read_at_the_active_voxels():
    torch.manual_seed(0)
    coords = _coords()
    feats = torch.randn(coords.shape[0], 5)
    conv = ts.SubMConv3d(5, 7)
    with torch.no_grad():
        conv.bias.normal_()
    out = conv(ts.SparseTensor(feats, coords)).feats
    side = int(coords[:, 1:].max()) + 1
    dense = torch.zeros(2, 5, side, side, side)
    b, x, y, z = coords.long().unbind(1)
    dense[b, :, x, y, z] = feats
    # spconv's [out, kx, ky, kz, in] -> Conv3d's [out, in, kx, ky, kz]
    weight = conv.weight.detach().permute(0, 4, 1, 2, 3)
    reference = F.conv3d(dense, weight, conv.bias.detach(), padding=1)[b, :, x, y, z]
    assert torch.allclose(out, reference, atol=1e-5)


def test_submanifold_conv_chunks_give_the_same_rows(monkeypatch):
    torch.manual_seed(1)
    coords = _coords(batch=1, count=500)
    feats = torch.randn(coords.shape[0], 4)
    conv = ts.SubMConv3d(4, 3)
    whole = conv(ts.SparseTensor(feats, coords)).feats
    monkeypatch.setattr(ts, "_CONV_CHUNK_ROWS", 37)
    chunked = conv(ts.SparseTensor(feats, coords)).feats
    # BLAS blocking can move the last bit when the row count changes.
    assert torch.allclose(whole, chunked, atol=1e-6, rtol=0)


def test_downsample_then_upsample_restores_the_voxels():
    torch.manual_seed(2)
    coords = _coords(batch=2, side=12)
    x = ts.SparseTensor(torch.randn(coords.shape[0], 3), coords)
    coarse = ts.sparse_downsample(x, 2)
    assert (coarse.coords[:, 1:] <= 5).all()
    assert torch.equal(
        coarse.coords[:, 0].unique(), torch.tensor([0, 1], dtype=coarse.coords.dtype)
    )
    fine = ts.sparse_upsample(coarse)
    assert torch.equal(fine.coords, coords) and fine.layout == x.layout
    # Upstream's pooling counts the zero it starts from (scatter_reduce include_self).
    groups = {}
    for row, c in enumerate(coords.tolist()):
        groups.setdefault((c[0], c[1] // 2, c[2] // 2, c[3] // 2), []).append(row)
    first = coarse.coords[0].tolist()
    members = groups[tuple(first)]
    expected = x.feats[members].sum(0) / (len(members) + 1)
    assert torch.allclose(coarse.feats[0], expected, atol=1e-6)
    with pytest.raises(ValueError, match="sparse_downsample"):
        ts.sparse_upsample(x)


def test_full_attention_is_per_batch_element():
    torch.manual_seed(3)
    coords = _coords(batch=2, count=40)
    qkv = torch.randn(coords.shape[0], 3, 2, 4)
    out = ts.sparse_self_attention(ts.SparseTensor(qkv, coords)).feats
    tensor = ts.SparseTensor(qkv, coords)
    for rows in tensor.layout:
        q, k, v = qkv[rows].unbind(1)
        ref = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)
        assert torch.allclose(out[rows], ref, atol=1e-5)


@pytest.mark.parametrize("shift", [0, 2])
def test_windowed_attention_matches_a_loop_over_windows(shift):
    torch.manual_seed(4)
    coords = _coords(batch=2, side=9, count=200)
    qkv = torch.randn(coords.shape[0], 3, 2, 4)
    x = ts.SparseTensor(qkv, coords)
    out = ts.sparse_windowed_self_attention(x, 4, shift).feats
    window = (coords[:, 1:] + shift) // 4
    keys = {}
    for row, (b, w) in enumerate(zip(coords[:, 0].tolist(), window.tolist())):
        keys.setdefault((b, *w), []).append(row)
    for rows in keys.values():
        q, k, v = qkv[rows].unbind(1)
        ref = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        ).transpose(0, 1)
        assert torch.allclose(out[rows], ref, atol=1e-5)


def test_sparse_arithmetic_broadcasts_per_batch_element():
    coords = _coords(batch=2, count=20)
    feats = torch.ones(coords.shape[0], 3)
    x = ts.SparseTensor(feats, coords)
    scale = torch.tensor([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    y = x * scale
    assert torch.all(y.feats[x.layout[0]] == 2) and torch.all(y.feats[x.layout[1]] == 3)
    assert torch.equal((x + feats).feats, feats * 2)
    assert torch.equal((2.0 * x).feats, feats * 2)


def test_hammersley_matches_its_definition():
    assert tnn.hammersley(3, 0, 4) == [0.0, 0.0, 0.0]
    assert tnn.hammersley(3, 1, 4) == [0.25, 0.5, 1 / 3]
    # 3 is '10' in base 3, reversed 0.01 = 1/9
    assert tnn.hammersley(3, 3, 4) == [0.75, 0.75, 1 / 9]


# ---------------------------------------------------------------------------
# Tiny pipeline
# ---------------------------------------------------------------------------

TINY = {
    "structure_flow": dict(
        resolution=4,
        in_channels=2,
        model_channels=32,
        cond_channels=16,
        out_channels=2,
        num_blocks=1,
        num_heads=2,
        patch_size=1,
        qk_rms_norm=True,
    ),
    "structure_decoder": dict(
        out_channels=1, latent_channels=2, num_res_blocks=1, channels=[8, 4, 4]
    ),
    "latent_flow": dict(
        resolution=16,
        in_channels=8,
        model_channels=32,
        cond_channels=16,
        out_channels=8,
        num_blocks=1,
        num_heads=2,
        patch_size=2,
        io_block_channels=[8],
        qk_rms_norm=True,
    ),
    "gaussian_decoder": dict(
        resolution=16,
        model_channels=32,
        latent_channels=8,
        num_blocks=2,
        num_heads=2,
        window_size=4,
        representation_config={
            "lr": {
                "_xyz": 1.0,
                "_features_dc": 1.0,
                "_opacity": 1.0,
                "_scaling": 1.0,
                "_rotation": 0.1,
            },
            "perturb_offset": True,
            "voxel_size": 1.5,
            "num_gaussians": 32,
            "3d_filter_kernel_size": 9e-4,
            "scaling_bias": 4e-3,
            "opacity_bias": 0.1,
            "scaling_activation": "softplus",
        },
    ),
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


def _tiny_modules(device="cpu", use_fp16=False):
    from libreyolo.models.gen.trellis.model import _TRELLISModules

    torch.manual_seed(0)
    nets = {
        "structure_flow": tnn.SparseStructureFlowModel(
            **TINY["structure_flow"], use_fp16=use_fp16
        ),
        "structure_decoder": tnn.SparseStructureDecoder(
            **TINY["structure_decoder"], use_fp16=use_fp16
        ),
        "latent_flow": tnn.SLatFlowModel(**TINY["latent_flow"], use_fp16=use_fp16),
        "gaussian_decoder": tnn.SLatGaussianDecoder(
            **TINY["gaussian_decoder"], use_fp16=use_fp16
        ),
    }
    # Bias the occupancy so the random structure keeps a fraction of the grid.
    with torch.no_grad():
        nets["structure_decoder"].out_layer[-1].bias.fill_(-0.05)
    nets = {name: net.eval().to(device) for name, net in nets.items()}
    return _TRELLISModules(
        device=torch.device(device),
        dinov2=_FakeDINOv2().to(device),
        birefnet=_CentreBlob().to(device),
        normalization={"mean": [0.0] * 8, "std": [1.0] * 8},
        **nets,
    )


STAGE = {"steps": 2, "guidance": 3.0, "interval": (0.5, 1.0), "shift": 3.0}


def _image():
    return Image.new("RGB", (90, 60), (10, 200, 30))


def test_preprocess_crops_and_premultiplies():
    prepared = tpipe.preprocess_image(_image(), _CentreBlob())
    assert prepared.size == (518, 518) and prepared.mode == "RGB"
    pixels = np.array(prepared)
    assert pixels[259, 259].tolist() == [10, 200, 30]
    assert pixels[0, 0].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="No object found"):
        tpipe.preprocess_image(Image.new("RGBA", (40, 40), (0, 0, 0, 0)), _CentreBlob())


def test_tiny_pipeline_end_to_end():
    modules = _tiny_modules()
    prepared = tpipe.preprocess_image(_image(), modules.birefnet)
    out = tpipe.run_stages(modules, prepared, 7, structure=STAGE, latent=STAGE)
    voxels = out["coords"].shape[0]
    assert voxels > 0 and (out["coords"][:, 1:] < 16).all()
    g = out["gaussians"]
    assert g["xyz"].shape == (voxels * 32, 3) and g["sh"].shape == (voxels * 32, 1, 3)
    assert torch.allclose(
        g["rotations"].norm(dim=1), torch.ones(voxels * 32), atol=1e-5
    )
    assert (
        (g["opacities"] > 0).all()
        and (g["opacities"] < 1).all()
        and (g["scales"] > 0).all()
    )
    assert g["xyz"].abs().max() < 0.6
    again = tpipe.run_stages(modules, prepared, 7, structure=STAGE, latent=STAGE)[
        "gaussians"
    ]
    assert torch.equal(g["xyz"], again["xyz"])
    other = tpipe.run_stages(modules, prepared, 8, structure=STAGE, latent=STAGE)[
        "gaussians"
    ]
    assert other["xyz"].shape != g["xyz"].shape or not torch.equal(
        other["xyz"], g["xyz"]
    )


def test_generate_through_the_public_call(tmp_path):
    model = LibreTRELLIS(device="cpu")
    model._pipeline = _tiny_modules()
    calls = []
    forward = model._pipeline.dinov2.forward
    model._pipeline.dinov2.forward = lambda *a, **k: calls.append(1) or forward(*a, **k)
    pair = model(_image(), seed=3, n=2, steps=2)
    assert len(pair) == 2 and len(calls) == 1  # one encoding for both seeds
    result = model(_image(), seed=3, steps=2)
    assert torch.equal(result.gaussians.xyz, pair[0].gaussians.xyz)
    g = result.gaussians
    assert len(g) % 32 == 0 and g.seed == 3
    assert g.params["model"] == "trellis-l" and g.params["steps"] == 2
    assert g.camera == {"margin": 1.2}
    assert g.input_image.size == (518, 518)
    result.save(tmp_path / "object.ply")
    assert (tmp_path / "object.ply").stat().st_size > len(g) * 14 * 4


def test_guidance_interval_skips_the_unconditional_pass_outside_it():
    calls = []

    class Recorder(torch.nn.Module):
        def forward(self, x, t, cond):
            calls.append((float(t[0]), bool(cond.abs().sum() == 0)))
            return torch.zeros_like(x)

    tpipe.sample_flow(
        Recorder(),
        torch.zeros(1, 2),
        torch.ones(1, 3, 4),
        steps=4,
        guidance=5.0,
        interval=(0.5, 1.0),
        shift=1.0,
    )
    conditional = [t for t, zero in calls if not zero]
    unconditional = [t for t, zero in calls if zero]
    assert conditional == [1000.0, 750.0, 500.0, 250.0]
    assert unconditional == [1000.0, 750.0, 500.0]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_tiny_pipeline_runs_on_mps_like_on_cpu():
    prepared = tpipe.preprocess_image(_image(), _CentreBlob())
    cpu = tpipe.run_stages(
        _tiny_modules("cpu"),
        prepared,
        5,
        structure=STAGE,
        latent=STAGE,
        stop_after="structure",
    )
    mps = tpipe.run_stages(
        _tiny_modules("mps"),
        prepared,
        5,
        structure=STAGE,
        latent=STAGE,
        stop_after="structure",
    )
    assert torch.allclose(
        cpu["structure_latent"], mps["structure_latent"].cpu(), atol=1e-4
    )
    full = tpipe.run_stages(
        _tiny_modules("mps", use_fp16=True), prepared, 5, structure=STAGE, latent=STAGE
    )
    assert (
        torch.isfinite(full["gaussians"]["xyz"]).all()
        and full["gaussians"]["xyz"].shape[0] > 0
    )
