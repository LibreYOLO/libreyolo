"""Offline unit tests for the TripoSR image-to-3D family: no weights, no network.

The networks run at a tiny size on CPU and, when available, Apple MPS; the
mesh path runs on an analytic density so the frame, scale and winding of the
returned ``Meshes`` can be checked exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGen, LibreTripoSR
from libreyolo.models.gen import _ALIASES
from libreyolo.models.gen.triposr import nn as tnn
from libreyolo.models.gen.triposr import preprocess as tpre
from libreyolo.models.gen.triposr.model import _TripoSRModules
from libreyolo.utils.results import Meshes

pytest.importorskip("skimage")

pytestmark = [pytest.mark.unit, pytest.mark.gen]

_TINY = dict(
    tnn.CONFIG,
    plane_size=4,
    num_channels=64,
    num_attention_heads=2,
    attention_head_dim=32,
    num_layers=2,
    cross_attention_dim=32,
    triplane_channels=8,
    mlp_neurons=16,
    mlp_hidden_layers=3,
)
_TINY_DINO = dict(
    tnn.DINO_CONFIG,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    image_size=32,
)
_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _tiny_network(device="cpu"):
    torch.manual_seed(0)
    return tnn.TripoSR(_TINY, _TINY_DINO).eval().to(device)


def _sphere_query(radius_obj=0.4):
    """A density that is 50 inside a sphere of ``radius_obj`` (upstream units); colour features = position."""

    def query_chunk(self, triplane, positions):
        r = positions.norm(dim=-1)
        density = 50.0 * torch.sigmoid((radius_obj - r) * 40.0)
        return density, positions / self.radius

    return query_chunk


def _cutout(size=96):
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[20:76, 30:60] = (200, 40, 40, 255)
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# Factory, knobs, snapshot
# ---------------------------------------------------------------------------


def test_alias_resolves_without_loading():
    assert _ALIASES["triposr"] == (LibreTripoSR, "")
    model = LibreGen("triposr", device="cpu")
    assert isinstance(model, LibreTripoSR)
    assert model.task == "generate3d" and model.model_name == "triposr"
    assert model._pipeline is None


def test_defaults_follow_upstream():
    knobs = LibreTripoSR(device="cpu")._resolve_knobs({})
    assert knobs["resolution"] == 256 and knobs["threshold"] == 25.0
    # run.py's --foreground-ratio 0.85, as crop side over object size.
    assert 1.0 / knobs["margin"] == 0.85


@pytest.mark.parametrize(
    ("knobs", "match"),
    [
        ({"resolution": 8}, "resolution must be"),
        ({"resolution": 64.0}, "resolution must be"),
        ({"threshold": 0}, "threshold must be"),
        ({"threshold": float("nan")}, "threshold must be"),
        ({"margin": 0.9}, "margin must be"),
        ({"steps": 10}, "does not accept"),
    ],
)
def test_knob_validation(knobs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        LibreTripoSR(device="cpu")._resolve_knobs(knobs)


def test_constructor_rejects_compile():
    with pytest.raises(ValueError, match="compile"):
        LibreTripoSR(device="cpu", compile=True)


def test_matches_its_snapshot_directory(tmp_path):
    assert not LibreTripoSR.matches_directory(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "image_tokenizer_cls: tsr.models.tokenizers.image.DINOSingleImageTokenizer\n"
    )
    (tmp_path / "model.ckpt").write_bytes(b"")
    assert LibreTripoSR.matches_directory(tmp_path)


def test_pins_every_download():
    from libreyolo.models.gen.triposr import model as tmodel

    for revision in (*LibreTripoSR.HF_REVISIONS.values(), tmodel._BIREFNET_REVISION):
        assert len(revision) == 40 and int(revision, 16) >= 0
    assert set(LibreTripoSR.HF_REVISIONS) == set(LibreTripoSR.HF_REPOS)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", _DEVICES)
def test_tiny_network_forward_and_query(device):
    net = _tiny_network(device)
    images = torch.rand(2, 3, 64, 64, device=device)
    with torch.no_grad():
        planes = net(images)
        assert planes.shape == (2, 3, 8, 8, 8)
        density, colour = net.query(
            planes[0], torch.rand(1000, 3, device=device) - 0.5, chunk=300
        )
    assert density.shape == (1000,) and colour.shape == (1000, 3)
    assert bool((density > 0).all()) and bool(((colour >= 0) & (colour <= 1)).all())


def test_query_chunks_give_the_same_values():
    net = _tiny_network()
    with torch.no_grad():
        planes = net(torch.rand(1, 3, 64, 64))[0]
        points = torch.rand(777, 3) - 0.5
        whole = net.query(planes, points)
        chunked = net.query(planes, points, chunk=100)
    torch.testing.assert_close(whole[0], chunked[0])
    torch.testing.assert_close(whole[1], chunked[1])


def test_density_grid_equals_a_query_over_the_whole_lattice():
    net = _tiny_network()
    with torch.no_grad():
        planes = net(torch.rand(1, 3, 64, 64))[0]
        axis = torch.linspace(0.0, 1.0, 9)
        lattice = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
        lattice = tnn._scale(
            lattice.reshape(-1, 3), (0.0, 1.0), (-net.radius, net.radius)
        )
        whole, _ = net.query(planes, lattice, chunk=100)
    grid = net.density_grid(planes, 9, chunk=100)
    assert grid.device.type == "cpu"
    assert torch.equal(grid, whole.view(9, 9, 9))


def test_vit_position_interpolation_keeps_the_native_grid():
    vit = tnn.DINOViT(_TINY_DINO)
    native = vit.embeddings._positions(4, 32, 32)
    assert native is vit.embeddings.position_embeddings
    assert vit.embeddings._positions(16, 64, 64).shape == (1, 17, 32)


def test_triplane_tokens_round_trip():
    tok = tnn.TriplaneTokenizer(4, 8)
    tokens = tok(2)
    assert tokens.shape == (2, 8, 48)
    torch.testing.assert_close(tok.detokenize(tokens)[0], tok.embeddings)


# ---------------------------------------------------------------------------
# Preprocessing and mesh
# ---------------------------------------------------------------------------


def test_frame_object_follows_upstream_crop():
    rgba = np.array(_cutout())
    framed = tpre.frame_object(rgba, 1 / 0.85)
    # Box rows 20..75, columns 30..59, sliced [min:max] -> 55 x 29 -> square 55.
    assert framed.shape == (int(55 / 0.85), int(55 / 0.85), 4)
    with pytest.raises(ValueError, match="No object"):
        tpre.frame_object(np.zeros((8, 8, 4), np.uint8), 1.2)


def test_preprocess_composites_on_grey_and_resizes():
    image = tpre.preprocess_image(_cutout(), birefnet=None, margin=1 / 0.85)
    pixels = np.array(image)
    assert tuple(pixels[0, 0]) == (127, 127, 127)
    assert tpre.to_tensor(image, 32).shape == (1, 3, 32, 32)


def test_extract_mesh_frame_and_winding(monkeypatch):
    monkeypatch.setattr(tnn.TripoSR, "_query_chunk", _sphere_query(0.4))
    net = _tiny_network()
    vertices, faces, colours = tnn.extract_mesh(net, torch.zeros(3, 8, 8, 8), 64, 25.0)
    radii = vertices.norm(dim=-1)
    assert abs(float(radii.mean()) - 0.4) < 0.01
    v = vertices.double()
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    volume = float(torch.einsum("ij,ij->i", a, torch.cross(b, c, dim=1)).sum() / 6)
    # Counter-clockwise seen from outside: positive enclosed volume.
    assert abs(volume - 4 / 3 * np.pi * 0.4**3) / (4 / 3 * np.pi * 0.4**3) < 0.03
    assert colours.shape == vertices.shape


def test_extract_mesh_without_surface_raises(monkeypatch):
    monkeypatch.setattr(tnn.TripoSR, "_query_chunk", _sphere_query(0.4))
    with pytest.raises(ValueError, match="No surface"):
        tnn.extract_mesh(_tiny_network(), torch.zeros(3, 8, 8, 8), 32, 80.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_generate_returns_a_mesh_payload(monkeypatch, tmp_path, device):
    monkeypatch.setattr(tnn.TripoSR, "_query_chunk", _sphere_query(0.4))
    model = LibreTripoSR(device=device)
    net = tnn.TripoSR(dict(_TINY, cond_image_size=64), _TINY_DINO).eval().to(device)
    model._pipeline = _TripoSRModules(network=net, birefnet=None)
    out = tmp_path / "obj.obj"
    result = model(_cutout(), resolution=48, seed=3, save=True, output_path=str(out))
    mesh = result.meshes
    assert (
        isinstance(mesh, Meshes) and result.gaussians is None and result.boxes is None
    )
    assert len(result) == 1 and len(mesh) == 1 and mesh.body_model == "object"
    assert mesh.vertices.shape[0] == 1 and mesh.faces.shape[1] == 3
    assert float(mesh.vertices.abs().max()) <= 0.5
    assert mesh.extras["vertex_colors"].shape == mesh.vertices.shape
    row = result.summary()[0]
    assert row["name"] == "mesh" and row["num_vertices"] == mesh.num_vertices
    lines = out.read_text().splitlines()
    first_vertex = next(line for line in lines if line.startswith("v "))
    assert len(first_vertex.split()) == 7
    assert sum(line.startswith("f ") for line in lines) == mesh.faces.shape[0]
    # Deterministic: n objects are the same mesh.
    many = model(_cutout(), resolution=48, n=2)
    torch.testing.assert_close(many[0].meshes.vertices, many[1].meshes.vertices)
    # ...but not the same tensors: editing one result leaves the other alone.
    many[0].meshes.vertices.zero_()
    many[0].meshes.extras["vertex_colors"].zero_()
    assert float(many[1].meshes.vertices.abs().max()) > 0
    assert float(many[1].meshes.extras["vertex_colors"].max()) > 0


def test_save_dir_uses_obj(monkeypatch, tmp_path):
    monkeypatch.setattr(tnn.TripoSR, "_query_chunk", _sphere_query(0.4))
    monkeypatch.chdir(tmp_path)
    model = LibreTripoSR(device="cpu")
    net = tnn.TripoSR(dict(_TINY, cond_image_size=64), _TINY_DINO).eval()
    model._pipeline = _TripoSRModules(network=net, birefnet=None)
    model(_cutout(), resolution=32, seed=5, save=True)
    assert [p.name for p in (tmp_path / "runs/generate/predict").glob("*")] == [
        "000_image_5.obj"
    ]
    with pytest.raises(ValueError, match=r"\.obj"):
        model(_cutout(), save=True, output_path=str(tmp_path / "x.ply"))
