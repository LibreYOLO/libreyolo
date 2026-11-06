"""CRM (LibreGen image-to-mesh) on tiny random networks: no weights, no downloads."""

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.gen]
pytest.importorskip("diffusers")
pytest.importorskip("scipy")

from libreyolo.models.gen import LibreGen
from libreyolo.models.gen.crm import LibreCRM
from libreyolo.models.gen.crm.diffusion import (
    MultiViewDiffusion,
    alphas_cumprod,
    cameras,
)
from libreyolo.models.gen.crm.flexicubes import FlexiCubes
from libreyolo.models.gen.crm.model import _CRMModules, mesh_payload
from libreyolo.models.gen.crm.pipeline import generate_mesh, preprocess_image
from libreyolo.models.gen.crm.reconstruction import (
    CRMNet,
    Grid,
    clean_mesh,
    rasterize_uv,
    triplane_input,
)
from libreyolo.utils.results import Results

_CLIP = {"width": 64, "layers": 2, "heads": 2, "vision": {"width": 64, "layers": 2, "heads": 2}}
_UNET = {
    "model_channels": 32,
    "channel_mult": (1, 1, 1, 1),
    "num_head_channels": 32,
    "context_dim": 64,
    "image_dim": 64,
    "resampler_heads": 2,
}


def _devices():
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _tiny_modules(device):
    device = torch.device(device)
    half = device.type != "cpu"
    torch.manual_seed(0)

    def init(module):
        for p in module.parameters():
            torch.nn.init.normal_(p, 0, 0.02)
        module = module.eval().requires_grad_(False)
        return (module.half() if half else module).to(device)

    pixel = init(MultiViewDiffusion(False, unet=_UNET, clip=_CLIP, vae_ch=32))
    ccm = init(MultiViewDiffusion(True, unet=_UNET, clip=_CLIP, vae_ch=32))
    net = CRMNet(block_out_channels=(32,) * 7)
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.02)
    net = net.eval().requires_grad_(False).to(device)
    return _CRMModules(
        device=device, half=half, pixel=pixel, ccm=ccm, net=net, grid=Grid(device, res=16), birefnet=None
    )


def _cutout():
    image = Image.new("RGBA", (120, 80), (0, 0, 0, 0))
    image.paste((200, 40, 40, 255), (40, 20, 80, 60))
    return image


@pytest.mark.parametrize("device", _devices())
def test_end_to_end_tiny(device, tmp_path):
    modules = _tiny_modules(device)
    prepared = preprocess_image(_cutout(), None)
    assert prepared.size == (120, 120) and prepared.mode == "RGB"
    assert prepared.getpixel((0, 0)) == (127, 127, 127)
    out = generate_mesh(modules, prepared, 3, steps=2, guidance=5.0, texture=False)
    assert out["views"].shape == (6, 256, 256, 3) and out["ccms"].shape == (6, 256, 256, 3)
    assert out["vertices"].ndim == 2 and out["faces"].shape[1] == 3 and len(out["faces"]) > 0
    assert out["faces"].max() < len(out["vertices"])
    assert np.abs(out["vertices"]).max() <= 0.51
    again = generate_mesh(modules, prepared, 3, steps=2, guidance=5.0, texture=False)
    np.testing.assert_array_equal(out["views"], again["views"])

    mesh = mesh_payload(out, seed=3, params={"model": "crm", "steps": 2})
    result = Results(None, (80, 120), meshes=mesh, names={})
    assert len(result) == 1 and mesh.body_model == "object"
    row = result.summary()[0]
    assert row["seed"] == 3 and row["num_faces"] == len(out["faces"])
    path = tmp_path / "object.obj"
    result.save(str(path))
    lines = path.read_text().splitlines()
    assert sum(line.startswith("v ") for line in lines) == len(out["vertices"])
    assert len(lines[2].split()) == 7  # v x y z r g b


def test_textured_obj(tmp_path):
    pytest.importorskip("xatlas")
    modules = _tiny_modules("cpu")
    out = generate_mesh(modules, preprocess_image(_cutout(), None), 0, steps=1, guidance=1.0, texture=True)
    assert out["texture"].shape == (1024, 1024, 3) and out["uv_faces"].shape == out["faces"].shape
    mesh = mesh_payload(out, seed=0, params={})
    Results(None, (80, 120), meshes=mesh, names={}).save(str(tmp_path / "t.obj"))
    assert (tmp_path / "t.png").is_file() and (tmp_path / "t.mtl").is_file()
    assert "vt " in (tmp_path / "t.obj").read_text()


def test_flexicubes_sphere():
    fc = FlexiCubes()
    verts, cubes = fc.construct_voxel_grid(16)
    assert verts.shape == (17**3, 3) and cubes.shape == (16**3, 8)
    assert float(verts.min()) == -0.5 and float(verts.max()) == 0.5
    sdf = verts.norm(dim=-1) - 0.3
    zeros = torch.zeros(cubes.shape[0], 12)
    v, f = fc(verts, sdf, cubes, 16, zeros, zeros[:, :8], zeros[:, 0])
    radius = v.norm(dim=-1)
    assert len(f) > 100 and float((radius - 0.3).abs().max()) < 0.03
    # Faces point toward positive SDF (outward here).
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normal = torch.cross(b - a, c - a, dim=-1)
    assert float((normal * (a + b + c)).sum(-1).gt(0).float().mean()) > 0.99


def test_clean_mesh_merges_and_drops():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1e-4, 0, 0], [5, 5, 5]], np.float32)
    faces = np.array([[0, 1, 2], [3, 1, 2], [0, 3, 1]])
    v, f = clean_mesh(vertices, faces)
    assert len(v) == 3 and f.tolist() == [[0, 1, 2]]


def test_rasterize_uv_covers_triangle():
    uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    face_id, bary = rasterize_uv(uvs, torch.tensor([[0, 1, 2]]), 32)
    covered = face_id >= 0
    assert 0.4 < covered.float().mean() < 0.6
    assert torch.allclose(bary[covered].sum(-1), torch.ones(int(covered.sum())))
    # Texel (row, col) samples uv ((col + .5) / R, (row + .5) / R).
    assert bool(covered[0, 0]) and not bool(covered[31, 31])


def test_triplane_layout():
    views = np.random.default_rng(0).integers(0, 255, (6, 256, 256, 3), dtype=np.uint8)
    triplane = triplane_input(views, views)
    assert triplane.shape == (1, 12, 256, 768)
    # The CCM half keeps one coordinate per plane.
    assert float(triplane[0, 7:9, :, :256].abs().max()) == 0


def test_schedule_and_cameras():
    schedule = alphas_cumprod(torch.float32)
    assert schedule[-1] == 0 and 0.99 < float(schedule[0]) < 1
    cams = cameras((1, 2, 3, 4, 5, 0, 0), 6)
    assert cams.shape == (7, 16) and float(cams[6].abs().sum()) == 0
    assert torch.isfinite(cams).all()


def test_alias_knobs_and_contract():
    model = LibreGen("crm", device="cpu")
    assert isinstance(model, LibreCRM) and model.task == "generate3d"
    assert model.SAVE_SUFFIXES == (".obj",)
    with pytest.raises(ValueError, match="steps"):
        model._resolve_knobs({"steps": 0})
    with pytest.raises(ValueError, match="texture"):
        model._resolve_knobs({"texture": "yes"})
    with pytest.raises(TypeError, match="gaussians"):
        model._resolve_knobs({"gaussians": 1})
    with pytest.raises(ValueError, match="compile"):
        LibreCRM(compile=True, device="cpu")
    with pytest.raises(ValueError, match=r"\.obj"):
        model(_cutout(), save=True, output_path="x.ply")
