"""Parity: LibreGen("triposr") vs upstream TripoSR, same process, CPU, fp32.

Stage by stage, each side fed the same inputs (RGBA examples, so neither side
removes a background): the preprocessed image, the DINO tokens, the triplane,
the density grid upstream meshes, the mesh, the vertex colours, and end to
end through ``LibreGen("triposr")``. Upstream meshes with torchmcubes, the
port with scikit-image: the isosurface is the same, the per-cube triangle
tables differ, so the mesh is compared by vertex and face count, enclosed
volume, surface area and the two-way vertex distance, in voxels.

Measured on an RTX 3060 box (torch 2.5.1), chair, horse and robot, on CPU and
CUDA: the preprocessed image is bitwise equal; DINO tokens differ by at most
1.1e-5 and the triplane by about 5e-5 relative; the density grid from the same
triplane is bitwise equal on CUDA (6.5e-3 on values up to 1.4e4 on CPU); the
meshes agree to 4e-5 voxel, with volume and area within 2e-4 and vertex
colours within one 8-bit level.

Prereqs (throwaway env, do NOT install into the main .venv; torchmcubes is
MPL-2.0 and only runs upstream here). Install LibreYOLO's core first and
upstream's pins last, so transformers stays at upstream's 4.35.0 (the port
does not import transformers):
    pip install -e . scikit-image
    pip install torch==2.5.1 torchvision==0.20.1 transformers==4.35.0 \
        omegaconf==2.3.0 einops==0.7.0 trimesh==4.0.5 "numpy<2" scipy \
        scikit_build_core pybind11 ninja cmake
    pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
    git clone https://github.com/VAST-AI-Research/TripoSR && \
        git -C TripoSR checkout 107cefdc244c39106fa830359024f6a2f1c78871
    hf download LibreYOLO/LibreTripoSR --local-dir weights/LibreTripoSR

Run:
    python weights/parity_triposr.py --upstream TripoSR --weights weights/LibreTripoSR \
        TripoSR/examples/chair.png TripoSR/examples/horse.png TripoSR/examples/robot.png
"""

import argparse
import sys
import time
import types

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("--resolution", type=int, default=256)
_parser.add_argument("--device", default="cpu")
_parser.add_argument(
    "--end-to-end",
    action="store_true",
    help="also run LibreGen('triposr') (downloads BiRefNet)",
)
_parser.add_argument(
    "--rgb", nargs="*", default=[], help="RGB images run end to end (BiRefNet cutout)"
)
_parser.add_argument("images", nargs="+")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)
# Upstream imports rembg and imageio at module import; RGBA inputs need neither.
for _name in ("rembg", "imageio"):
    sys.modules[_name] = types.ModuleType(_name)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402

from libreyolo.models.gen.triposr import nn as tnn  # noqa: E402
from libreyolo.models.gen.triposr import preprocess as tpre  # noqa: E402

torch.manual_seed(0)
upstream = TSR.from_pretrained(
    ARGS.weights, config_name="config.yaml", weight_name="model.ckpt"
)
upstream.renderer.set_chunk_size(8192)
upstream.eval().to(ARGS.device)
port = tnn.TripoSR()
missing = port.load_state_dict(
    torch.load(f"{ARGS.weights}/model.ckpt", map_location="cpu", weights_only=True),
    strict=True,
)
port.eval().to(ARGS.device)
print(
    "state dict: strict load OK,",
    sum(p.numel() for p in port.parameters()),
    "parameters",
)


def _t(x):
    return (
        x.detach().cpu().double()
        if torch.is_tensor(x)
        else torch.as_tensor(np.asarray(x)).double()
    )


def rss(stage):
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak / 2**30 if sys.platform == "darwin" else peak / 2**20
    print(f"  [peak RSS after {stage}: {peak:.1f} GB]", flush=True)


def cmp(name, a, b):
    a, b = (
        _t(a),
        _t(b),
    )
    assert a.shape == b.shape, (name, a.shape, b.shape)
    d = (a - b).abs()
    scale = b.abs().max().item()
    print(
        f"  {name:<24} shape {tuple(a.shape)}  max|d| {d.max().item():.3e}  mean|d| {d.mean().item():.3e}  (max|ref| {scale:.3g})  bitwise {bool((d == 0).all())}"
    )


_MODEL = []


def get_model():
    if not _MODEL:
        from libreyolo import LibreGen

        _MODEL.append(LibreGen(ARGS.weights, device=ARGS.device))
    return _MODEL[0]


def sync():
    if ARGS.device.startswith("cuda"):
        torch.cuda.synchronize()


def mesh_report(v_a, f_a, v_b, f_b, R):
    """Mesh a (port) vs mesh b (reference); returns b->a nearest indices and distances."""
    voxel = 2 * 0.87 / (R - 1)
    vol_a, area_a = volume_area(v_a, f_a)
    vol_b, area_b = volume_area(v_b, f_b)
    d_ab, _ = cKDTree(v_b).query(v_a)
    d_ba, idx = cKDTree(v_a).query(v_b)
    print(f"  vertices                port {len(v_a)}, reference {len(v_b)}")
    print(f"  faces                   port {len(f_a)}, reference {len(f_b)}")
    print(
        f"  enclosed volume         port {vol_a:.6f}, reference {vol_b:.6f}, rel diff {abs(vol_a - vol_b) / abs(vol_b):.2e}"
    )
    print(
        f"  surface area            port {area_a:.6f}, reference {area_b:.6f}, rel diff {abs(area_a - area_b) / area_b:.2e}"
    )
    print(
        f"  vertex distance (vox)   port->ref max {d_ab.max() / voxel:.2e}, ref->port max {d_ba.max() / voxel:.2e}"
    )
    return idx, d_ba


def volume_area(v, f):
    v = v.astype(np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2
    return volume, area


for path in ARGS.images:
    print(f"\n== {path}")
    image = Image.open(path)
    # Upstream's run.py (RGBA input, so remove_background is a no-op).
    up_img = resize_foreground(image, 0.85)
    arr = np.array(up_img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    up_img = Image.fromarray((arr * 255.0).astype(np.uint8))
    port_img = tpre.preprocess_image(image, None, margin=1 / 0.85)
    cmp("preprocessed image", np.array(port_img), np.array(up_img))

    with torch.no_grad():
        rgb = upstream.image_processor(up_img, 512)[:, None].to(ARGS.device)
        port_px = tpre.to_tensor(port_img, 512).to(ARGS.device)
        cmp("encoder input", port_px, rgb[:, 0].permute(0, 3, 1, 2))
        up_tokens = upstream.image_tokenizer(rgb.permute(0, 1, 4, 2, 3))  # [B, 1, C, N]
        port_tokens = port.image_tokenizer(port_px)
        cmp("DINO tokens", port_tokens, up_tokens[:, 0].permute(0, 2, 1))
        t0 = time.perf_counter()
        up_codes = upstream([up_img], device=ARGS.device)
        t1 = time.perf_counter()
        port_codes = port(port_px)
        t2 = time.perf_counter()
        cmp("triplane", port_codes, up_codes)
        print(f"  forward time: upstream {t1 - t0:.2f}s, port {t2 - t1:.2f}s")

        # The density grid upstream meshes, both from upstream's triplane.
        R = ARGS.resolution
        upstream.set_marching_cubes_resolution(R)
        helper = upstream.isosurface_helper
        from tsr.utils import scale_tensor

        up_density = upstream.renderer.query_triplane(
            upstream.decoder,
            scale_tensor(
                helper.grid_vertices.to(ARGS.device), helper.points_range, (-0.87, 0.87)
            ),
            up_codes[0],
        )["density_act"].view(R, R, R)
        port_density = port.density_grid(up_codes[0], R)
        cmp("density grid", port_density, up_density)

    rss("density grids")
    del up_density, port_density
    up_mesh = upstream.extract_mesh(up_codes, True, resolution=R)[0]
    rss("upstream mesh")
    v_port, f_port, c_port = tnn.extract_mesh(port, up_codes[0], R, 25.0)
    rss("port mesh")
    v_up, f_up = np.asarray(up_mesh.vertices, np.float32), np.asarray(up_mesh.faces)
    print(
        "  -- mesh from upstream's triplane: port (scikit-image) vs upstream (torchmcubes)"
    )
    idx, d_up = mesh_report(v_port.numpy(), f_port.numpy(), v_up, f_up, R)
    up_colors = np.asarray(up_mesh.visual.vertex_colors)[:, :3].astype(np.float64)
    port_colors = np.round(c_port.numpy()[idx] * 255.0)
    near = d_up < 1e-3 * 2 * 0.87 / (R - 1)
    print(
        f"  vertex colours (8-bit)  at {near.sum()} coincident vertices: "
        f"max|d| {np.abs(up_colors[near] - port_colors[near]).max():.0f} levels"
    )

    if ARGS.end_to_end:
        t0 = time.perf_counter()
        mesh = get_model()(path, resolution=R).meshes
        sync()
        print(
            f"  -- end to end: LibreGen('triposr') on {ARGS.device} ({time.perf_counter() - t0:.2f}s) vs upstream"
        )
        mesh_report(
            (mesh.vertices[0] * (2 * 0.87)).cpu().numpy(),
            mesh.faces.cpu().numpy(),
            v_up,
            f_up,
            R,
        )

for path in ARGS.rgb:
    t0 = time.perf_counter()
    result = get_model()(path, resolution=ARGS.resolution)
    sync()
    m = result.meshes
    print(
        f"\n== {path} (RGB, BiRefNet cutout) on {ARGS.device}: {m.num_vertices} vertices, "
        f"{m.faces.shape[0]} faces, max|v| {float(m.vertices.abs().max()):.3f}, "
        f"{time.perf_counter() - t0:.2f}s"
    )
