"""Parity: LibreGen("direct3d-s2") (plain PyTorch) vs upstream Direct3D-S2 (torchsparse, flash-attn, Triton), same GPU, same process.

Stage by stage, each side fed the same inputs: one torchsparse convolution
(fp32 and fp16, which pins the kernel layout and offset order), the
preprocessed image (an RGBA example, so neither side removes a background),
the DINOv2 tokens, one forward and the whole sampler of each of the three
flow transformers (dense, sparse 512, sparse 1024), the dense decoder's
voxels, the 512 and 1024 SDF decoders, both refiners' dense grids and
meshes, the 1024 voxelisation of the 512 mesh, and end to end through each
side's own call with the same seed.

Prereqs (a CUDA machine; throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" omegaconf einops trimesh triton==3.1.0
    pip install <flash-attn 2.7.3 wheel for your torch/CUDA>
    git clone https://github.com/mit-han-lab/torchsparse && git -C torchsparse checkout 385f5ce8718fcae93540511b7f5832f4e71fd835
    FORCE_CUDA=1 pip install --no-build-isolation ./torchsparse
    git clone https://github.com/DreamTechAI/Direct3D-S2 && git -C Direct3D-S2 checkout a1cf235b2881cff04a91900060a9546b40e7ee5d
    pip install --no-build-isolation ./Direct3D-S2/third_party/voxelize
    hf download wushuang98/Direct3D-S2 --revision 8b04a8eddb7a56a0f4e89fe5f5b840c7d5610c00 --include "direct3d-s2-v-1-1/*" --local-dir SNAP
    (plus birefnet.safetensors from LibreYOLO/LibreTRELLISl in SNAP)

Run:
    python weights/parity_direct3d_s2.py --upstream Direct3D-S2 --weights SNAP [images...]
"""

import argparse
import json
import sys
import time
from types import ModuleType

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--out", default="parity_direct3d_s2.json")
_parser.add_argument("images", nargs="*")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)
# Upstream's hole filling (remesh=True, off by default) imports mesh tools
# this gate never calls; stub them rather than install them.
for _name in ("utils3d", "utils3d.torch", "pymeshfix", "igraph", "pyvista"):
    sys.modules[_name] = ModuleType(_name)
sys.modules["pymeshfix"]._meshfix = None

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

import direct3d_s2.modules.sparse as up_sp  # noqa: E402
from direct3d_s2.pipeline import Direct3DS2Pipeline  # noqa: E402
from direct3d_s2.utils import mesh as up_mesh  # noqa: E402
from direct3d_s2.utils import sort_block as up_sort_block  # noqa: E402

from libreyolo.models.gen.direct3d_s2 import LibreDirect3DS2  # noqa: E402
from libreyolo.models.gen.direct3d_s2 import pipeline as lp  # noqa: E402
from libreyolo.models.gen.direct3d_s2 import sparse as ls  # noqa: E402
from libreyolo.models.gen.trellis.pipeline import encode_image  # noqa: E402

DEVICE = torch.device("cuda")
REPORT = {}


def stat(name, a, b):
    a, b = a.float().cpu(), b.float().cpu()
    d = (a - b).abs()
    scale = b.abs().mean().item() + 1e-12
    row = {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_mean": d.mean().item() / scale,
        "bitwise": bool(torch.equal(a, b)),
    }
    REPORT[name] = row
    print(f"{name:40s} {json.dumps(row)}", flush=True)
    return row


def note(name, value):
    REPORT[name] = value
    print(f"{name:40s} {json.dumps(value)}", flush=True)


def keyed(coords, feats):
    """Sort sparse rows by (b, x, y, z) so two tensors with the same voxels line up."""
    c = coords.long()
    key = ((c[:, 0] * 4096 + c[:, 1]) * 4096 + c[:, 2]) * 4096 + c[:, 3]
    order = torch.argsort(key)
    return key[order], feats[order]


def same_voxels(name, a, b):
    ka = torch.unique(keyed(a, a[:, :1])[0])
    kb = torch.unique(keyed(b, b[:, :1])[0])
    note(name, {"a": len(ka), "b": len(kb), "equal": bool(torch.equal(ka.cpu(), kb.cpu()))})


def chamfer(name, va, fa, vb, fb, samples=200_000):
    """Symmetric mean surface distance between two meshes, in output units ([-1, 1] cube)."""
    import trimesh

    def points(v, f):
        mesh = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False)
        p, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
        return torch.as_tensor(p, dtype=torch.float32, device=DEVICE)

    pa, pb = points(va, fa), points(vb, fb)

    def nearest(x, y):
        return torch.cat([torch.cdist(c, y).min(dim=1).values for c in x.split(4096)])

    ab, ba = nearest(pa, pb), nearest(pb, pa)
    note(
        name,
        {
            "vertices": [len(va), len(vb)],
            "faces": [len(fa), len(fb)],
            "chamfer_mean": float((ab.mean() + ba.mean()) / 2),
            "hausdorff_p99": float(torch.quantile(torch.cat([ab, ba])[:1_000_000], 0.99)),
        },
    )


def signed_volume(v, f):
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f)
    return float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)


# -- 0. The torchsparse convolution -------------------------------------------


def conv_check():
    torch.manual_seed(0)
    coords = torch.unique(torch.randint(0, 24, (4000, 3)), dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int().to(DEVICE)
    for dtype in (torch.float32, torch.float16):
        feats = torch.randn(len(coords), 32, device=DEVICE, dtype=dtype)
        up_conv = up_sp.SparseConv3d(32, 16, 3, padding=1).to(DEVICE, dtype)
        mine = ls.SparseConv3d(32, 16, 3).to(DEVICE, dtype)
        mine.load_state_dict(up_conv.state_dict())
        a = up_conv(up_sp.SparseTensor(feats, coords))
        b = mine(ls.SparseTensor(feats, coords))
        ka, fa = keyed(a.coords, a.feats)
        kb, fb = keyed(b.coords, b.feats)
        assert torch.equal(ka, kb), "torchsparse changed the voxels of a stride-1 conv"
        stat(f"conv3x3 {dtype}", fb, fa)
        up_conv = up_sp.SparseConv3d(32, 16, 1, padding=0).to(DEVICE, dtype)
        mine = ls.SparseConv3d(32, 16, 1).to(DEVICE, dtype)
        mine.load_state_dict(up_conv.state_dict())
        a = up_conv(up_sp.SparseTensor(feats, coords))
        b = mine(ls.SparseTensor(feats, coords))
        stat(f"conv1x1 {dtype}", keyed(b.coords, b.feats)[1], keyed(a.coords, a.feats)[1])


# -- Upstream's sampling loop, exposed ----------------------------------------


@torch.no_grad()
def up_sample(pipe, dit, scheduler, noise, cond, steps, guidance, coords=None):
    uncond = torch.zeros_like(cond)
    scheduler.set_timesteps(steps, device=DEVICE)
    latents = noise
    for t in scheduler.timesteps:
        tt = torch.tensor([t], dtype=latents.dtype, device=DEVICE)
        x = latents if coords is None else up_sp.SparseTensor(latents, coords.int())
        c = dit(x=x, t=tt, cond=cond)
        u = dit(x=x, t=tt, cond=uncond)
        if coords is not None:
            c, u = c.feats, u.feats
        latents = scheduler.step(u + guidance * (c - u), t, latents).prev_sample
    return latents


def captured_marching_cubes(module):
    """Record the volume upstream hands to marching cubes inside ``module``."""
    store = {}
    real = module.measure.marching_cubes

    def wrapper(volume, *args, **kwargs):
        store["volume"] = np.array(volume)
        return real(volume, *args, **kwargs)

    module.measure.marching_cubes = wrapper
    return store


@torch.no_grad()
def stages(pipe, model, image_path, seed):
    import direct3d_s2.models.refiner.unet_refiner as up_refiner

    M = model._pipeline
    tag = image_path.split("/")[-1]
    image = Image.open(image_path)
    image.load()
    up_image = pipe.prepare_image(image)
    prepared = lp.preprocess_image(image)
    stat(f"{tag} preprocess", TF.to_tensor(prepared).to(DEVICE), up_image[0, :3])
    cond_up, _ = pipe.encode_image(up_image, pipe.dense_image_encoder, True, False)
    cond_mine = encode_image(M.dinov2, prepared, DEVICE)
    stat(f"{tag} dinov2 tokens", cond_mine, cond_up)
    cond = cond_up  # each later stage starts from the same tokens

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # Dense.
    noise = torch.randn((1, 8, 16, 16, 16), dtype=torch.float16, device=DEVICE, generator=g)
    t = torch.tensor([1000.0], dtype=torch.float16, device=DEVICE)
    stat(f"{tag} dense dit forward", M.dense_dit(noise, t, cond), pipe.dense_dit(noise, t, cond))
    lat_up = up_sample(pipe, pipe.dense_dit, pipe.dense_scheduler, noise, cond, 50, 7.0)
    lat_mine = lp.sample(M.dense_dit, noise, cond, steps=50, guidance=7.0, shift=M.shifts[0])
    stat(f"{tag} dense latent (50 steps)", lat_mine, lat_up)
    vox_up = pipe.dense_vae.decode_mesh(lat_up, return_index=True, mc_threshold=0.1)[0]
    vox_mine = M.dense_vae.occupied_voxels(lat_up, 0.1)[0]
    same_voxels(f"{tag} dense voxels (same latent)", vox_mine, vox_up)
    vox = up_sort_block(vox_up, 8)
    assert torch.equal(ls.sort_block(vox, 8), vox)

    # Sparse 512.
    noise = torch.randn((len(vox), 16), dtype=torch.float16, device=DEVICE, generator=g)
    t = torch.tensor([600.0], dtype=torch.float16, device=DEVICE)
    a = pipe.sparse_dit_512(up_sp.SparseTensor(noise, vox.int()), t, cond).feats
    b = M.sparse_dit_512(ls.SparseTensor(noise, vox.int()), t, cond).feats
    stat(f"{tag} sparse512 dit forward ({len(vox)} voxels)", b, a)
    lat_up = up_sample(pipe, pipe.sparse_dit_512, pipe.sparse_scheduler_512, noise, cond, 30, 7.0, vox)
    lat_mine = lp.sample(M.sparse_dit_512, noise, cond, steps=30, guidance=7.0, shift=M.shifts[1], coords=vox.int())
    stat(f"{tag} sparse512 latent (30 steps)", lat_mine, lat_up)
    sdf_up, feat_up = pipe.sparse_vae_512.decode_mesh(
        up_sp.SparseTensor(lat_up, vox.int()), mc_threshold=0.2, return_feat=True
    )
    sdf_mine, feat_mine = M.sparse_vae_512.decoder(ls.SparseTensor(lat_up, vox.int()), factor=1.0, return_feat=True)
    ka, fa = keyed(sdf_up.coords, sdf_up.feats)
    kb, fb = keyed(sdf_mine.coords, sdf_mine.feats)
    note(f"{tag} sdf512 voxels equal", bool(torch.equal(ka, kb)))
    stat(f"{tag} sdf512 values", fb, fa)
    stat(f"{tag} sdf512 features", keyed(feat_mine.coords, feat_mine.feats)[1], keyed(feat_up.coords, feat_up.feats)[1])
    store = captured_marching_cubes(up_refiner)
    mesh_up = pipe.refiner.run(sdf_up, feat_up, mc_threshold=0.4)[0]
    volume = M.refiner.refine(
        ls.SparseTensor(sdf_up.feats, sdf_up.coords), ls.SparseTensor(feat_up.feats, feat_up.coords)
    )
    stat(f"{tag} refined 512^3 SDF", volume[0, 0], torch.as_tensor(store["volume"]))
    v_mine, f_mine = lp.marching_cubes(volume[0, 0].cpu().numpy(), 0.4)
    chamfer(f"{tag} mesh512 (same sdf)", v_mine, f_mine, mesh_up.vertices, mesh_up.faces)
    note(f"{tag} mesh512 upstream signed volume", signed_volume(mesh_up.vertices, mesh_up.faces))

    # 1024 voxelisation of the same 512 mesh.
    normalized = up_mesh.normalize_mesh(mesh_up.copy())
    idx_up = up_mesh.mesh2index(normalized, size=1024, factor=8)
    idx_mine = lp.mesh_to_voxels(lp.normalize_vertices(np.asarray(mesh_up.vertices)), mesh_up.faces, DEVICE)
    same_voxels(f"{tag} 1024 voxels (same mesh)", idx_mine, idx_up)
    vox = up_sort_block(idx_up, 8)

    # Sparse 1024.
    noise = torch.randn((len(vox), 16), dtype=torch.float16, device=DEVICE, generator=g)
    a = pipe.sparse_dit_1024(up_sp.SparseTensor(noise, vox.int()), t, cond).feats
    b = M.sparse_dit_1024(ls.SparseTensor(noise, vox.int()), t, cond).feats
    stat(f"{tag} sparse1024 dit forward ({len(vox)} voxels)", b, a)
    lat_up = up_sample(pipe, pipe.sparse_dit_1024, pipe.sparse_scheduler_1024, noise, cond, 15, 7.0, vox)
    lat_mine = lp.sample(M.sparse_dit_1024, noise, cond, steps=15, guidance=7.0, shift=M.shifts[2], coords=vox.int())
    stat(f"{tag} sparse1024 latent (15 steps)", lat_mine, lat_up)
    sdf_up = pipe.sparse_vae_1024.decode_mesh(
        up_sp.SparseTensor(lat_up, vox.int()), voxel_resolution=1024, mc_threshold=0.2, return_feat=True
    )
    sdf_mine = M.sparse_vae_1024.decoder(ls.SparseTensor(lat_up, vox.int()), factor=1.0)
    ka, fa = keyed(sdf_up.coords, sdf_up.feats)
    kb, fb = keyed(sdf_mine.coords, sdf_mine.feats)
    note(f"{tag} sdf1024 voxels equal", bool(torch.equal(ka, kb)))
    stat(f"{tag} sdf1024 values", fb, fa)
    store = captured_marching_cubes(up_refiner)
    mesh_up = pipe.refiner_1024.run(sdf_up, mc_threshold=0.2)[0]
    volume = M.refiner_1024.refine(ls.SparseTensor(sdf_up.feats, sdf_up.coords))
    reference = torch.as_tensor(store.pop("volume"))
    stat(f"{tag} signed 1024^3 SDF", volume[0, 0].cpu(), reference)
    del reference
    v_mine, f_mine = lp.marching_cubes(volume[0, 0].cpu().numpy(), 0.2)
    del volume
    chamfer(f"{tag} mesh1024 (same sdf)", v_mine, f_mine, mesh_up.vertices, mesh_up.faces)
    note(f"{tag} mesh1024 upstream signed volume", signed_volume(mesh_up.vertices, mesh_up.faces))
    torch.cuda.empty_cache()


def silhouettes(vertices, path, side=256):
    """Three orthographic point projections: looking down -x, -y and -z (rows up = +2nd axis)."""
    panels = []
    for drop in range(3):
        keep = [a for a in range(3) if a != drop]
        uv = (vertices[:, keep] * 0.5 + 0.5) if np.abs(vertices).max() > 0.6 else vertices[:, keep] + 0.5
        pix = np.clip((uv * (side - 1)).astype(int), 0, side - 1)
        panel = np.zeros((side, side), np.uint8)
        panel[side - 1 - pix[:, 1], pix[:, 0]] = 255
        panels.append(panel)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


@torch.no_grad()
def end_to_end(pipe, model, image_path, seed, resolution):
    tag = image_path.split("/")[-1]
    image = Image.open(image_path)
    torch.manual_seed(seed)
    started = time.perf_counter()
    mesh_up = pipe(image, sdf_resolution=resolution)["mesh"]
    up_seconds = time.perf_counter() - started
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = model(image, seed=seed, resolution=resolution)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**30
    mesh = result.meshes
    # Back from the object frame to upstream's for the comparison.
    v = mesh.vertices[0].cpu().numpy() * 2
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    f = mesh.faces.cpu().numpy()[:, ::-1]
    chamfer(f"{tag} end to end {resolution} (seed {seed})", v, f, mesh_up.vertices, mesh_up.faces)
    note(
        f"{tag} end to end {resolution} timing",
        {"upstream_s": round(up_seconds, 1), "libreyolo_s": round(seconds, 1), "libreyolo_peak_gib": round(peak, 1)},
    )
    extent = np.ptp(np.asarray(mesh_up.vertices), axis=0).tolist()
    note(f"{tag} upstream mesh extent xyz", extent)
    if resolution == 512:
        silhouettes(np.asarray(mesh_up.vertices), f"{tag}_upstream_views.png")
        mine = mesh.vertices[0].cpu().numpy()
        silhouettes(mine, f"{tag}_libreyolo_object_frame_views.png")
    torch.cuda.empty_cache()


def main():
    images = ARGS.images or [f"{ARGS.upstream}/assets/test/13.png"]
    conv_check()
    pipe = Direct3DS2Pipeline.from_pretrained(f"{ARGS.weights}/direct3d-s2-v-1-1")
    pipe.to("cuda")
    model = LibreDirect3DS2(checkpoint_dir=ARGS.weights, device="cuda")
    model._ensure_loaded()
    for path in images:
        stages(pipe, model, path, ARGS.seed)
        end_to_end(pipe, model, path, ARGS.seed, 512)
        end_to_end(pipe, model, path, ARGS.seed, 1024)
    with open(ARGS.out, "w") as handle:
        json.dump(REPORT, handle, indent=2)


if __name__ == "__main__":
    main()
