"""Parity: LibreGen("trellis2") (plain PyTorch) vs upstream TRELLIS.2 (FlexGEMM, flash-attn, o-voxel), same GPU, same process.

Geometry path, stage by stage, each side fed the same inputs (upstream's):
the preprocessed image (an RGBA example, so neither side removes a
background), the DINOv3 tokens at 512 and 1024, one forward and the whole
sampler of the sparse-structure flow, its decoder and the occupied voxels,
one forward and the whole sampler of the 512 and 1024 shape flows, the
decoder's predicted voxels for the cascade, the decoder's mesh fields, the
dual-grid meshing. End to end (each side's own stages, same seed, one process
per side so each has the GPU to itself) is weights/parity_trellis2_e2e.py. Every reduced-precision network is also run in fp32 (the port's
networks with fp32 weights, from the same files) as a reference, so each
stage reports how far port and upstream each are from it.

Upstream's dense attention runs once with its flash-attn backend and once
with its own SDPA backend (``*_sdpa``): with SDPA the dense stages must be
bitwise equal. Its sparse stages have no SDPA backend; ``*_same_attention``
swaps the port's sparse attention into upstream, which isolates the
convolutions (FlexGEMM vs the gathered GEMM) and the glue.

Measured on an A100 40 GB (torch 2.6.0+cu124, flash-attn 2.7.3, transformers
4.57.1): every stage in BITWISE is bitwise equal; the flash-attn / FlexGEMM
stages are as close to the fp32 reference as upstream is (sparse-structure
sampler mean |diff| to fp32: port 0.018, upstream 0.023; 1024 shape sampler:
0.0518 vs 0.0520); the decoder's predicted voxels overlap upstream's at IoU
0.998, as upstream's overlap the fp32 run's.

Prereqs (a CUDA machine with nvcc; throwaway env, NOT the main .venv):
    pip install -e ".[gen]" transformers==4.57.1 flash-attn==2.7.3 easydict opencv-python-headless trimesh zstandard plyfile
    git clone --recursive https://github.com/microsoft/TRELLIS.2 && git -C TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
    pip install --no-build-isolation ./TRELLIS.2/o-voxel git+https://github.com/JeffreyXiang/FlexGEMM@6dd94a859c26ee8246888502eada3dd8ad85532e
    hf download microsoft/TRELLIS.2-4B --revision af44b45f2e35a493886929c6d786e563ec68364d --local-dir weights/TRELLIS.2-4B

Run:
    python weights/parity_trellis2.py --upstream TRELLIS.2 --weights weights/TRELLIS.2-4B
"""

import argparse
import json
import os
import sys
import time

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--report", default="parity_trellis2_report.json")
_parser.add_argument("images", nargs="*")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)

# Upstream imports renderers and mesh tools at package import. This gate
# needs none of them: nvdiffrast (NVIDIA's non-commercial licence) is not
# installed, CuMesh only runs hole filling after the geometry compared here.
from unittest import mock

for _name in (
    "nvdiffrast",
    "nvdiffrast.torch",
    "cumesh",
    "utils3d",
    "utils3d.torch",
    "lpips",
    "spconv",
    "spconv.pytorch",
):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = mock.MagicMock()

from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

# Stages that must be bitwise equal.
BITWISE = (
    "image",
    "dinov3_512",
    "dinov3_1024",
    "ss_flow_forward_sdpa",
    "ss_sample_sdpa",
    "ss_decoder",
    "mesh_from_upstream_fields",
)


def cmp(a, b, ref=None):
    """Port vs upstream; with ``ref``, both against the fp32 reference."""
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"shapes": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    out = {
        "equal": bool(torch.equal(a, b)),
        "max": float(d.max()) if d.numel() else 0.0,
        "mean": float(d.mean()) if d.numel() else 0.0,
        "scale": float(b.abs().max()) if b.numel() else 0.0,
    }
    if ref is not None:
        ref = ref.detach().float().cpu()
        out["port_vs_fp32_mean"] = float((a - ref).abs().mean())
        out["upstream_vs_fp32_mean"] = float((b - ref).abs().mean())
    return out


def coord_sets(a, b):
    """Overlap of two voxel sets: counts, intersection over union, and the row maps."""
    a, b = a.cpu().long(), b.cpu().long()
    size = int(max(a.max(), b.max())) + 1

    def key(c):
        return ((c[:, 0] * size + c[:, 1]) * size + c[:, 2]) * size + c[:, 3]

    ka, kb = key(a), key(b)
    common = np.intersect1d(ka.numpy(), kb.numpy())
    return {
        "port": int(a.shape[0]),
        "upstream": int(b.shape[0]),
        "iou": float(len(common) / len(np.union1d(ka.numpy(), kb.numpy()))),
        "same": bool(torch.equal(a, b)),
    }


# -- Upstream ----------------------------------------------------------------

import trellis2.modules.attention.config as uattn_config
import trellis2.modules.sparse.attention.modules as usattn
import trellis2.pipelines.rembg as urembg
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder as UDecoder
from trellis2.modules import sparse as usp
from trellis2.pipelines import Trellis2ImageTo3DPipeline


class _NoRembg:
    """Upstream's default remover is BRIA's RMBG-2.0 (non-commercial); the examples carry alpha."""

    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    cpu = cuda = to


urembg.BiRefNet = _NoRembg
def geometry_only_snapshot(weights):
    """A temporary copy of upstream's snapshot for this gate: ckpts/ linked, pipeline.json edited.

    Upstream reads Meta's gated DINOv3 repo; LibreYOLO's byte-identical mirror
    needs no login. The texture networks are neither downloaded nor loaded.
    The snapshot given on the command line is left untouched.
    """
    import tempfile

    folder = tempfile.mkdtemp(prefix="trellis2_parity_")
    os.symlink(os.path.abspath(os.path.join(weights, "ckpts")), os.path.join(folder, "ckpts"))
    cfg = json.load(open(os.path.join(weights, "pipeline.json")))
    cfg["args"]["image_cond_model"]["args"]["model_name"] = "LibreYOLO/LibreDINOv3vitl16"
    for key in ("tex_slat_flow_model_512", "tex_slat_flow_model_1024", "tex_slat_decoder"):
        cfg["args"]["models"].pop(key, None)
    json.dump(cfg, open(os.path.join(folder, "pipeline.json"), "w"), indent=4)
    return folder

up = Trellis2ImageTo3DPipeline.from_pretrained(geometry_only_snapshot(ARGS.weights))
# A 40 GB GPU holds the port's networks plus one other network at a time:
# upstream's and the fp32 reference's stay on the CPU and move to the GPU for
# each call (upstream's own low-VRAM mode does the same in its pipeline).
up.low_vram = True
up.to(torch.device("cuda"))


@contextmanager
def gpu(module):
    module = module.m if isinstance(module, OnGPU) else module
    module.cuda()
    try:
        yield module
    finally:
        module.cpu()
        torch.cuda.empty_cache()


class OnGPU:
    """A CPU-resident network that runs on the GPU for each call."""

    _METHODS = ("upsample", "decode_fields")

    def __init__(self, module):
        object.__setattr__(self, "m", module)

    def __call__(self, *args, **kwargs):
        with gpu(self.m) as m:
            return m(*args, **kwargs)

    def __setattr__(self, name, value):
        setattr(self.m, name, value)

    def __getattr__(self, name):
        value = getattr(self.m, name)
        if name in self._METHODS:

            def run(*args, **kwargs):
                with gpu(self.m):
                    return value(*args, **kwargs)

            return run
        return value


for _name in list(up.models):
    up.models[_name] = OnGPU(up.models[_name].cpu())
UM = up.models

# -- Port --------------------------------------------------------------------

from libreyolo import LibreGen
from libreyolo.models.gen.trellis import sparse as ps
from libreyolo.models.gen.trellis2 import mesh as pmesh
from libreyolo.models.gen.trellis2 import pipeline as P

port = LibreGen(ARGS.weights, device="cuda")
M = port.pipeline


# fp32 reference: the port built for the CPU loads every network in fp32
# from the same files (published bf16/fp16 weights, exact in fp32).
_CPU = LibreGen(ARGS.weights, device="cpu").pipeline
REF = {
    name: OnGPU(getattr(_CPU, name))
    for name in ("structure_flow", "structure_decoder", "shape_flow_512", "shape_flow_1024", "shape_decoder")
}


def with_attention(backend, fn):
    saved = uattn_config.BACKEND
    uattn_config.BACKEND = backend
    try:
        return fn()
    finally:
        uattn_config.BACKEND = saved


def port_sparse_attention(*args):
    """Upstream's sparse SDPA entry point, served by the port's attention."""
    if len(args) == 1:
        qkv = args[0]
        out = ps.sparse_self_attention(ps.SparseTensor(qkv.feats, qkv.coords)).feats
        return qkv.replace(out)
    q, k, v = args
    parts = [
        ps._sdpa(q.feats[s][None], k[i : i + 1], v[i : i + 1])[0]
        for i, s in enumerate(q.layout)
    ]
    return q.replace(torch.cat(parts))


def with_port_sparse_attention(fn):
    saved = usattn.sparse_scaled_dot_product_attention
    usattn.sparse_scaled_dot_product_attention = port_sparse_attention
    try:
        return fn()
    finally:
        usattn.sparse_scaled_dot_product_attention = saved


def upstream_sample(sampler, model, noise, cond, params, **kw):
    return sampler.sample(
        model, noise, cond=cond, neg_cond=torch.zeros_like(cond), **params, verbose=False, **kw
    ).samples


def port_params(params):
    return P.sampler_settings(params, params["steps"], params["guidance_strength"])


images = ARGS.images or ["T.png", "0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp"]
report = []
for name in images:
    src = Image.open(os.path.join(ARGS.upstream, "assets", "example_image", name))
    if src.mode != "RGBA":
        print(f"skip {name}: no alpha channel", flush=True)
        continue
    row = {"case": name}
    t_case = time.time()
    up_img = up.preprocess_image(src)
    port_img = P.preprocess_image(src, M.birefnet)
    row["image"] = cmp(torch.from_numpy(np.array(port_img)), torch.from_numpy(np.array(up_img)))

    conds = {}
    for size in (512, 1024):
        up_cond = up.get_cond([up_img], size)["cond"]
        row[f"dinov3_{size}"] = cmp(P.encode_image(M.dinov3, up_img, size, "cuda"), up_cond)
        conds[size] = up_cond

    # Sparse structure flow.
    ss_params = up.sparse_structure_sampler_params
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(1, 8, 16, 16, 16, generator=g).cuda()
    t = torch.tensor([700.0], device="cuda")
    with torch.no_grad():
        ref = REF["structure_flow"](noise, t, conds[512])
        row["ss_flow_forward"] = cmp(
            M.structure_flow(noise, t, conds[512]), UM["sparse_structure_flow_model"](noise, t, conds[512]), ref
        )
        row["ss_flow_forward_sdpa"] = cmp(
            M.structure_flow(noise, t, conds[512]),
            with_attention("sdpa", lambda: UM["sparse_structure_flow_model"](noise, t, conds[512])),
        )
    z_ref = P.sample_flow(REF["structure_flow"], noise, conds[512], **port_params(ss_params))
    z_port = P.sample_flow(M.structure_flow, noise, conds[512], **port_params(ss_params))
    z_up = upstream_sample(up.sparse_structure_sampler, UM["sparse_structure_flow_model"], noise, conds[512], ss_params)
    row["ss_sample"] = cmp(z_port, z_up, z_ref)
    z_up_sdpa = with_attention(
        "sdpa",
        lambda: upstream_sample(up.sparse_structure_sampler, UM["sparse_structure_flow_model"], noise, conds[512], ss_params),
    )
    row["ss_sample_sdpa"] = cmp(z_port, z_up_sdpa)
    with torch.no_grad():
        occ_port = M.structure_decoder(z_up)
        occ_up = UM["sparse_structure_decoder"](z_up)
    row["ss_decoder"] = cmp(occ_port, occ_up)
    decoded = torch.nn.functional.max_pool3d((occ_up > 0).float(), 2, 2, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    decoded_p = torch.nn.functional.max_pool3d((occ_port > 0).float(), 2, 2, 0) > 0.5
    row["ss_voxels"] = coord_sets(torch.argwhere(decoded_p)[:, [0, 2, 3, 4]].int(), coords)

    # Shape flow at 512 on upstream's voxels.
    sh_params = up.shape_slat_sampler_params
    feats = torch.randn(coords.shape[0], 32, generator=g).cuda()
    with torch.no_grad():
        v_up = UM["shape_slat_flow_model_512"](usp.SparseTensor(feats=feats, coords=coords), t, conds[512])
        v_port = M.shape_flow_512(ps.SparseTensor(feats, coords), t, conds[512])
        v_ref = REF["shape_flow_512"](ps.SparseTensor(feats, coords), t, conds[512])
        v_up_same = with_port_sparse_attention(
            lambda: UM["shape_slat_flow_model_512"](usp.SparseTensor(feats=feats, coords=coords), t, conds[512])
        )
    row["shape512_flow_forward"] = cmp(v_port.feats, v_up.feats, v_ref.feats)
    row["shape512_flow_forward_same_attention"] = cmp(v_port.feats, v_up_same.feats)
    lr_ref = P.sample_flow(REF["shape_flow_512"], ps.SparseTensor(feats, coords), conds[512], **port_params(sh_params))
    lr_port = P.sample_flow(M.shape_flow_512, ps.SparseTensor(feats, coords), conds[512], **port_params(sh_params))
    lr_up = upstream_sample(
        up.shape_slat_sampler, UM["shape_slat_flow_model_512"], usp.SparseTensor(feats=feats, coords=coords), conds[512], sh_params
    )
    row["shape512_sample"] = cmp(lr_port.feats, lr_up.feats, lr_ref.feats)

    # Cascade voxels: the decoder's 4 predicted subdivisions of upstream's 512 latent.
    norm = up.shape_slat_normalization
    std = torch.tensor(norm["std"])[None].cuda()
    mean = torch.tensor(norm["mean"])[None].cuda()
    lr_up = lr_up * std + mean
    with torch.no_grad():
        hr_up = UM["shape_slat_decoder"].upsample(lr_up, upsample_times=4)
        hr_port = M.shape_decoder.upsample(ps.SparseTensor(lr_up.feats, lr_up.coords), upsample_times=4)
        hr_ref = REF["shape_decoder"].upsample(ps.SparseTensor(lr_up.feats, lr_up.coords), upsample_times=4)
    row["cascade_hr_voxels"] = coord_sets(hr_port, hr_up)
    row["cascade_hr_voxels_fp32"] = {
        "port": coord_sets(hr_port, hr_ref)["iou"],
        "upstream": coord_sets(hr_up, hr_ref)["iou"],
    }
    hr_coords, hr_res = P._cascade_coords(hr_up, 512, 1024, 49152)
    row["cascade_voxels"] = {"count": int(hr_coords.shape[0]), "resolution": hr_res}

    # Shape flow at 1024 on those voxels.
    feats = torch.randn(hr_coords.shape[0], 32, generator=g).cuda()
    with torch.no_grad():
        v_up = UM["shape_slat_flow_model_1024"](usp.SparseTensor(feats=feats, coords=hr_coords), t, conds[1024])
        v_port = M.shape_flow_1024(ps.SparseTensor(feats, hr_coords), t, conds[1024])
        v_ref = REF["shape_flow_1024"](ps.SparseTensor(feats, hr_coords), t, conds[1024])
    row["shape1024_flow_forward"] = cmp(v_port.feats, v_up.feats, v_ref.feats)
    hr_ref_s = P.sample_flow(REF["shape_flow_1024"], ps.SparseTensor(feats, hr_coords), conds[1024], **port_params(sh_params))
    hr_port_s = P.sample_flow(M.shape_flow_1024, ps.SparseTensor(feats, hr_coords), conds[1024], **port_params(sh_params))
    hr_up_s = upstream_sample(
        up.shape_slat_sampler, UM["shape_slat_flow_model_1024"], usp.SparseTensor(feats=feats, coords=hr_coords), conds[1024], sh_params
    )
    row["shape1024_sample"] = cmp(hr_port_s.feats, hr_up_s.feats, hr_ref_s.feats)
    slat = hr_up_s * std + mean

    # Decoder fields on upstream's latent, then meshing.
    UM["shape_slat_decoder"].set_resolution(hr_res)
    M.shape_decoder.set_resolution(hr_res)
    REF["shape_decoder"].set_resolution(hr_res)
    with torch.no_grad():
        with gpu(UM["shape_slat_decoder"]) as dec:
            h_up, _ = UDecoder.forward(dec, slat, return_subs=True)
        fields_port, _ = M.shape_decoder.decode_fields(ps.SparseTensor(slat.feats, slat.coords))
        fields_ref, _ = REF["shape_decoder"].decode_fields(ps.SparseTensor(slat.feats, slat.coords))
        meshes_up = UM["shape_slat_decoder"](slat)
    row["decoder_voxels"] = coord_sets(fields_port["coords"], h_up.coords)
    row["decoder_voxels_fp32"] = {
        "port": coord_sets(fields_port["coords"], fields_ref["coords"])["iou"],
        "upstream": coord_sets(h_up.coords, fields_ref["coords"])["iou"],
    }
    margin = UM["shape_slat_decoder"].voxel_margin
    up_fields = {
        "dual_vertices": (1 + 2 * margin) * torch.sigmoid(h_up.feats[:, 0:3]) - margin,
        "intersected": h_up.feats[:, 3:6] > 0,
        "split_weight": torch.nn.functional.softplus(h_up.feats[:, 6:7]),
    }
    if row["decoder_voxels"]["same"]:
        row["decoder_dual_vertices"] = cmp(fields_port["dual_vertices"], up_fields["dual_vertices"], fields_ref["dual_vertices"])
        row["decoder_intersected_agree"] = float(
            (fields_port["intersected"] == up_fields["intersected"]).float().mean()
        )
    v, f = pmesh.flexible_dual_grid_to_mesh(
        h_up.coords[:, 1:], up_fields["dual_vertices"], up_fields["intersected"], up_fields["split_weight"], hr_res
    )
    row["mesh_from_upstream_fields"] = {
        "equal": bool(torch.equal(v.cpu(), meshes_up[0].vertices.cpu()) and torch.equal(f.cpu().long(), meshes_up[0].faces.cpu().long())),
        "faces": [int(f.shape[0]), int(meshes_up[0].faces.shape[0])],
    }

    row["failures"] = [k for k in BITWISE if not row[k].get("equal")]
    row["seconds"] = round(time.time() - t_case, 1)
    print(json.dumps(row), flush=True)
    report.append(row)

passed = bool(report) and all(not row["failures"] for row in report)
json.dump(
    {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "upstream_commit": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        "weights_revision": "af44b45f2e35a493886929c6d786e563ec68364d",
        "bitwise_stages": BITWISE,
        "all_bitwise_pass": passed,
        "cases": report,
    },
    open(ARGS.report, "w"),
    indent=1,
)
print("peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
print("ALL BITWISE STAGES PASS" if passed else "MISMATCH")
sys.exit(0 if passed else 1)
