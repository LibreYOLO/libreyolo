"""Parity: LibreGen("trellis") (plain PyTorch) vs upstream TRELLIS (spconv + xformers), same GPU, same process.

Stage by stage, each side fed the same inputs: the preprocessed image (an
RGBA example, so neither side removes a background), the DINOv2 tokens, one
forward and the whole 25-step sampler of the sparse-structure flow, its
decoder and the occupied voxels, one forward of the structured-latent flow
(also with upstream's attention swapped for the port's, which isolates the
convolutions), its whole sampler, the Gaussian decoder, and end to end
through each side's own call. Measured on an RTX 3090 (torch 2.7.1, spconv
2.3.8, xformers 0.0.31), three examples: the preprocessed image, the whole
sparse-structure stage and the voxels are bitwise equal; the structured
latent differs by fp16 noise (mean 6e-4 after 25 steps); with the same
attention kernel the Gaussian decoder is bitwise equal, and with upstream's
xformers its outputs differ by attention-kernel noise as large as
upstream's own distance to an fp32 reference.

Prereqs (a CUDA machine; throwaway env, do NOT install into the main .venv):
    pip install -e ".[gen]" spconv-cu126 xformers easydict plyfile
    pip install https://github.com/EasternJournalist/utils3d/archive/9a4eb15e4021b67b12c460c7057d642626897ec8.zip
    git clone https://github.com/microsoft/TRELLIS && git -C TRELLIS checkout 193e0502b5cb3eb5997ec457d5340e95d09b158e
    hf download microsoft/TRELLIS-image-large --revision 25e0d31ffbebe4b5a97464dd851910efc3002d96 --local-dir weights/TRELLIS-image-large

Run (upstream's DINOv2 then avoids xformers, as the port's does):
    XFORMERS_DISABLED=1 python weights/parity_trellis.py --upstream TRELLIS --weights weights/TRELLIS-image-large
"""

import json
import os
import sys
import time

os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPCONV_ALGO", "native")
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("images", nargs="*")
ARGS = _parser.parse_args()
sys.path.insert(0, ARGS.upstream)
# Upstream imports renderers and mesh tools at package import; this gate needs none of them.
from unittest import mock

for _name in (
    "open3d",
    "kaolin",
    "kaolin.utils",
    "kaolin.utils.testing",
    "nvdiffrast",
    "nvdiffrast.torch",
    "diffoctreerast",
    "diff_gaussian_rasterization",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
    "trellis.representations.mesh.flexicubes",
    "trellis.representations.mesh.flexicubes.flexicubes",
):
    if _name not in sys.modules:
        sys.modules[_name] = mock.MagicMock()
import numpy as np
import torch
from PIL import Image

# Stages that must be bitwise equal, and mean-absolute-difference limits for
# the rest: about 3x what an RTX 3090 measured, so a real regression (wrong
# kernel layout, wrong cast, wrong sampler) fails while kernel noise passes.
BITWISE = ("image", "ss_flow_forward", "ss_sample", "ss_decoder")
LIMITS = {
    "dinov2": 1e-5,
    "slat_flow_forward": 3e-3,
    "slat_flow_forward_same_attention": 3e-3,
    "slat_sample": 3e-3,
    "gs_xyz": 3e-5,
    "gs_scales": 5e-6,
    "gs_rotations": 2e-3,
    "gs_opacities": 2e-3,
    "gs_sh": 3e-3,
}


def check(row):
    """Names of the stages outside their limit (shape mismatches included)."""
    failures = [k for k in BITWISE if not row[k].get("equal")]
    failures += [
        k
        for k, limit in LIMITS.items()
        if "mean" not in row[k]
        or not np.isfinite(row[k]["mean"])
        or row[k]["mean"] > limit
    ]
    if not row["voxels"]["same"]:
        failures.append("voxels")
    return failures


def cmp(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"shapes": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "scale": float(b.abs().max()),
    }


import trellis.modules.sparse.attention.modules as uattn
from trellis.modules import sparse as usp
from trellis.pipelines import TrellisImageTo3DPipeline

up = TrellisImageTo3DPipeline.from_pretrained(ARGS.weights)
up.cuda()

from libreyolo.models.gen.trellis import pipeline as P
from libreyolo.models.gen.trellis import sparse as ps
from libreyolo.models.gen.trellis.model import LibreTRELLIS

port = LibreTRELLIS(device="cuda")
M = port.pipeline

images = ARGS.images or [
    "typical_creature_dragon.png",
    "typical_building_castle.png",
    "typical_misc_crate.png",
]
report = []
for name in images:
    src = Image.open(os.path.join(ARGS.upstream, "assets", "example_image", name))
    row = {"case": name}
    up_img = up.preprocess_image(src)
    port_img = P.preprocess_image(src, M.birefnet)
    row["image"] = cmp(
        torch.from_numpy(np.array(port_img)), torch.from_numpy(np.array(up_img))
    )
    up_cond = up.get_cond([up_img])["cond"]
    port_cond = P.encode_image(M.dinov2, up_img, "cuda")
    row["dinov2"] = cmp(port_cond, up_cond)

    # Sparse structure flow: one forward, then the whole sampler, from upstream's tokens.
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(1, 8, 16, 16, 16, generator=g).cuda()
    t = torch.tensor([700.0], device="cuda")
    with torch.no_grad():
        row["ss_flow_forward"] = cmp(
            M.structure_flow(noise, t, up_cond),
            up.models["sparse_structure_flow_model"](noise, t, up_cond),
        )
    stage = dict(steps=25, guidance=5.0, interval=(0.5, 1.0), shift=3.0)
    z_port = P.sample_flow(M.structure_flow, noise, up_cond, **stage)
    z_up = up.sparse_structure_sampler.sample(
        up.models["sparse_structure_flow_model"],
        noise,
        cond=up_cond,
        neg_cond=torch.zeros_like(up_cond),
        steps=25,
        cfg_strength=5.0,
        cfg_interval=(0.5, 1.0),
        rescale_t=3.0,
        verbose=False,
    ).samples
    row["ss_sample"] = cmp(z_port, z_up)
    with torch.no_grad():
        occ_port = M.structure_decoder(z_up)
        occ_up = up.models["sparse_structure_decoder"](z_up)
    row["ss_decoder"] = cmp(occ_port, occ_up)
    coords = torch.argwhere(occ_up > 0)[:, [0, 2, 3, 4]].int()
    coords_port = torch.argwhere(occ_port > 0)[:, [0, 2, 3, 4]].int()
    row["voxels"] = {
        "upstream": int(coords.shape[0]),
        "port": int(coords_port.shape[0]),
        "same": bool(torch.equal(coords, coords_port)),
    }

    # SLat flow from the same voxels, noise and tokens.
    feats = torch.randn(coords.shape[0], 8, generator=g).cuda()
    with torch.no_grad():
        v_up = up.models["slat_flow_model"](
            usp.SparseTensor(feats=feats, coords=coords), t, up_cond
        )
        v_port = M.latent_flow(ps.SparseTensor(feats, coords), t, up_cond)
    row["slat_flow_forward"] = cmp(v_port.feats, v_up.feats)
    # Same, with upstream's sparse attention replaced by the port's: isolates convolution and glue.
    saved = (
        uattn.sparse_scaled_dot_product_attention,
        uattn.sparse_windowed_scaled_dot_product_self_attention,
    )

    def full(*args):
        if len(args) == 1:
            qkv = args[0]
            out = ps.sparse_self_attention(ps.SparseTensor(qkv.feats, qkv.coords)).feats
            return qkv.replace(out)
        q, kv = args
        out = ps.sparse_cross_attention(ps.SparseTensor(q.feats, q.coords), kv).feats
        return q.replace(out)

    def windowed(qkv, window_size, shift_window=(0, 0, 0)):
        out = ps.sparse_windowed_self_attention(
            ps.SparseTensor(qkv.feats, qkv.coords), window_size, shift_window[0]
        ).feats
        return qkv.replace(out)

    (
        uattn.sparse_scaled_dot_product_attention,
        uattn.sparse_windowed_scaled_dot_product_self_attention,
    ) = full, windowed
    with torch.no_grad():
        v_up_sdpa = up.models["slat_flow_model"](
            usp.SparseTensor(feats=feats, coords=coords), t, up_cond
        )
    row["slat_flow_forward_same_attention"] = cmp(v_port.feats, v_up_sdpa.feats)
    (
        uattn.sparse_scaled_dot_product_attention,
        uattn.sparse_windowed_scaled_dot_product_self_attention,
    ) = saved

    slat_port = P.sample_flow(
        M.latent_flow, ps.SparseTensor(feats, coords), up_cond, **stage
    )
    slat_up = up.slat_sampler.sample(
        up.models["slat_flow_model"],
        usp.SparseTensor(feats=feats, coords=coords),
        cond=up_cond,
        neg_cond=torch.zeros_like(up_cond),
        steps=25,
        cfg_strength=5.0,
        cfg_interval=(0.5, 1.0),
        rescale_t=3.0,
        verbose=False,
    ).samples
    row["slat_sample"] = cmp(slat_port.feats, slat_up.feats)

    # Gaussian decoder from upstream's latent (denormalised as upstream does).
    std = torch.tensor(up.slat_normalization["std"])[None].cuda()
    mean = torch.tensor(up.slat_normalization["mean"])[None].cuda()
    slat_up = slat_up * std + mean
    with torch.no_grad():
        g_up = up.models["slat_decoder_gs"](slat_up)[0]
        g_port = M.gaussian_decoder.to_gaussians(
            M.gaussian_decoder(ps.SparseTensor(slat_up.feats, slat_up.coords))
        )
    row["gs_xyz"] = cmp(g_port["xyz"], g_up.get_xyz)
    row["gs_scales"] = cmp(g_port["scales"], g_up.get_scaling)
    row["gs_rotations"] = cmp(g_port["rotations"], g_up.get_rotation)
    row["gs_opacities"] = cmp(g_port["opacities"], g_up.get_opacity.reshape(-1))
    row["gs_sh"] = cmp(g_port["sh"], g_up._features_dc)

    # End to end through the public call: port vs upstream run() with the same seed.
    t0 = time.time()
    torch.cuda.synchronize()
    out_up = up.run(up_img, seed=42, formats=["gaussian"], preprocess_image=False)[
        "gaussian"
    ][0]
    torch.cuda.synchronize()
    t_up = time.time() - t0
    t0 = time.time()
    out_port = P.run_stages(M, up_img, 42, structure=stage, latent=stage)
    torch.cuda.synchronize()
    t_port = time.time() - t0
    n_up, n_port = out_up.get_xyz.shape[0], out_port["gaussians"]["xyz"].shape[0]
    row["end_to_end"] = {
        "gaussians_upstream": n_up,
        "gaussians_port": n_port,
        "seconds_upstream": round(t_up, 1),
        "seconds_port": round(t_port, 1),
    }
    row["failures"] = check(row)
    print(json.dumps(row), flush=True)
    report.append(row)
passed = all(not row["failures"] for row in report)
json.dump(
    {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "limits": LIMITS,
        "all_pass": passed,
        "cases": report,
    },
    open("parity_trellis_report.json", "w"),
    indent=1,
)
print("peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
print("ALL PASS" if passed else "MISMATCH")
sys.exit(0 if passed else 1)
