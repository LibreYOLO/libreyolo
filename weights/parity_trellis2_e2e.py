"""End-to-end parity for LibreGen("trellis2"): one image, one seed, each side's own stages, one process per side.

The stage-by-stage gate is weights/parity_trellis2.py. Here each side runs
the whole geometry path from the same preprocessed image and seed, alone on
the GPU (the 1024 decode of a detailed object needs most of a 40 GB card):

    python weights/parity_trellis2_e2e.py upstream --upstream TRELLIS.2 --weights weights/TRELLIS.2-4B IMAGE
    python weights/parity_trellis2_e2e.py port     --upstream TRELLIS.2 --weights weights/TRELLIS.2-4B IMAGE
    python weights/parity_trellis2_e2e.py fp32     --upstream TRELLIS.2 --weights weights/TRELLIS.2-4B IMAGE
    python weights/parity_trellis2_e2e.py compare  --upstream TRELLIS.2 --weights weights/TRELLIS.2-4B IMAGE

``fp32`` is the port with every network in fp32 (the same files, cast), the
reference both sides are measured against. Sampling is chaotic: a flipped
subdivision moves whole voxels, so the meshes are compared by voxel overlap
at 32^3, face count and a symmetric surface distance, not element-wise.
Prereqs: as weights/parity_trellis2.py.
"""

import argparse
import json
import os
import sys
import time

_parser = argparse.ArgumentParser()
_parser.add_argument("side", choices=("upstream", "port", "fp32", "compare"))
_parser.add_argument("--upstream", required=True)
_parser.add_argument("--weights", required=True)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--resolution", type=int, default=1024)
_parser.add_argument("--out", default="parity_trellis2_e2e")
_parser.add_argument("image")
ARGS = _parser.parse_args()

import torch
from PIL import Image

os.makedirs(ARGS.out, exist_ok=True)
SRC = Image.open(os.path.join(ARGS.upstream, "assets", "example_image", ARGS.image))
PIPELINE = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}[ARGS.resolution]


def path(side):
    return os.path.join(ARGS.out, f"{os.path.splitext(ARGS.image)[0]}_{ARGS.resolution}_{side}.pt")


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

@torch.no_grad()
def chamfer(va, fa, vb, fb, n=100_000):
    """Symmetric mean distance between face-centroid samples of two meshes (object units)."""
    g = torch.Generator(device="cuda").manual_seed(0)

    def sample(v, f):
        c = v.cuda()[f.cuda().long()].mean(dim=1)
        idx = torch.randint(0, c.shape[0], (min(n, c.shape[0]),), device="cuda", generator=g)
        return c[idx].float()

    a, b = sample(va, fa), sample(vb, fb)

    def one_way(x, y):
        return torch.cat(
            [torch.cdist(x[i : i + 512], y).min(dim=1).values for i in range(0, len(x), 512)]
        ).mean()

    return float((one_way(a, b) + one_way(b, a)) / 2)


if ARGS.side == "compare":
    # A side that failed (upstream runs out of memory at 1024 on a 40 GB card) is reported as missing.
    sides = {s: torch.load(path(s)) for s in ("upstream", "port", "fp32") if os.path.exists(path(s))}

    def voxels_iou(a, b):
        ka = {tuple(r) for r in a.tolist()}
        kb = {tuple(r) for r in b.tolist()}
        return len(ka & kb) / len(ka | kb)

    out = {"image": ARGS.image, "seed": ARGS.seed, "resolution": ARGS.resolution}
    out["missing"] = [s for s in ("upstream", "port", "fp32") if s not in sides]
    for s, d in sides.items():
        out[s] = {"faces": int(d["faces"].shape[0]), "seconds": d["seconds"], "peak_gib": d["peak_gib"]}
    for a, b in (("port", "upstream"), ("port", "fp32"), ("upstream", "fp32")):
        if a not in sides or b not in sides:
            continue
        out[f"voxels_32_iou_{a}_{b}"] = voxels_iou(sides[a]["coords"], sides[b]["coords"])
        out[f"chamfer_{a}_{b}"] = chamfer(
            sides[a]["vertices"], sides[a]["faces"], sides[b]["vertices"], sides[b]["faces"]
        )
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(ARGS.out, "report.json"), "w"), indent=1)
    sys.exit(0)

torch.cuda.reset_peak_memory_stats()
if ARGS.side == "upstream":
    sys.path.insert(0, ARGS.upstream)
    from unittest import mock

    for _name in ("nvdiffrast", "nvdiffrast.torch", "cumesh", "utils3d", "utils3d.torch", "lpips", "spconv", "spconv.pytorch"):
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = mock.MagicMock()
    import trellis2.pipelines.rembg as urembg
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    class _NoRembg:
        def __init__(self, *a, **k):
            pass

        def to(self, *a, **k):
            return self

        cpu = cuda = to

    urembg.BiRefNet = _NoRembg
    up = Trellis2ImageTo3DPipeline.from_pretrained(geometry_only_snapshot(ARGS.weights))
    up.cuda()  # low_vram: models move to the GPU stage by stage
    image = up.preprocess_image(SRC)
    torch.cuda.synchronize()
    t0 = time.time()
    torch.manual_seed(ARGS.seed)
    c512 = up.get_cond([image], 512)
    c1024 = up.get_cond([image], 1024) if ARGS.resolution != 512 else None
    coords = up.sample_sparse_structure(c512, 32, 1, {})
    if ARGS.resolution == 512:
        slat = up.sample_shape_slat(c512, up.models["shape_slat_flow_model_512"], coords, {})
        res = 512
    else:
        slat, res = up.sample_shape_slat_cascade(
            c512, c1024, up.models["shape_slat_flow_model_512"], up.models["shape_slat_flow_model_1024"],
            512, ARGS.resolution, coords, {}, 49152,
        )
    mesh = up.decode_shape_slat(slat, res)[0][0]
    torch.cuda.synchronize()
    result = {"coords": coords.cpu(), "vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu()}
else:
    from libreyolo import LibreGen
    from libreyolo.models.gen.trellis2 import pipeline as P

    if ARGS.side == "port":
        modules = LibreGen(ARGS.weights, device="cuda").pipeline
    else:
        # fp32 reference: the CPU build loads every network in fp32, then moves to the GPU.
        modules = LibreGen(ARGS.weights, device="cpu").pipeline
        for name in ("dinov3", "birefnet", "structure_flow", "structure_decoder", "shape_flow_512", "shape_flow_1024", "shape_decoder"):
            getattr(modules, name).cuda()
        modules.device = torch.device("cuda")
    image = P.preprocess_image(SRC, modules.birefnet)
    params = json.load(open(os.path.join(ARGS.weights, "pipeline.json")))["args"]
    structure = P.sampler_settings(params["sparse_structure_sampler"]["params"], 12, 7.5)
    shape = P.sampler_settings(params["shape_slat_sampler"]["params"], 12, 7.5)
    torch.cuda.synchronize()
    t0 = time.time()
    out = P.run_stages(modules, image, ARGS.seed, resolution=ARGS.resolution, structure=structure, shape=shape)
    torch.cuda.synchronize()
    result = {"coords": out["coords"].cpu(), "vertices": out["vertices"].cpu(), "faces": out["faces"].cpu()}
result["seconds"] = round(time.time() - t0, 1)
result["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
torch.save(result, path(ARGS.side))
print(json.dumps({"side": ARGS.side, "faces": int(result["faces"].shape[0]), "seconds": result["seconds"], "peak_gib": result["peak_gib"]}), flush=True)
