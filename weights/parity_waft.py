"""WAFT parity: upstream princeton-vl/WAFT vs LibreYOLO on a frame sequence.

Run on a CUDA box from the LibreYOLO checkout, with an upstream WAFT clone and
its requirements (timm) installed. Upstream encoders are built without
pretrained downloads (the WAFT checkpoint holds every tensor).

    python weights/parity_waft.py --upstream /path/to/WAFT \
        --ckpt a1/tar-c-t.pth --libre weights/LibreWAFTm-flow.pt \
        --frames /path/to/frames --size m
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def build_upstream(upstream: str, size: str, ckpt: str, device: str):
    sys.path.insert(0, upstream)
    import timm

    real_create = timm.create_model
    timm.create_model = lambda *a, **k: real_create(*a, **{**k, "pretrained": False})
    real_hub = torch.hub.load

    def hub_load(repo, name, *a, **k):
        k.pop("weights", None)
        return real_hub(os.path.join(upstream, repo), name, *a, **{**k, "pretrained": False})

    torch.hub.load = hub_load
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

        os.makedirs("depth-anything-ckpts", exist_ok=True)
        torch.save(
            DepthAnythingV2("vits", 64, [48, 96, 192, 384]).state_dict(),
            "depth-anything-ckpts/depth_anything_v2_vits.pth",
        )
        if size == "m":
            from model.waft_a1 import ViTWarpV8

            args = SimpleNamespace(dav2_backbone="vits", network_backbone="vits", iters=5)
            net = ViTWarpV8(args)
        else:
            from model.waft_a2 import WAFTv2

            enc = {"t": "dinov3", "s": "dav2", "l": "twins"}[size]
            args = SimpleNamespace(feature_encoder=enc, iterative_module="vits", iters=5)
            net = WAFTv2(args)
    finally:
        os.chdir(cwd)
        timm.create_model = real_create
        torch.hub.load = real_hub
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"upstream load: missing={missing[:5]} unexpected={unexpected[:5]}")
    return net.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--libre", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--save-flows", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"

    frames = sorted(glob.glob(os.path.join(a.frames, "*.png")))
    sys.path.insert(0, os.getcwd())
    from libreyolo import LibreYOLO

    libre = LibreYOLO(a.libre, device=device)
    results = list(libre(frames, stream=True))
    ours = [r.flow.data.cpu().numpy() if torch.is_tensor(r.flow.data) else r.flow.data for r in results]

    up = build_upstream(a.upstream, a.size, a.ckpt, device)
    def load(f):
        arr = np.asarray(Image.open(f).convert("RGB")).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)

    maxd, meand, epes = [], [], []
    with torch.no_grad():
        for i in range(len(frames) - 1):
            ref = up(load(frames[i]), load(frames[i + 1]), iters=5)["flow"][-1][0].permute(1, 2, 0).cpu().numpy()
            d = np.abs(ref - ours[i])
            maxd.append(float(d.max()))
            meand.append(float(d.mean()))
            epes.append(float(np.linalg.norm(ref - ours[i], axis=-1).mean()))
    out = {
        "pairs": len(maxd),
        "max_abs_diff_px": max(maxd),
        "mean_abs_diff_px": float(np.mean(meand)),
        "mean_epe_between_px": float(np.mean(epes)),
        "mean_flow_magnitude_px": float(np.mean([np.linalg.norm(f, axis=-1).mean() for f in ours])),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)
    if a.save_flows:
        np.save(a.save_flows, np.stack(ours))


if __name__ == "__main__":
    main()
