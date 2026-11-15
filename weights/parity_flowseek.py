"""FlowSeek parity: upstream mattpoggi/flowseek vs LibreYOLO on a frame sequence.

Runs on an NVIDIA GPU in fp32 (TF32 off). Upstream is imported from a local
clone (``--upstream``, run ``scripts/get_weights.sh`` there first). For every
size the N frames give N-1 consecutive pairs; the LibreYOLO side goes through
the public predict path (``model(frames)``) so preprocessing, the runner pair
window and postprocess are all covered.

    python weights/parity_flowseek.py --upstream /path/flowseek \
        --frames /path/frames --weights-dir /path/converted --save-dir /path/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_CKPT = {"t": "flowseek_T_TartanCT_TSKH.pth", "s": "flowseek_T_TartanCT_TSKH.pth",
                 "m": "flowseek_M_TartanCT_TSKH.pth", "l": "flowseek_M_TartanCT_TSKH.pth"}
UPSTREAM_CFG = {"t": "T", "s": "S", "m": "M", "l": "L"}


def load_upstream(upstream: Path, size: str):
    sys.path.insert(0, str(upstream / "core"))
    sys.path.insert(0, str(upstream))
    from config.parser import json_to_args  # noqa: E402
    from flowseek import FlowSeek  # noqa: E402

    args = json_to_args(str(upstream / "config/eval" / f"flowseek-{UPSTREAM_CFG[size]}.json"))
    cwd = os.getcwd()
    os.chdir(upstream)  # upstream reads weights/depth_anything_v2_*.pth relative to cwd
    try:
        model = FlowSeek(args)
        sd = torch.load(upstream / "weights" / UPSTREAM_CKPT[size], map_location="cpu")
        model.load_state_dict(sd, strict=False)  # upstream utils.load_ckpt
    finally:
        os.chdir(cwd)
    return model.cuda().eval(), args.iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--frames", required=True, type=Path)
    p.add_argument("--weights-dir", required=True, type=Path)
    p.add_argument("--sizes", default="t,s,m,l")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument(
        "--manual-attn",
        action="store_true",
        help="Run the vendored DINOv2 attention as q @ k.T -> softmax -> @ v (upstream's math) instead of SDPA",
    )
    a = p.parse_args()
    if a.manual_attn:
        import libreyolo.models.depth_anything._vendor.dinov2_layers.attention as attn

        attn.manual_attention_required = lambda: True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from libreyolo import LibreYOLO

    frames = sorted(a.frames.glob("*.png"))
    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    report = {}
    for size in a.sizes.split(","):
        up, iters = load_upstream(a.upstream, size)
        ours = LibreYOLO(str(a.weights_dir / f"LibreFlowSeek{size}-flow.pt"), device="cuda")
        assert ours.iterations == iters, (ours.iterations, iters)
        results = ours([str(f) for f in frames])
        assert len(results) == len(frames) - 1
        maxd, meand, epes, ours_all, p999 = [], [], [], [], []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                t1 = torch.from_numpy(imgs[i]).permute(2, 0, 1).float()[None].cuda()
                t2 = torch.from_numpy(imgs[i + 1]).permute(2, 0, 1).float()[None].cuda()
                ref = up(t1, t2, iters=iters, test_mode=True)["final"][0].permute(1, 2, 0).cpu().numpy()
                got = np.asarray(results[i].flow.data.cpu() if torch.is_tensor(results[i].flow.data) else results[i].flow.data)
                d = np.abs(got - ref)
                maxd.append(float(d.max()))
                meand.append(float(d.mean()))
                p999.append(float(np.percentile(d, 99.9)))
                epes.append(float(np.linalg.norm(got - ref, axis=-1).mean()))
                ours_all.append(got)
        report[size] = {
            "pairs": len(maxd),
            "iters": iters,
            "max_abs_diff_px": max(maxd),
            "mean_abs_diff_px": float(np.mean(meand)),
            "p99.9_abs_diff_px": float(np.max(p999)),
            "pairs_max_over_1e-2": int(sum(m > 1e-2 for m in maxd)),
            "mean_epe_between_px": float(np.mean(epes)),
            "mean_flow_magnitude_px": float(np.mean([np.linalg.norm(f, axis=-1).mean() for f in ours_all])),
        }
        print(size, json.dumps(report[size]))
        if a.save_dir is not None:
            a.save_dir.mkdir(parents=True, exist_ok=True)
            np.save(a.save_dir / f"flowseek_{size}_flows.npy", np.stack(ours_all).astype(np.float32))
        del up, ours
        torch.cuda.empty_cache()
    if a.json is not None:
        a.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
