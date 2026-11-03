"""Upstream MEMFOF vs LibreYOLO parity on a frame sequence (run on a GPU box).

For every interior frame t of the sequence (frames 1 .. N-2, 0-indexed) the
upstream model gets the (t-1, t, t+1) window exactly as its README shows
(float [0, 255], ``[B=1, T=3, C=3, H, W]``, ``model(x)["flow"][-1]``, forward
flow at index 1) and LibreYOLO gets the whole sequence through
``LibreYOLO(ckpt)(frames)``, whose result t is the forward flow of frame t
from the same window. Reports max / mean absolute difference in pixels and
the mean end-point error between the two fields.

Usage::

    PYTHONPATH=/path/to/upstream/memfof python weights/parity_memfof.py \
        --frames /workspace/clip/frames --upstream-dir /workspace/memfof_w/Tartan-T-TSKH \
        --ckpt /workspace/out/memfof_ckpt/LibreMEMFOFm-flow.pt [--bf16]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--bf16", action="store_true", help="autocast both sides to bfloat16")
    parser.add_argument("--save-npz", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from memfof import MEMFOF  # upstream package on PYTHONPATH
    from libreyolo import LibreYOLO

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    paths = sorted(Path(args.frames).glob("*.png"))
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    print(f"{len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]}")

    def autocast():
        if args.bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    upstream = MEMFOF.from_pretrained(args.upstream_dir).eval().to(device)
    up_flows = {}
    with torch.inference_mode(), autocast():
        for t in range(1, len(frames) - 1):
            window = torch.stack(
                [torch.tensor(frames[i], dtype=torch.float32).permute(2, 0, 1) for i in (t - 1, t, t + 1)]
            )[None].to(device)
            fw = upstream(window)["flow"][-1][:, 1]
            up_flows[t] = fw[0].float().permute(1, 2, 0).cpu().numpy()
    del upstream
    torch.cuda.empty_cache()

    model = LibreYOLO(args.ckpt, device="cuda")
    with torch.inference_mode(), autocast():
        results = model([Image.fromarray(f) for f in frames])
    assert len(results) == len(frames) - 1

    rows = []
    for t, up in up_flows.items():
        ly = results[t].flow.data
        ly = ly.cpu().numpy() if isinstance(ly, torch.Tensor) else np.asarray(ly)
        diff = np.abs(ly - up)
        epe = np.linalg.norm(ly - up, axis=-1)
        rows.append((t, float(diff.max()), float(diff.mean()), float(epe.mean()),
                     float(np.linalg.norm(up, axis=-1).mean())))
    worst = max(r[1] for r in rows)
    summary = {
        "precision": "bf16 autocast" if args.bf16 else "fp32 (TF32 off)",
        "frames_compared": len(rows),
        "max_abs_diff_px": worst,
        "mean_abs_diff_px": float(np.mean([r[2] for r in rows])),
        "mean_epe_px": float(np.mean([r[3] for r in rows])),
        "mean_flow_magnitude_px": float(np.mean([r[4] for r in rows])),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    for t, mx, mn, epe, mag in rows:
        print(f"frame {t + 1:03d}: max {mx:.3e} mean {mn:.3e} epe {epe:.3e} |flow| {mag:.2f}")
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    if args.save_npz:
        np.savez_compressed(args.save_npz, **{f"t{t:03d}": v for t, v in up_flows.items()})


if __name__ == "__main__":
    main()
