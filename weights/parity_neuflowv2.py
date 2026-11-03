"""Parity proof for NeuFlow v2 against the pinned Apache-2.0 upstream.

Requires a checkout of ``neufieldrobotics/NeuFlow_v2`` at commit
``204b5e3744461d90303b9ff82caa7a1bb56a2ca2`` and a CUDA device. For every
consecutive pair of the given frames it feeds identical tensors (LibreYOLO's
preprocess: RGB, stretch-resized to 432x768, scaled to the [0, 255] range
upstream expects) to the upstream ``NeuFlow`` module and to the LibreYOLO
checkpoint, both in fp32 with TF32 off and BatchNorm unfused, and reports:

- canvas: the raw ``(2, H, W)`` network output, upstream vs LibreYOLO;
- e2e: ``LibreYOLO(ckpt).predict(frames)`` ``Results.flow`` on the original
  canvas vs the upstream output resized and rescaled the same way.

Each line gives max and mean absolute difference in pixels and the mean
end-point error (L2 norm of the vector difference) between the two fields.

Usage::

    python weights/parity_neuflowv2.py weights/LibreNeuFlowv2s-flow.pt \
        --upstream-dir /path/to/NeuFlow_v2 --upstream-ckpt neuflow_mixed.pth \
        --frames '/path/to/frames/*.png'
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path

UPSTREAM_COMMIT = "204b5e3744461d90303b9ff82caa7a1bb56a2ca2"


def _stats(a: torch.Tensor, b: torch.Tensor, channel_dim: int) -> dict:
    diff = (a.double() - b.double()).abs()
    epe = (a.double() - b.double()).pow(2).sum(dim=channel_dim).sqrt()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "epe": float(epe.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--upstream-ckpt", required=True)
    parser.add_argument("--frames", required=True, help="glob of frames, sorted by name")
    parser.add_argument("--json", default=None, help="write the per-pair numbers here")
    args = parser.parse_args()

    upstream_dir = Path(args.upstream_dir).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        print(f"upstream at {commit}, expected {UPSTREAM_COMMIT}", file=sys.stderr)
        return 2

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    sys.path.insert(0, str(upstream_dir))
    from NeuFlow.neuflow import NeuFlow  # noqa: E402

    upstream = NeuFlow().to(device)
    ckpt = torch.load(args.upstream_ckpt, map_location="cpu", weights_only=False)
    upstream.load_state_dict(ckpt["model"], strict=True)
    upstream.eval()

    model = LibreYOLO(args.checkpoint, device="cuda")
    model.model.eval()

    frames = sorted(glob.glob(args.frames))
    if len(frames) < 2:
        print(f"need at least two frames, got {len(frames)}", file=sys.stderr)
        return 2

    tensors = [model._preprocess(f)[0] for f in frames]
    _, _, orig_size, _ = model._preprocess(frames[0])
    h, w = tensors[0].shape[-2:]
    upstream.init_bhwd(1, h, w, "cuda", amp=False)

    results = model.predict(frames)
    if len(results) != len(frames) - 1:
        print(f"predict returned {len(results)} results for {len(frames)} frames", file=sys.stderr)
        return 1

    pairs = []
    with torch.no_grad():
        for i in range(len(frames) - 1):
            a = tensors[i].to(device)
            b = tensors[i + 1].to(device)
            up = upstream(a.clone() * 255.0, b.clone() * 255.0)[-1]
            ours = model._forward_flow(torch.cat([a, b], dim=0))
            up_orig = model._postprocess(up, 0.0, 0.0, orig_size)["flow"]
            e2e = results[i].flow.data
            e2e = e2e if isinstance(e2e, torch.Tensor) else torch.as_tensor(e2e)
            pairs.append(
                {
                    "pair": [Path(frames[i]).name, Path(frames[i + 1]).name],
                    "canvas": _stats(up[0].cpu(), ours[0].cpu(), channel_dim=0),
                    "e2e": _stats(up_orig, e2e.cpu(), channel_dim=-1),
                    "upstream_mean_magnitude": float(up[0].norm(dim=0).mean()),
                }
            )

    def agg(key):
        return {
            "max_abs": max(p[key]["max_abs"] for p in pairs),
            "mean_abs": sum(p[key]["mean_abs"] for p in pairs) / len(pairs),
            "epe": sum(p[key]["epe"] for p in pairs) / len(pairs),
        }

    summary = {
        "checkpoint": Path(args.checkpoint).name,
        "upstream_ckpt": Path(args.upstream_ckpt).name,
        "upstream_commit": commit,
        "pairs": len(pairs),
        "canvas_hw": [int(h), int(w)],
        "orig_wh": list(orig_size),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "canvas": agg("canvas"),
        "e2e": agg("e2e"),
        "upstream_mean_magnitude": sum(p["upstream_mean_magnitude"] for p in pairs) / len(pairs),
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=2))
    return 0 if summary["canvas"]["max_abs"] < 1e-2 and summary["e2e"]["max_abs"] < 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
