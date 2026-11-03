"""Parity check: ptlflow DPFlow vs LibreYOLO DPFlow on consecutive frames.

Run in two environments (ptlflow is not a LibreYOLO dependency):

    # 1. ptlflow venv: upstream flows for every consecutive pair
    python weights/parity_dpflow.py upstream --frames DIR --ckpt dpflow-things.ckpt --out up.npz
    # 2. LibreYOLO venv: LibreYOLO flows and the comparison
    python weights/parity_dpflow.py libreyolo --frames DIR --weights LibreDPFlowm-flow.pt \
        --out lib.npz --compare up.npz

Upstream reads frames with OpenCV (BGR in [0, 1]), as ptlflow's own inference
script does; LibreYOLO goes through ``model.predict`` on the PNG paths. Both
run fp32 on the same device with TF32 disabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _frames(directory: str) -> list[Path]:
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(paths) < 2:
        raise SystemExit(f"need at least two frames in {directory}")
    return paths


def _no_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_upstream(frames: list[Path], ckpt: str, out: str) -> None:
    import cv2
    import ptlflow

    _no_tf32()
    model = ptlflow.get_model("dpflow", ckpt_path=ckpt).cuda().eval()
    flows = []
    with torch.no_grad():
        for a, b in zip(frames[:-1], frames[1:]):
            imgs = [cv2.imread(str(p)).astype(np.float32) / 255.0 for p in (a, b)]
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)[None].cuda()
            pred = model({"images": x})["flows"][0, 0]
            flows.append(pred.permute(1, 2, 0).float().cpu().numpy())
    np.savez(out, flows=np.stack(flows))
    print(f"upstream: {len(flows)} pairs -> {out}")


def run_libreyolo(frames: list[Path], weights: str, out: str, compare: str | None) -> None:
    from libreyolo import LibreYOLO

    _no_tf32()
    model = LibreYOLO(weights, device="cuda")
    results = list(model.predict([str(p) for p in frames]))
    flows = np.stack([np.asarray(r.flow.data, dtype=np.float32) for r in results])
    np.savez(out, flows=flows)
    print(f"libreyolo: {len(flows)} pairs -> {out}")
    if compare:
        ref = np.load(compare)["flows"]
        assert ref.shape == flows.shape, (ref.shape, flows.shape)
        diff = np.abs(flows - ref)
        epe = np.linalg.norm(flows - ref, axis=-1)
        report = {
            "pairs": int(len(flows)),
            "shape": list(flows.shape[1:]),
            "max_abs_diff_px": float(diff.max()),
            "mean_abs_diff_px": float(diff.mean()),
            "mean_epe_px": float(epe.mean()),
            "per_pair_max_abs_diff_px": [float(d.max()) for d in diff],
            "upstream_mean_magnitude_px": float(np.linalg.norm(ref, axis=-1).mean()),
        }
        print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("upstream", "libreyolo"))
    parser.add_argument("--frames", required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--weights")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    frames = _frames(args.frames)
    if args.mode == "upstream":
        run_upstream(frames, args.ckpt, args.out)
    else:
        run_libreyolo(frames, args.weights, args.out, args.compare)


if __name__ == "__main__":
    main()
