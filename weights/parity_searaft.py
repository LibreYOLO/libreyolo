"""SEA-RAFT parity: upstream princeton-vl/SEA-RAFT vs LibreYOLO on a frame sequence.

Runs upstream ``RAFT(args)`` (``config/eval/sintel-{S,M,L}.json``, scale 0, the
native-resolution setting) loaded the way upstream ``custom.py`` does, and
LibreYOLO through the public API (``LibreYOLO(pt)(frames)``), in fp32 with TF32
disabled, and reports the max / mean absolute difference of the flow in pixels,
the mean end-point error between the two fields and the confidence difference.

    python weights/parity_searaft.py --upstream /workspace/searaft_up \
        --frames /workspace/clip/frames --weights-dir /workspace/out/searaft \
        --upstream-ckpt-dir /workspace/searaft_ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def _stub_hub_mixin():
    # Upstream raft.py imports PyTorchModelHubMixin only to offer
    # from_pretrained; it does not affect the forward pass.
    if "huggingface_hub" in sys.modules:
        return
    try:
        import huggingface_hub  # noqa: F401
        return
    except ImportError:
        pass
    mod = types.ModuleType("huggingface_hub")

    class PyTorchModelHubMixin:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

    mod.PyTorchModelHubMixin = PyTorchModelHubMixin
    sys.modules["huggingface_hub"] = mod


def upstream_model(upstream: Path, size: str, ckpt: Path, device):
    _stub_hub_mixin()
    sys.path.insert(0, str(upstream / "core"))
    sys.path.insert(0, str(upstream))
    from config.parser import json_to_args
    from raft import RAFT
    from safetensors.torch import load_file

    args = json_to_args(str(upstream / f"config/eval/sintel-{size.upper()}.json"))
    model = RAFT(args)
    # upstream load_ckpt / hub mixin: strict=False
    missing, unexpected = model.load_state_dict(load_file(str(ckpt)), strict=False)
    model.to(device).eval()
    return model, args, missing, unexpected


def upstream_confidence(info, args):
    # upstream custom.py::get_heatmap, normalised the same way as LibreYOLO
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1)
    return (1.0 - heatmap / args.var_max).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--upstream-ckpt-dir", type=Path, required=True)
    ap.add_argument("--sizes", default="s,m,l")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from libreyolo import LibreYOLO

    frames = sorted(a.frames.glob("*.png")) or sorted(a.frames.glob("*.jpg"))
    rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frames]
    report = {"frames": len(frames), "pairs": len(frames) - 1,
              "shape_hw": list(rgb[0].shape[:2]), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "torch": torch.__version__, "sizes": {}}

    for size in a.sizes.split(","):
        ck = a.upstream_ckpt_dir / ("S" if size == "s" else "M") / "model.safetensors"
        up, args, missing, unexpected = upstream_model(a.upstream, size, ck, device)
        ours = LibreYOLO(str(a.weights_dir / f"LibreSEARAFT{size}-flow.pt"), device=str(device))
        results = ours([p.as_posix() for p in frames])
        assert len(results) == len(frames) - 1, len(results)

        max_abs, sum_abs, n_el, epe_sum, n_px = 0.0, 0.0, 0, 0.0, 0
        conf_max, mags = 0.0, []
        for i in range(len(frames) - 1):
            t1 = torch.tensor(rgb[i], dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            t2 = torch.tensor(rgb[i + 1], dtype=torch.float32).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                out = up(t1, t2, iters=args.iters, test_mode=True)
            f_up = out["flow"][-1][0].permute(1, 2, 0).cpu()
            c_up = upstream_confidence(out["info"][-1], args)[0].cpu()
            f_ours = torch.as_tensor(results[i].flow.data).cpu()
            c_ours = torch.as_tensor(results[i].flow.confidence).cpu()
            d = (f_ours - f_up).abs()
            max_abs = max(max_abs, float(d.max()))
            sum_abs += float(d.sum()); n_el += d.numel()
            epe = (f_ours - f_up).norm(dim=-1)
            epe_sum += float(epe.sum()); n_px += epe.numel()
            conf_max = max(conf_max, float((c_ours - c_up).abs().max()))
            mags.append(float(f_up.norm(dim=-1).mean()))
        report["sizes"][size] = {
            "iters": args.iters,
            "upstream_missing_keys": len(missing), "upstream_unexpected_keys": len(unexpected),
            "max_abs_diff_px": max_abs, "mean_abs_diff_px": sum_abs / n_el,
            "mean_epe_between_px": epe_sum / n_px, "confidence_max_abs_diff": conf_max,
            "mean_flow_magnitude_px": float(np.mean(mags)),
        }
        print(size, json.dumps(report["sizes"][size]))
        del up, ours
        torch.cuda.empty_cache()
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
