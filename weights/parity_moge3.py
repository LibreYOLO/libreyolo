"""Compare LibreYOLO MoGe-3 against the pinned Microsoft source.

Development-time parity check. It imports the official checkout supplied with
``--upstream-repo`` (which needs FlexGEMM and utils3d installed) and refuses to
run unless that checkout is at the audited commit. Upstream outputs are
computed first and moved to the CPU so only one model is resident at a time.

Reported comparisons, all FP32 on the same input tensor:

- ``forward`` with ``refine_steps=0``: encoder, neck, heads, metric scale;
- ``refine_step``: one refiner call on identical point and feature inputs
  (upstream FlexGEMM kernels against the plain PyTorch sparse ops);
- ``forward`` with the default three refinement steps;
- ``infer``: upstream ``infer()`` against ``LibreMoGe3.predict()``.

Example::

    python weights/parity_moge3.py --upstream-repo ../MoGe \\
        --checkpoint model.pt --image image.jpg --size l
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UPSTREAM_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"


def _verify_upstream_checkout(path: Path) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"MoGe parity must use the audited commit {UPSTREAM_COMMIT}; got {actual}."
        )


def _stats(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    expected = expected.detach().double().cpu()
    actual = actual.detach().double().cpu()
    both = torch.isfinite(expected) & torch.isfinite(actual)
    same_nonfinite = (~torch.isfinite(expected)) == (~torch.isfinite(actual))
    diff = (expected[both] - actual[both]).abs()
    scale = expected[both].abs().clamp_min(1e-12)
    return {
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.numel() else 0.0,
        "max_rel_diff": float((diff / scale).max()) if diff.numel() else 0.0,
        "p99_rel_diff": float(torch.quantile((diff / scale).float()[:16_000_000], 0.99))
        if diff.numel()
        else 0.0,
        "nonfinite_pattern_equal": bool(same_nonfinite.all()),
    }


def _cpu(output: dict) -> dict:
    return {
        key: value.detach().float().cpu()
        for key, value in output.items()
        if isinstance(value, torch.Tensor)
    }


def run_parity(args) -> dict:
    _verify_upstream_checkout(args.upstream_repo)
    sys.path.insert(0, str(args.upstream_repo))

    from libreyolo.models.moge3.model import LibreMoGe3
    from libreyolo.models.moge3.utils import num_tokens_for, preprocess_numpy

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import flex_gemm.config

    flex_gemm.config.SPCONV_ALLOW_TF32 = bool(args.upstream_tf32)
    from moge.model.v3 import MoGeModel

    device = torch.device(args.device)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    chw, _ = preprocess_numpy(rgb, args.imgsz)
    image = torch.from_numpy(chw).unsqueeze(0).to(device)
    tokens = num_tokens_for(args.imgsz)

    captured = {}
    upstream = MoGeModel.from_pretrained(args.checkpoint).eval().to(device)
    with torch.inference_mode():
        up0 = _cpu(upstream(image, num_tokens=tokens, refine_steps=0))
        up3 = _cpu(upstream(image, num_tokens=tokens, refine_steps=3))
        up_infer = _cpu(upstream.infer(image[0], num_tokens=tokens))
    original_refine = type(upstream)._refine_logz
    del upstream
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "size": args.size,
        "image": str(args.image),
        "tokens": tokens,
        "upstream_tf32": bool(args.upstream_tf32),
        "shape": list(image.shape),
    }

    depth = LibreMoGe3(
        str(args.checkpoint), size=args.size, task="depth", device=args.device
    )
    net = depth.model.eval()
    our_refine = type(net)._refine_logz

    def capture(self, coord, feature):
        captured.setdefault("coord", coord.clone())
        captured.setdefault("feature", feature.clone())
        return our_refine(self, coord, feature)

    with torch.inference_mode():
        ours0 = _cpu(net(image, num_tokens=tokens, refine_steps=0))
        type(net)._refine_logz = capture
        try:
            ours3 = _cpu(net(image, num_tokens=tokens, refine_steps=3))
        finally:
            type(net)._refine_logz = our_refine
        ours_step = net._refine_logz(captured["coord"], captured["feature"]).cpu()
    result = depth.predict(str(args.image), imgsz=args.imgsz, verbose=False)
    if isinstance(result, list):
        result = result[0]

    up_points0 = up0["points"][:, -1] if up0["points"].ndim == 5 else up0["points"]
    up_points3 = up3["points"][:, -1] if up3["points"].ndim == 5 else up3["points"]
    report["forward_refine0"] = {
        "points": _stats(up_points0, ours0["points"]),
        "mask": _stats(up0["mask"], ours0["mask"]),
        "metric_scale": _stats(up0["metric_scale"], ours0["metric_scale"]),
    }
    report["forward_refine3"] = {
        "points": _stats(up_points3, ours3["points"]),
        "mask": _stats(up3["mask"], ours3["mask"]),
    }
    dm = result.depth_map
    width = image.shape[-1]
    height = image.shape[-2]
    k = torch.as_tensor(dm.intrinsics).double()
    k_norm = torch.tensor(
        [
            [k[0, 0] / width, 0, (k[0, 2] + 0.5) / width],
            [0, k[1, 1] / height, (k[1, 2] + 0.5) / height],
            [0, 0, 1],
        ],
        dtype=torch.float64,
    )
    up_mask = up_infer["mask"] > 0.5
    our_mask = torch.as_tensor(dm.valid_mask)
    report["infer"] = {
        "depth": _stats(up_infer["depth"], torch.as_tensor(dm.data)),
        "points": _stats(up_infer["points"], torch.as_tensor(dm.point_map)),
        "intrinsics_normalized": _stats(up_infer["intrinsics"], k_norm),
        "mask_disagreement_fraction": float((up_mask != our_mask).float().mean()),
    }
    del depth, net
    gc.collect()
    torch.cuda.empty_cache()

    # Refiner in isolation: the same inputs through upstream's refiner.
    upstream = MoGeModel.from_pretrained(args.checkpoint).eval().to(device)
    with torch.inference_mode():
        up_step = original_refine(
            upstream, captured["coord"].to(device), captured["feature"].to(device)
        ).cpu()
        up_normal = upstream(image, num_tokens=tokens, refine_steps=0)["normal"].cpu()
    report["refine_step_logz"] = _stats(up_step, ours_step)
    del upstream
    gc.collect()
    torch.cuda.empty_cache()

    normal = LibreMoGe3(
        str(args.checkpoint), size=args.size, task="normal", device=args.device
    )
    with torch.inference_mode():
        ours_normal = normal.model(image, num_tokens=tokens).permute(0, 2, 3, 1).cpu()
    report["forward_normal"] = _stats(up_normal, ours_normal)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--size", choices=["l", "g"], required=True)
    parser.add_argument("--imgsz", type=int, default=840)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--upstream-tf32",
        action="store_true",
        help="Leave FlexGEMM's default TF32 sparse GEMMs on.",
    )
    args = parser.parse_args()
    args.upstream_repo = args.upstream_repo.resolve()
    print(json.dumps(run_parity(args), indent=2))


if __name__ == "__main__":
    main()
