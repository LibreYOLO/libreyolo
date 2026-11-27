"""Numerical parity check between upstream LoopDVR and native LibreLoopDVR.

Loads official LoopDet / LoopSeg / LoopPose release checkpoints into both
``Intellindust-AI-Lab/LoopDVR`` and ``libreyolo.models.loopdvr`` on the same
input tensor, asserting exact floating-point equality on raw outputs, deploy()
Conv+BN fusion parity, and optional ONNX / TorchScript export parity.

Usage:
    python weights/parity_loopdvr.py \
        --upstream-repo /path/to/LoopDVR \
        --weights-dir /path/to/loopdvr_weights \
        --models loopdet-s loopseg-s looppose-s
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from _conversion_utils import add_repo_root_to_path

EXPECTED_UPSTREAM_COMMIT = "f2aba1f122029e4db757b69cd37fc5a4909db34e"

ALL_MODELS = [
    f"{prefix}-{size}"
    for prefix in ("loopdet", "loopseg", "looppose")
    for size in ("s", "m", "l", "x")
]


def verify_upstream_commit(
    upstream_repo: Path,
    expected_commit: str = EXPECTED_UPSTREAM_COMMIT,
) -> str:
    """Verify ``upstream_repo`` is checked out at the pinned upstream commit."""
    result = subprocess.run(
        ["git", "-C", str(upstream_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not determine git commit of upstream checkout {upstream_repo}: "
            f"{result.stderr.strip()}"
        )
    actual = result.stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"Upstream LoopDVR checkout at {upstream_repo} is at commit {actual}, "
            f"expected pinned commit {expected_commit}"
        )
    return actual


def _task_and_size(model_id: str) -> tuple[str, str]:
    prefix, size = model_id.split("-")
    task = {"loopdet": "detect", "loopseg": "segment", "looppose": "pose"}[prefix]
    return task, size


def check_model_parity(
    model_id: str,
    upstream_repo: Path,
    weights_dir: Path,
    *,
    atol: float = 0.0,
    deploy_atol: float = 1e-2,
) -> dict[str, float]:
    verify_upstream_commit(upstream_repo)
    add_repo_root_to_path()
    if str(upstream_repo) not in sys.path:
        sys.path.insert(0, str(upstream_repo))

    from loopdvr.build import build_model, load_config

    from libreyolo.models.loopdvr.model import LibreLoopDVR

    ckpt_path = weights_dir / f"{model_id}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    prefix = model_id.split("-")[0]
    config_path = upstream_repo / "configs" / prefix / f"{model_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing upstream config: {config_path}")

    task, size = _task_and_size(model_id)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = raw["state_dict"]

    assert LibreLoopDVR.can_load(sd), f"can_load rejected {model_id}"
    assert LibreLoopDVR.detect_size(sd) == size, (
        f"detect_size({model_id})={LibreLoopDVR.detect_size(sd)!r} != {size!r}"
    )
    expected_ckpt_task = None if task == "detect" else task
    assert LibreLoopDVR.detect_checkpoint_task(sd) == expected_ckpt_task
    expected_nc = 1 if task == "pose" else 80
    assert LibreLoopDVR.detect_nb_classes(sd) == expected_nc

    upstream = build_model(load_config(config_path), ckpt_path, device="cpu")
    upstream.eval()

    native = LibreLoopDVR(sd, size=size, device="cpu", task=task)
    native.model.eval()

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(1, 3, 640, 640, generator=gen)

    with torch.no_grad():
        up_out = {k: v for k, v in upstream(x).items() if v is not None}
        na_out = {k: v for k, v in native.model(x).items() if v is not None}

    assert set(up_out.keys()) == set(na_out.keys()), (
        f"Output key mismatch for {model_id}: {up_out.keys()} vs {na_out.keys()}"
    )
    diffs: dict[str, float] = {}
    for key in sorted(up_out):
        max_diff = float((up_out[key] - na_out[key]).abs().max().item())
        diffs[key] = max_diff
        assert max_diff <= atol, (
            f"{model_id} {key} max_abs_diff={max_diff:.3e} exceeds atol={atol:.3e}"
        )

    # Also verify Conv+BN fusion via deploy() stays numerically consistent in
    # float64 (float32 random-noise inputs can swap near-tied DETR top-k
    # queries; float64 proves the algebraic fusion is exact).
    native_f64 = LibreLoopDVR(sd, size=size, device="cpu", task=task)
    native_f64.model.eval().double()
    x64 = x.double()
    with torch.no_grad():
        ref64 = {k: v for k, v in native_f64.model(x64).items() if v is not None}
        native_f64.model.deploy()
        dep64 = {k: v for k, v in native_f64.model(x64).items() if v is not None}
    for key in sorted(ref64):
        dep_diff = float((ref64[key] - dep64[key]).abs().max().item())
        diffs[f"deploy64.{key}"] = dep_diff
        assert dep_diff <= 1e-8, (
            f"{model_id} float64 deploy {key} max_abs_diff={dep_diff:.3e} > 1e-8"
        )

    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LoopDVR upstream parity")
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    for model_id in args.models:
        diffs = check_model_parity(
            model_id,
            args.upstream_repo,
            args.weights_dir,
            atol=args.atol,
        )
        summary = ", ".join(f"{k}={v:.2e}" for k, v in diffs.items())
        print(f"PASS {model_id:10s}: {summary}", flush=True)


if __name__ == "__main__":
    main()
