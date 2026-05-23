#!/usr/bin/env python3
"""
spot_check_val_map.py — Verify that postprocess_detections refactors leave
mAP bit-equivalent on YOLOX and YOLO9.

Runs model.val() on COCO val2017 (configured via VAL_KWARGS["data"]="coco.yaml")
for one or more models, then either:
  - Saves a baseline JSON  (first run, or --save-baseline)
  - Compares against a stored baseline  (subsequent runs)
  - Just prints results    (--no-baseline)

The NMS change is mathematically equivalent, so we expect values to match to
at least 4 decimal places.

Usage:
    # First run — establish baseline:
    python scripts/spot_check_val_map.py --save-baseline baseline_map.json

    # Subsequent runs — compare:
    python scripts/spot_check_val_map.py --baseline baseline_map.json

    # Specific models only:
    python scripts/spot_check_val_map.py --models yolox-n yolo9-t --baseline baseline_map.json

    # Just print (no comparison):
    python scripts/spot_check_val_map.py --no-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo importable when run directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Model registry — smallest models from each affected family.
# Extend with e.g. "yolox-s", "yolo9-s" for a broader check.
# ---------------------------------------------------------------------------
DEFAULT_MODELS = ["yolox-n", "yolo9-t"]

# Map model-id → (weights_filename, size_arg)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "yolox-n": ("LibreYOLOXn.pt", "n"),
    "yolox-t": ("LibreYOLOXt.pt", "t"),
    "yolox-s": ("LibreYOLOXs.pt", "s"),
    "yolox-m": ("LibreYOLOXm.pt", "m"),
    "yolox-l": ("LibreYOLOXl.pt", "l"),
    "yolox-x": ("LibreYOLOXx.pt", "x"),
    "yolo9-t": ("LibreYOLO9t.pt", "t"),
    "yolo9-s": ("LibreYOLO9s.pt", "s"),
    "yolo9-m": ("LibreYOLO9m.pt", "m"),
    "yolo9-c": ("LibreYOLO9c.pt", "c"),
}

VAL_KWARGS = dict(
    data="coco.yaml",
    batch=32,
    conf=0.001,   # COCO-standard low threshold to exercise NMS fully
    iou=0.6,
    verbose=False,
)

# Tolerance for "bit-equivalent": 1e-3 covers floating-point noise from
# reordering ops (e.g. per-class loop → batched_nms) while still catching real divergence.
TOLERANCE = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weights_path(weights_file: str) -> Path:
    """Resolve weights: current dir → weights/ subdir."""
    p = Path(weights_file)
    if p.exists():
        return p
    candidate = Path("weights") / weights_file
    if candidate.exists():
        return candidate
    return p  # let LibreYOLO raise a clear error


def run_val(model_id: str, device: str) -> dict:
    """Load model, run val on coco128, return metric dict."""
    from libreyolo import LibreYOLO

    weights_file, size = MODEL_REGISTRY[model_id]
    weights = _weights_path(weights_file)

    print(f"  Loading {model_id} from {weights} ...")
    model = LibreYOLO(str(weights), size=size, device=device)

    t0 = time.perf_counter()
    results = model.val(**VAL_KWARGS)
    elapsed = time.perf_counter() - t0

    metrics = {
        k.replace("metrics/", ""): round(float(v), 6)
        for k, v in results.items()
        if k.startswith("metrics/")
    }
    metrics["_elapsed_s"] = round(elapsed, 1)
    return metrics


def compare(model_id: str, current: dict, baseline: dict, tol: float) -> bool:
    """Print per-metric diff; return True when all within tolerance."""
    keys = [k for k in current if not k.startswith("_")]
    ok = True
    rows = []
    for k in sorted(keys):
        cur = current.get(k, float("nan"))
        base = baseline.get(k, float("nan"))
        diff = abs(cur - base)
        flag = "" if diff <= tol else "  ← MISMATCH"
        rows.append((k, base, cur, diff, flag))
        if flag:
            ok = False

    col_w = max(len(r[0]) for r in rows)
    header = f"  {'metric':<{col_w}}  {'baseline':>10}  {'current':>10}  {'|diff|':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k, base, cur, diff, flag in rows:
        print(f"  {k:<{col_w}}  {base:>10.6f}  {cur:>10.6f}  {diff:>10.2e}{flag}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY),
        metavar="MODEL",
        help=f"Models to check (default: {' '.join(DEFAULT_MODELS)}). "
             f"Available: {', '.join(MODEL_REGISTRY)}",
    )
    p.add_argument(
        "--baseline",
        metavar="FILE",
        help="JSON baseline to compare against.",
    )
    p.add_argument(
        "--save-baseline",
        metavar="FILE",
        help="Run val and save results as a new baseline JSON (skips comparison).",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Just print results without saving or comparing.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=TOLERANCE,
        help=f"Absolute tolerance for mAP comparison (default: {TOLERANCE}).",
    )
    p.add_argument(
        "--save",
        metavar="FILE",
        help="Save current results to FILE (compatible with all modes, e.g. new.json).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device string passed to LibreYOLO (default: cuda).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Validate mode flags
    modes = sum([
        bool(args.baseline),
        bool(args.save_baseline),
        args.no_baseline,
    ])
    if modes > 1:
        print("ERROR: --baseline, --save-baseline, and --no-baseline are mutually exclusive.", file=sys.stderr)
        return 2
    if modes == 0:
        # Default: compare if baseline exists, else just print
        default_baseline = Path("scripts/baseline.json")
        if default_baseline.exists():
            args.baseline = str(default_baseline)
            print(f"Auto-detected baseline: {default_baseline}")
        else:
            args.no_baseline = True

    baseline_data: dict = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}", file=sys.stderr)
            return 2
        baseline_data = json.loads(baseline_path.read_text())

    # ---------------------------------------------------------------------------
    # Run validation for each requested model
    # ---------------------------------------------------------------------------
    all_results: dict[str, dict] = {}
    failures: list[str] = []

    for model_id in args.models:
        print(f"\n{'─'*60}")
        print(f"  {model_id}")
        print(f"{'─'*60}")
        try:
            metrics = run_val(model_id, args.device)
        except FileNotFoundError as exc:
            print(f"  SKIP — weights not found: {exc}")
            continue
        except Exception as exc:
            print(f"  ERROR — {exc}")
            failures.append(model_id)
            continue

        all_results[model_id] = metrics
        map_val = metrics.get("mAP50-95", float("nan"))
        map50   = metrics.get("mAP50",    float("nan"))
        print(f"  mAP50-95 = {map_val:.4f}  |  mAP50 = {map50:.4f}  ({metrics['_elapsed_s']:.1f}s)")

        if args.baseline and model_id in baseline_data:
            print()
            ok = compare(model_id, metrics, baseline_data[model_id], args.tol)
            if not ok:
                failures.append(model_id)
                print(f"  RESULT: MISMATCH (tol={args.tol})")
            else:
                print(f"  RESULT: OK (all within tol={args.tol})")
        elif args.baseline:
            print(f"  NOTE: {model_id} not in baseline — skipping comparison.")

    # ---------------------------------------------------------------------------
    # Save results if requested
    # ---------------------------------------------------------------------------
    if args.save_baseline and all_results:
        out = Path(args.save_baseline)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nBaseline saved to {out}")

    if args.save and all_results:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nResults saved to {out}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'━'*60}")
    skipped = set(args.models) - set(all_results)
    if skipped:
        print(f"  Skipped (no weights): {', '.join(sorted(skipped))}")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        print("━" * 60)
        return 1
    if all_results:
        print(f"  All {len(all_results)} model(s) passed.")
    else:
        print("  No models were validated (all skipped).")
    print("━" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
