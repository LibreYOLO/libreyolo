"""Compare batch_size=32 vs batch_size=2+accum=16 on YOLOX-nano.

Both settings produce the same number of optimizer steps per epoch:
  e.g. on mask-wearing (105 images):
  - Run A: 105 / 32 = 4 batches  → 4 steps  (accum=1)
  - Run B: 105 /  2 = 53 batches → 4 steps  (accum=16)

If gradient accumulation is correct the loss and mAP curves should track.

Usage:
    python scripts/compare_grad_accum.py --data ~/datasets/mask-wearing/data.yaml
    python scripts/compare_grad_accum.py --data coco128.yaml --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add repo root so the script works without pip-install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_grad_accum")


def run_one(
    tag: str,
    data: str,
    batch: int,
    grad_accum_steps: int,
    epochs: int,
    imgsz: int,
    device: str,
    project: str,
    eval_interval: int,
    size: str,
    lr0: float,
) -> dict:
    from libreyolo import LibreYOLOX

    logger.info(
        "Starting run '%s'  model=YOLOX-%s  batch=%d  grad_accum_steps=%d  "
        "(effective batch=%d  lr0=%.4f)",
        tag, size, batch, grad_accum_steps, batch * grad_accum_steps, lr0,
    )

    # Pass the weight filename so pretrained COCO weights are downloaded/loaded.
    # LibreYOLOX(size=...) leaves model_path=None and starts from scratch.
    model = LibreYOLOX(f"LibreYOLOX{size}.pt")
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        device=device,
        project=project,
        name=tag,
        exist_ok=True,
        amp=False,          # keep AMP off for reproducibility
        ema=False,          # EMA masks raw loss differences; off for comparison
        eval_interval=eval_interval,
        workers=2,
        grad_accum_steps=grad_accum_steps,
        seed=42,
    )
    logger.info("Run '%s' done — epoch losses: %s", tag, results["epoch_losses"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="coco128.yaml", help="Dataset config yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--device", default="", help="Device ('' = auto)")
    parser.add_argument("--project", default="runs/grad_accum_compare", help="Output directory")
    parser.add_argument("--eval-interval", type=int, default=5, help="Validate every N epochs (0 = skip)")
    parser.add_argument("--size", default="s", help="YOLOX size: n / s / m / l / x")
    parser.add_argument("--lr0", type=float, default=0.001, help="Base LR (fine-tune default 0.001, scratch ~0.01)")
    args = parser.parse_args()

    common = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        eval_interval=args.eval_interval,
        size=args.size,
        lr0=args.lr0,
    )

    # Run A: large batch, no accumulation
    results_a = run_one(tag="bs32_accum1",  batch=32, grad_accum_steps=1,  **common)
    # Run B: small batch, 16-step accumulation (same effective batch)
    results_b = run_one(tag="bs2_accum16", batch=2,  grad_accum_steps=16, **common)

    W = 72
    print("\n" + "=" * W)
    print("LOSS COMPARISON")
    print("=" * W)
    print(f"{'Epoch':<8}{'bs=32 accum=1':>18}{'bs=2 accum=16':>18}{'|diff|':>12}")
    print("-" * W)

    losses_a = results_a["epoch_losses"]
    losses_b = results_b["epoch_losses"]
    for ep, (la, lb) in enumerate(zip(losses_a, losses_b), 1):
        print(f"{ep:<8}{la:>18.4f}{lb:>18.4f}{abs(la - lb):>12.4f}")

    avg_diff = sum(abs(a - b) for a, b in zip(losses_a, losses_b)) / len(losses_a)
    print("=" * W)
    print(f"Mean |loss diff|: {avg_diff:.4f}")
    print()

    mAP_a = results_a.get("best_mAP50_95", 0.0)
    mAP_b = results_b.get("best_mAP50_95", 0.0)
    mAP50_a = results_a.get("best_mAP50", 0.0)
    mAP50_b = results_b.get("best_mAP50", 0.0)

    if mAP_a or mAP_b:
        print("METRIC COMPARISON (best checkpoint)")
        print("=" * W)
        print(f"{'Metric':<20}{'bs=32 accum=1':>18}{'bs=2 accum=16':>18}{'|diff|':>12}")
        print("-" * W)
        print(f"{'mAP50':<20}{mAP50_a:>18.4f}{mAP50_b:>18.4f}{abs(mAP50_a - mAP50_b):>12.4f}")
        print(f"{'mAP50-95':<20}{mAP_a:>18.4f}{mAP_b:>18.4f}{abs(mAP_a - mAP_b):>12.4f}")
        print(f"{'best epoch':<20}{results_a.get('best_epoch', '-'):>18}{results_b.get('best_epoch', '-'):>18}")
        print("=" * W)

    verdict = "OK" if avg_diff < 0.5 else "WARNING"
    symbol = "✓" if avg_diff < 0.5 else "✗"
    print(f"\n{symbol} {verdict} — mean loss diff {avg_diff:.4f} "
          f"({'within' if avg_diff < 0.5 else 'above'} tolerance 0.5). "
          "Gradient accumulation looks correct." if avg_diff < 0.5 else
          f"\n{symbol} {verdict} — mean loss diff {avg_diff:.4f} above tolerance 0.5. Investigate.")

    # Save raw results
    out = Path(args.project) / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "bs32_accum1": {k: v for k, v in results_a.items() if isinstance(v, (int, float, str, list))},
                "bs2_accum16": {k: v for k, v in results_b.items() if isinstance(v, (int, float, str, list))},
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
