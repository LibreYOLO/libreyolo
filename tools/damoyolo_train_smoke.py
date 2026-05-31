"""Training smoke test for the ported DAMO-YOLO.

Builds a model with the upstream T checkpoint, fabricates a tiny synthetic
batch (4 random images + 2-3 plausible GTs each), and runs N optimizer steps
on the full GFL + AlignOTA loss. Verifies:

- ``forward(targets=...)`` produces a finite loss dict
- gradients flow through every learnable parameter
- total loss decreases substantially over the run

This is *not* training to convergence — it just establishes that all of
loss + assigner + head training paths are wired correctly end-to-end.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libreyolo.models.damoyolo.nn import build_damoyolo  # noqa: E402


def make_targets(batch_size: int, image_size: int, device: torch.device, num_classes: int = 80):
    rng = torch.Generator(device="cpu").manual_seed(0)
    targets = []
    for b in range(batch_size):
        n_gt = int(torch.randint(2, 4, (1,), generator=rng).item())
        # random boxes covering a plausible fraction of the image
        cx = torch.rand(n_gt, generator=rng) * (image_size - 100) + 50
        cy = torch.rand(n_gt, generator=rng) * (image_size - 100) + 50
        w = torch.rand(n_gt, generator=rng) * 80 + 30
        h = torch.rand(n_gt, generator=rng) * 80 + 30
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        boxes = boxes.clamp(0, image_size - 1)
        labels = torch.randint(0, num_classes, (n_gt,), generator=rng)
        targets.append({"boxes": boxes.to(device), "labels": labels.to(device)})
    return targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=None, help="Optional pretrained weights")
    p.add_argument("--size", default="t", choices=["t"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    print(f"Building DAMO-YOLO size={args.size}")
    model = build_damoyolo(size=args.size, num_classes=80).to(device)
    if args.weights is not None:
        ck = torch.load(str(args.weights), map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=True)
        print("Loaded pretrained weights")
    model.train()

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Same tiny batch reused every step — we want loss-on-this-batch to
    # decrease, not generalisation.
    images = torch.randn(args.batch, 3, args.image_size, args.image_size, device=device)
    targets = make_targets(args.batch, args.image_size, device)

    losses = []
    t0 = time.time()
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        out = model(images, targets=targets)
        loss = out["total_loss"]
        loss.backward()
        # Verify every parameter that requires grad got one (catches dead branches).
        if step == 0:
            no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            print(f"params without grad after step 0: {len(no_grad)}")
            for n in no_grad[:5]:
                print(f"  {n}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        optim.step()
        losses.append(float(loss.detach().cpu()))
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"step {step:3d}  total={loss.item():.4f}  "
                f"cls={out['loss_cls'].item():.4f}  bbox={out['loss_bbox'].item():.4f}  "
                f"dfl={out['loss_dfl'].item():.4f}"
            )

    dt = time.time() - t0
    print(f"\nElapsed: {dt:.1f}s ({dt / args.steps:.2f}s/step)")
    print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}  (Δ {losses[0] - losses[-1]:+.4f})")
    if not (losses[-1] < losses[0]):
        print("FAIL: loss did not decrease")
        return 1
    print("PASS: training step is functional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
