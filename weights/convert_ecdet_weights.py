"""Convert upstream EdgeCrafter ECDet COCO weights into LibreYOLO format.

Upstream releases ship as ``{"model": state_dict}``. LibreYOLO checkpoints add
metadata (``model_family``, ``nc``, ``size``, ``names``) so the unified
``LibreYOLO()`` factory can route without filename heuristics.

ECDet's module names already match the LibreECDET port byte-for-byte, so this
is a metadata wrap — no key remapping required.

Usage:
    python weights/convert_ecdet_weights.py downloads/ec_weights/ecdet_s.pth weights/LibreECDETs.pt --size s
    python weights/convert_ecdet_weights.py downloads/ec_weights/ecdet_m.pth weights/LibreECDETm.pt --size m
    python weights/convert_ecdet_weights.py downloads/ec_weights/ecdet_l.pth weights/LibreECDETl.pt --size l
    python weights/convert_ecdet_weights.py downloads/ec_weights/ecdet_x.pth weights/LibreECDETx.pt --size x

Add ``--verify`` to load the converted weights into a LibreECDET wrapper and
run a smoke forward pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _unwrap(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    ema = checkpoint.get("ema")
    if isinstance(ema, dict):
        module = ema.get("module")
        if isinstance(module, dict):
            return module
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def convert_weights(input_path: str, output_path: str, size: str, nc: int = 80) -> dict:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from libreyolo.utils.general import COCO_CLASSES

    print(f"Loading upstream weights from {input_path}")
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap(raw)
    print(f"Found {len(state_dict)} parameter entries")

    names = (
        {i: n for i, n in enumerate(COCO_CLASSES)}
        if nc == 80
        else {i: f"class_{i}" for i in range(nc)}
    )

    libreyolo_ckpt = {
        "model": state_dict,
        "model_family": "ecdet",
        "size": size,
        "nc": nc,
        "names": names,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str) -> bool:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from libreyolo import LibreECDET

    print(f"\nLoading converted weights into LibreECDET-{size}...")
    m = LibreECDET(converted_path, size=size, device="cpu")
    print(f"  family={m.FAMILY} size={m.size} nc={m.nb_classes}")

    m.model.eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 640, 640))
    assert "pred_logits" in out and "pred_boxes" in out
    assert out["pred_logits"].shape == (1, 300, 80)
    assert out["pred_boxes"].shape == (1, 300, 4)
    print("  forward pass OK — shapes match")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ECDet weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream ECDet checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=["s", "m", "l", "x"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size)
