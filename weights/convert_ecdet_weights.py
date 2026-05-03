"""Convert upstream EdgeCrafter ECDET COCO weights into LibreYOLO format.

Upstream releases ship as ``{"model": state_dict}``. LibreYOLO checkpoints add
metadata (``model_family``, ``nc``, ``size``, ``names``) so the unified
``LibreYOLO()`` factory can route without filename heuristics.

ECDET's module names already match the LibreECDET port byte-for-byte, so this
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

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert_weights(input_path: str, output_path: str, size: str, nc: int = 80) -> dict:
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Could not extract state dict from {input_path}")
    print(f"Found {len(state_dict)} parameter entries")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict, model_family="ecdet", size=size, nc=nc,
    )
    out = save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str) -> bool:
    add_repo_root_to_path()
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
    parser = argparse.ArgumentParser(description="Convert ECDET weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream ECDET checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=["s", "m", "l", "x"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size)
