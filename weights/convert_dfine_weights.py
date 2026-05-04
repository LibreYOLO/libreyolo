"""Convert upstream D-FINE COCO weights into LibreYOLO format.

Upstream releases ship as ``{"model": state_dict}`` (or ``{"ema": {"module": state_dict}, ...}``).
LibreYOLO checkpoints add metadata (``model_family``, ``nc``, ``size``, ``names``)
so the unified ``LibreYOLO()`` factory can route correctly without filename heuristics.

D-FINE's module names already match LibreYOLO's, so this is a metadata wrap —
no key remapping required.

Usage:
    python weights/convert_dfine_weights.py weights/dfine_n_coco.pth weights/LibreDFINEn.pt --size n
    python weights/convert_dfine_weights.py weights/dfine_s_coco.pth weights/LibreDFINEs.pt --size s
    python weights/convert_dfine_weights.py weights/dfine_m_coco.pth weights/LibreDFINEm.pt --size m
    python weights/convert_dfine_weights.py weights/dfine_l_coco.pth weights/LibreDFINEl.pt --size l
    python weights/convert_dfine_weights.py weights/dfine_x_coco.pth weights/LibreDFINEx.pt --size x

Add ``--verify`` to load the converted weights into a LibreDFINE wrapper and
run a smoke inference, confirming round-trip integrity.
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
    print(f"Found {len(state_dict)} parameter entries")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="dfine",
        size=size,
        nc=nc,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str) -> bool:
    """Load via LibreDFINE wrapper and run a smoke forward pass."""
    add_repo_root_to_path()
    from libreyolo import LibreDFINE

    print(f"\nLoading converted weights into LibreDFINE-{size}...")
    m = LibreDFINE(converted_path, size=size, device="cpu")
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
    parser = argparse.ArgumentParser(
        description="Convert D-FINE weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream D-FINE checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        required=True,
        choices=["n", "s", "m", "l", "x"],
        help="Size code",
    )
    parser.add_argument(
        "--nc", type=int, default=80, help="Number of classes (default: 80)"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size)
