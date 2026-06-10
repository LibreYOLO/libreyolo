"""Convert upstream RT-DETRv4 COCO weights into LibreYOLO format.

Usage::

    python weights/convert_rtdetrv4_weights.py \\
        downloads/v4_ckpts/rtv4_hgnetv2_s_coco.pth \\
        weights/LibreRTDETRv4s.pt --size s
"""

from __future__ import annotations

import argparse

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    nc: int = 80,
) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.rtdetrv4.convert import drop_training_only_keys

    print(f"Loading upstream RT-DETRv4 weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries (EMA-preferred)")

    cleaned, dropped = drop_training_only_keys(state_dict)
    if dropped:
        print(f"Stripped {len(dropped)} training-only keys:")
        for k in dropped:
            print(f"  - {k}")
    else:
        print("No training-only keys to strip (unexpected — verify upstream layout)")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        cleaned,
        model_family="rtdetrv4",
        size=size,
        nc=nc,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RT-DETRv4 weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream RT-DETRv4 checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        required=True,
        choices=["s", "m", "l", "x"],
        help="Size code (RT-DETRv4 ships s/m/l/x; no 'n')",
    )
    parser.add_argument(
        "--nc", type=int, default=80, help="Number of classes (default: 80)"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
