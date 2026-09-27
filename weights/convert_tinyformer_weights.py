"""Convert upstream TinyFormer weights into LibreYOLO format.

The released checkpoints (mmpmmpmmpjosh/TinyFormer, Google Drive mirror) are
solver dumps whose EMA weights already use the exact key layout of the native
port, so conversion is a metadata wrap: extract the EMA state dict and stamp
v1.0 LibreYOLO metadata.

Usage:
    python weights/convert_tinyformer_weights.py TinyFormer-S-pbm.pth \
        weights/LibreTinyFormers.pt --size s
    python weights/convert_tinyformer_weights.py TinyFormer-S-pbm-visdrone.pth \
        weights/LibreTinyFormers-visdrone.pt --size s --variant visdrone
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

# Order matches the upstream visdrone2coco.py conversion (category_id 0-9).
VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

_VARIANT_DEFAULTS = {
    None: {"nc": 80, "names": None},
    "obj2coco": {"nc": 80, "names": None},
    "visdrone": {"nc": 10, "names": VISDRONE_NAMES},
}


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    variant: str | None = None,
    nc: int | None = None,
) -> dict:
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    print(f"Found {len(state_dict)} parameter entries")

    defaults = _VARIANT_DEFAULTS[variant]
    effective_nc = nc if nc is not None else defaults["nc"]

    extra = {}
    if variant is not None:
        extra["weight_variant"] = variant

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="tinyformer",
        size=size,
        nc=effective_nc,
        names=defaults["names"] if nc is None else None,
        task="detect",
        supported_tasks=("detect",),
        default_task="detect",
        **extra,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(libreyolo_ckpt, tmp)
    tmp.rename(out)
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreTinyFormer
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    ckpt = load_untrusted_torch_file(converted_path)
    validate_checkpoint_metadata(ckpt)
    print("  metadata schema OK")

    print(f"\nLoading converted weights into LibreTinyFormer-{size}...")
    model = LibreTinyFormer(converted_path, size=size, device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, model.input_size, model.input_size))
    assert "pred_logits" in out and "pred_boxes" in out
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_logits"].shape[-1] == model.nb_classes
    assert out["pred_boxes"].shape[-1] == 4
    print("  forward pass OK")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TinyFormer weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream TinyFormer checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size", required=True, choices=["s", "m", "l", "x", "xl"],
        help="TinyFormer size",
    )
    parser.add_argument(
        "--variant",
        choices=["visdrone", "obj2coco"],
        default=None,
        help="Dataset variant (default: plain COCO weights)",
    )
    parser.add_argument(
        "--nc", type=int, default=None,
        help="Number of classes (default: 80, or 10 for --variant visdrone)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.variant, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size)
