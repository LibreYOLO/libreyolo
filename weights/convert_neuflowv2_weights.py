"""Convert official NeuFlow v2 weights into LibreYOLO format.

Upstream (https://github.com/neufieldrobotics/NeuFlow_v2, Apache-2.0) commits
three checkpoints in its repository root, each ``{"model": state_dict, ...}``
with the same architecture:

- ``neuflow_mixed.pth``  -> ``LibreNeuFlowv2s-flow.pt`` (canonical)
- ``neuflow_sintel.pth`` -> ``LibreNeuFlowv2s-flow-sintel.pt``
- ``neuflow_things.pth`` -> ``LibreNeuFlowv2s-flow-things.pt``

The model code is vendored unchanged in its parameter names under
``libreyolo/models/neuflowv2/_vendor``, so the script only wraps the state
dict in the LibreYOLO v1.0 checkpoint schema (0 missing / 0 unexpected keys).

Usage::

    python weights/convert_neuflowv2_weights.py neuflow_mixed.pth \
        weights/LibreNeuFlowv2s-flow.pt

The variant is read from the upstream filename (``mixed`` / ``sintel`` /
``things``); pass ``--variant`` to override. ``--verify`` loads the result
through ``LibreYOLO(...)`` and runs a forward pass.
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
)

VARIANTS = ("mixed", "sintel", "things")
CANONICAL_VARIANT = "mixed"
_IMGSZ_HW = (432, 768)


def output_filename(variant: str) -> str:
    """Canonical LibreYOLO filename for an upstream variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown NeuFlow v2 variant {variant!r}; expected {VARIANTS}.")
    suffix = "" if variant == CANONICAL_VARIANT else f"-{variant}"
    return f"LibreNeuFlowv2s-flow{suffix}.pt"


def detect_variant(input_path: str) -> str | None:
    name = Path(input_path).name.lower()
    for variant in VARIANTS:
        if variant in name:
            return variant
    return None


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    variant: str | None = None,
) -> dict:
    variant = variant or detect_variant(input_path)
    if variant not in VARIANTS:
        raise ValueError(
            f"Could not tell the NeuFlow v2 variant from {input_path!r}; pass "
            f"--variant {{{','.join(VARIANTS)}}}."
        )

    add_repo_root_to_path()
    from libreyolo.models.neuflowv2.model import LibreNeuFlowv2
    from libreyolo.models.neuflowv2.nn import LibreNeuFlowv2Net
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    print(f"Loading upstream NeuFlow v2 weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path))
    if not LibreNeuFlowv2.can_load(state_dict):
        raise ValueError(
            "This does not look like a NeuFlow v2 checkpoint (expected "
            "backbone.block_8_1.*, cross_attn_s16.layers.*, refine_s8.* and "
            "upsample_s8.* keys)."
        )
    state_dict = {k: v.detach().clone() for k, v in state_dict.items()}

    # Strict load into the native module proves the keys and shapes match.
    LibreNeuFlowv2Net().load_state_dict(state_dict, strict=True)

    imgsz_h, imgsz_w = _IMGSZ_HW
    ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="neuflowv2",
        size="s",
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=max(imgsz_h, imgsz_w),
        imgsz_h=imgsz_h,
        imgsz_w=imgsz_w,
        variant=variant,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    save_checkpoint(ckpt, str(tmp))
    tmp.replace(out)
    print(f"Saved {out} (variant={variant}, {len(state_dict)} tensors)")
    return ckpt


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    m = LibreYOLO(converted_path, device="cpu")
    print(f"  family={m.FAMILY} size={m.size} task={m.task} names={m.names}")
    with torch.no_grad():
        out = m._forward_flow(torch.zeros(2, 3, *_IMGSZ_HW))
    print(f"  forward OK, flow shape {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NeuFlow v2 weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream checkpoint (neuflow_{mixed,sintel,things}.pth)")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .pt (default: weights/<canonical name for the variant>)",
    )
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    variant = args.variant or detect_variant(args.input)
    output = args.output or str(Path("weights") / output_filename(variant or ""))
    convert_weights(args.input, output, variant=variant)
    if args.verify:
        verify_conversion(output)
