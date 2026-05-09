"""Convert lyuwenyu RT-DETRv2 ResNet PyTorch weights to LibreYOLO format.

Source weights (Apache-2.0, lyuwenyu/RT-DETR upstream).

Adaptation steps (same as the existing HGNetv2 converter):
  1. Unwrap the EMA wrapper: ckpt["ema"]["module"] -> raw state_dict.
  2. Remap encoder/decoder input_proj and decoder.enc_output keys from
     v2's named-submodule style (.conv./.norm./.proj.) to LibreYOLO's
     Sequential numeric style (.0./.1.).
  3. Drop v2-only tensors LibreYOLO's v1-style RT-DETR module does not have:
       - decoder.anchors, decoder.valid_mask    (precomputed eval buffers)
       - cross_attn.num_points_scale            (v2 discrete-sampling scale)
  4. Wrap with model_family="rtdetrv2" metadata so the factory routes to
     LibreRTDETRv2 instead of LibreRTDETR.
  5. Save as weights/LibreRTDETRv2{r18,r34,r50,r50m,r101}.pt

Usage::

    python weights/convert_rtdetrv2_weights.py downloads/v2_ckpts/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \\
        weights/LibreRTDETRv2r18.pt --size r18
"""

from __future__ import annotations

import argparse

from _conversion_utils import (
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


_DROP_FRAGMENTS: tuple[str, ...] = ()
# v2 registers ``decoder.anchors`` / ``decoder.valid_mask`` and
# ``cross_attn.num_points_scale`` as buffers; keep them all so the strict load
# overrides our init-time values with the upstream-saved tensors. (Initial
# values differ by ~3e-4 due to torch-version/precision drift.)


def _remap_key(k: str) -> str:
    # Only ``encoder.input_proj`` needs remapping: v1's HybridEncoder uses
    # numeric Sequential keys (``.0/.1``) but upstream uses named submodules
    # (``.conv/.norm``). The v2 decoder we ported preserves upstream's named
    # submodules for ``decoder.enc_output`` and ``decoder.input_proj``, so
    # those keys pass through unchanged.
    if k.startswith("encoder.input_proj."):
        parts = k.split(".")
        if len(parts) >= 4:
            sub = parts[3]
            if sub == "conv":
                parts[3] = "0"
                return ".".join(parts)
            if sub == "norm":
                parts[3] = "1"
                return ".".join(parts)
    return k


def convert_weights(input_path: str, output_path: str, size: str, nc: int = 80) -> dict:
    print(f"Loading upstream RT-DETRv2 weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries (EMA-preferred)")

    out = {}
    dropped = []
    for k, v in state_dict.items():
        if any(frag in k for frag in _DROP_FRAGMENTS):
            dropped.append(k)
            continue
        out[_remap_key(k)] = v.float().clone()

    print(f"Stripped {len(dropped)} v2-only tensors:")
    for k in dropped[:6]:
        print(f"  - {k}")
    if len(dropped) > 6:
        print(f"  ... +{len(dropped) - 6} more")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        out, model_family="rtdetrv2", size=size, nc=nc,
    )
    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}  ({len(out)} tensors)")
    return libreyolo_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RT-DETRv2 weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream RT-DETRv2 checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        required=True,
        choices=["r18", "r34", "r50", "r50m", "r101"],
        help="Size code matching the upstream backbone",
    )
    parser.add_argument(
        "--nc", type=int, default=80, help="Number of classes (default: 80)"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
