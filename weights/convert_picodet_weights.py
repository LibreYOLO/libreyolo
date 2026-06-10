"""Convert Bo396543018/Picodet_Pytorch checkpoints to LibreYOLO format.

Per-size repos: ``LibrePICODETs``, ``LibrePICODETm``, ``LibrePICODETl``.

Bo's checkpoints carry mmdet-style key naming because his ``ESNet`` /
``CSPPAN`` / ``PICODETHead`` are wrapped in mmcv's ``ConvModule`` /
``DepthwiseSeparableConvModule`` / ``SELayer`` and registered as a
detector via ``@DETECTORS``. LibreYOLO's port keeps the same numerics
but flattens those wrappers, so the key remap is purely syntactic:

  bbox_head.*                           -> head.*
  backbone.<stage>_<i>.*                -> backbone.blocks.<flat_idx>.*
  neck.trans.trans.<i>.*                -> neck.trans.<i>.*
  *.se.conv{1,2}.conv.{w,b}             -> *.se.conv{1,2}.{w,b}

Usage::

    python weights/convert_picodet_weights.py \
        --src ~/picodet_s_320_coco-some-epoch.pth \
        --size s --nc 80 \
        --dst weights/LibrePICODETs.pt
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

# The key remapping lives in ``libreyolo.models.picodet.convert`` (shared with
# the runtime auto-converter); this script is the offline CLI wrapper.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Path to Bo's .pth checkpoint")
    parser.add_argument("--dst", required=True, help="Output LibreYOLO checkpoint path")
    parser.add_argument("--size", required=True, choices=["s", "m", "l"])
    parser.add_argument("--nc", type=int, default=80, help="Number of classes")
    args = parser.parse_args()

    add_repo_root_to_path()
    from libreyolo.models.picodet.convert import convert_upstream
    from libreyolo.models.picodet.nn import LibrePICODETModel

    print(f"Loading {args.src}")
    raw = load_checkpoint(args.src)
    sd = extract_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"Could not extract state dict from {args.src}")

    # Keeps the regular (non-EMA) weights — that is what Bo's mmdet
    # ``init_detector`` actually loads — and drops the integral.project
    # buffer (LibreYOLO computes DFL inline in PicoHead).
    print(f"Converting {len(sd)} keys")
    sd = convert_upstream(sd)

    # Sanity-load into a fresh LibreYOLO model and report missing/unexpected.
    target = LibrePICODETModel(size=args.size, nb_classes=args.nc)
    missing, unexpected = target.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys (in target, not in source): {len(missing)}")
        for k in missing[:10]:
            print(f"  + {k}")
    if unexpected:
        print(f"Unexpected keys (in source, not in target): {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    if not missing and not unexpected:
        print("All keys matched cleanly.")

    wrapped = wrap_libreyolo_checkpoint(
        sd, model_family="picodet", size=args.size, nc=args.nc,
    )
    out = save_checkpoint(wrapped, args.dst)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
