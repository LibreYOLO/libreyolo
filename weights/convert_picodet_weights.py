"""Convert Bo396543018/Picodet_Pytorch checkpoints to LibreYOLO format.

Bo's checkpoints carry mmdet-style key naming because his ``ESNet`` /
``CSPPAN`` / ``PicoDetHead`` are wrapped in mmcv's ``ConvModule`` /
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
        --dst weights/LibrePicoDets.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


# ESNet stage repeats: stage_id (2,3,4) -> repeats. Used to flatten Bo's
# ``<stage>_<i>`` (1-indexed) names into ``blocks.<flat_idx>`` (0-indexed).
ESNET_STAGE_REPEATS = (3, 7, 3)


def _build_block_index_map() -> Dict[str, int]:
    """Map ``<stage>_<i>`` -> flat block index. Bo numbers stages 2,3,4."""
    out: Dict[str, int] = {}
    flat = 0
    for stage_idx, repeats in enumerate(ESNET_STAGE_REPEATS):
        stage_id = stage_idx + 2
        for i in range(repeats):
            out[f"{stage_id}_{i + 1}"] = flat
            flat += 1
    return out


_BLOCK_MAP = _build_block_index_map()
# Pattern matches Bo's per-block prefix: e.g. ``backbone.2_1.`` or ``backbone.4_3.``
_BACKBONE_BLOCK_RE = re.compile(r"^backbone\.(\d+_\d+)\.")
# SE wraps with ConvModule, adding an extra ``.conv.`` we need to drop.
_SE_CONV_RE = re.compile(r"\.se\.conv([12])\.conv\.")


def remap_key(key: str) -> str | None:
    """Translate a single Bo-style key to LibreYOLO naming.

    Returns ``None`` if the key should be dropped (e.g. a buffer LibreYOLO
    doesn't carry). Currently no keys are dropped — all numerics survive.
    """
    new = key

    # Top-level rename: bbox_head -> head
    if new.startswith("bbox_head."):
        new = "head." + new[len("bbox_head.") :]

    # Backbone block flattening: backbone.<stage>_<i>. -> backbone.blocks.<flat>.
    m = _BACKBONE_BLOCK_RE.match(new)
    if m is not None:
        token = m.group(1)
        flat = _BLOCK_MAP.get(token)
        if flat is None:
            raise ValueError(
                f"Unexpected backbone block token {token!r} in key {key!r}; "
                "expected one of " + ", ".join(sorted(_BLOCK_MAP))
            )
        new = f"backbone.blocks.{flat}." + new[m.end() :]

    # Neck transformation: neck.trans.trans.X.* -> neck.trans.X.*
    if new.startswith("neck.trans.trans."):
        new = "neck.trans." + new[len("neck.trans.trans.") :]

    # SE ConvModule unwrap: *.se.conv1.conv.X -> *.se.conv1.X (and conv2)
    new = _SE_CONV_RE.sub(lambda mm: f".se.conv{mm.group(1)}.", new)

    return new


def remap_state_dict(state_dict: dict) -> dict:
    """Remap an entire Bo-format state dict. Detects collisions early."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new = remap_key(k)
        if new is None:
            continue
        if new in out:
            raise ValueError(f"Key collision after remap: {k!r} and another -> {new!r}")
        out[new] = v
    return out


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
    from libreyolo.models.picodet.nn import LibrePicoDetModel

    print(f"Loading {args.src}")
    raw = load_checkpoint(args.src)
    sd = extract_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"Could not extract state dict from {args.src}")

    print(f"Remapping {len(sd)} keys")
    sd = remap_state_dict(sd)

    # Sanity-load into a fresh LibreYOLO model and report missing/unexpected.
    target = LibrePicoDetModel(size=args.size, nb_classes=args.nc)
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
