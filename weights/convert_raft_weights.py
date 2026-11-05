"""Convert upstream RAFT checkpoints to the LibreYOLO checkpoint format.

Upstream (https://github.com/princeton-vl/RAFT, BSD-3-Clause) saves
``torch.nn.DataParallel`` state dicts, so every key carries a ``module.``
prefix. ``LibreRAFTNet`` subclasses the vendored ``RAFT`` module, so after
stripping that prefix the tensors load unchanged: conversion is a metadata
wrap with ``task="flow"``, ``nc=1``, ``names={0: "flow"}``.

Usage::

    python weights/convert_raft_weights.py models/raft-things.pth weights/LibreRAFTl-flow.pt
    python weights/convert_raft_weights.py models/raft-small.pth weights/LibreRAFTs-flow.pt
    python weights/convert_raft_weights.py models/raft-sintel.pth weights/LibreRAFTl-flow-sintel.pt --variant sintel
    python weights/convert_raft_weights.py models/raft-kitti.pth weights/LibreRAFTl-flow-kitti.pt --variant kitti

Size is read from the tensors (``s`` raft-small, ``l`` raft) unless given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    strip_state_dict_prefix,
    wrap_libreyolo_checkpoint,
)

VARIANTS = ("sintel", "kitti")
IMGSZ = 640


def extract_raft_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the RAFT tensors with any DataParallel ``module.`` prefix removed."""
    state_dict = extract_state_dict(checkpoint, prefer_ema=False)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = strip_state_dict_prefix(state_dict, "module.")
    return dict(state_dict)


def build_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    size: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    add_repo_root_to_path()
    from libreyolo.models.raft.model import LibreRAFT

    if not LibreRAFT.can_load(state_dict):
        raise ValueError("Input is not an upstream RAFT checkpoint (raft or raft-small).")
    detected = LibreRAFT.detect_size(state_dict)
    if size is not None and size != detected:
        raise ValueError(f"--size {size} does not match the checkpoint tensors (size {detected}).")
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
    return wrap_libreyolo_checkpoint(
        state_dict,
        model_family=LibreRAFT.FAMILY,
        size=detected,
        nc=1,
        names={0: "flow"},
        task="flow",
        imgsz=IMGSZ,
        variant=variant,
    )


def convert(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
    variant: str | None = None,
    verify: bool = True,
) -> Path:
    state_dict = extract_raft_state_dict(load_checkpoint(input_path))
    print(f"Extracted {len(state_dict)} tensors from {input_path}")
    wrapped = build_checkpoint(state_dict, size=size, variant=variant)

    if verify:
        from libreyolo.models.raft.nn import LibreRAFTNet

        net = LibreRAFTNet(small=wrapped["size"] == "s")
        res = net.load_state_dict(wrapped["model"], strict=True)
        print(f"strict load ok (missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)})")

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"Wrote {out} (size={wrapped['size']}, variant={variant or '-'})")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input", help="upstream raft-*.pth")
    p.add_argument("output", help="LibreRAFT<size>-flow[-variant].pt")
    p.add_argument("--size", choices=["s", "l"], default=None)
    p.add_argument("--variant", choices=list(VARIANTS), default=None)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    convert(a.input, a.output, size=a.size, variant=a.variant, verify=not a.no_verify)


if __name__ == "__main__":
    main()
