"""Convert official Apple Depth Pro 2 checkpoints into LibreYOLO format.

Official Apple checkpoints from ``apple/DepthPro2-{S,B,L,H}`` (and the companion
``apple/DepthPro2-focal`` field-of-view estimator) already use the exact module
and parameter names preserved by LibreYOLO's native port, so conversion merges
the optional FoV weights under ``fov.*`` and wraps the state dict with strict
LibreYOLO v1.0 depth metadata without modifying learned tensors.

Canonical variants:

* ``apple/DepthPro2-S`` (+ ``apple/DepthPro2-focal``) -> ``LibreDepthPro2s-depth.pt``
* ``apple/DepthPro2-B`` (+ ``apple/DepthPro2-focal``) -> ``LibreDepthPro2b-depth.pt``
* ``apple/DepthPro2-L`` (+ ``apple/DepthPro2-focal``) -> ``LibreDepthPro2l-depth.pt``
* ``apple/DepthPro2-H`` (+ ``apple/DepthPro2-focal``) -> ``LibreDepthPro2h-depth.pt``

Usage::

    python weights/convert_depthpro2_weights.py \\
        depthpro2_s/model.safetensors LibreDepthPro2s-depth.pt \\
        --fov depthpro2_focal/model.safetensors --verify
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from _conversion_utils import add_repo_root_to_path, save_checkpoint


def _dry_load(state_dict: dict, size: str, *, with_fov: bool) -> None:
    """Verify exact state-dict compatibility before writing."""
    add_repo_root_to_path()
    from libreyolo.models.depthpro2.nn import build_depthpro2_model

    model = build_depthpro2_model(size, fov=with_fov)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {result.missing_keys}")
    print(f"Unexpected keys: {result.unexpected_keys}")
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Depth Pro 2 checkpoint is not an exact match for the native architecture."
        )


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    fov_path: str | None = None,
    size: str | None = None,
) -> dict:
    """Wrap an official Depth Pro 2 checkpoint (+ optional FoV checkpoint) in LibreYOLO v1.0 metadata."""
    add_repo_root_to_path()
    from libreyolo.models.depthpro2.convert import (
        detect_size,
        has_fov_weights,
        is_depthpro2_state_dict,
        load_raw_state_dict,
        merge_fov_state_dict,
        wrap_upstream_state_dict,
    )

    print(f"Loading official Depth Pro 2 weights from {input_path}")
    depth_sd = load_raw_state_dict(input_path)
    print(f"Found {len(depth_sd)} depth parameter entries")

    if not is_depthpro2_state_dict(depth_sd):
        raise ValueError(
            "Input does not look like a supported Depth Pro 2 checkpoint."
        )

    detected = detect_size(depth_sd)
    if detected is None:
        raise ValueError("Could not detect Depth Pro 2 size from encoder.cls_token.")
    if size is None:
        size = detected
        print(f"Auto-detected size: {size}")
    elif size != detected:
        raise ValueError(
            f"--size {size!r} contradicts the checkpoint signature ({detected!r})."
        )

    fov_sd = None
    if fov_path is not None:
        print(f"Loading companion FoV estimator weights from {fov_path}")
        fov_sd = load_raw_state_dict(fov_path)
        print(f"Found {len(fov_sd)} FoV parameter entries")

    merged = merge_fov_state_dict(depth_sd, fov_sd)
    with_fov = has_fov_weights(merged)

    canonical_name = f"LibreDepthPro2{size}-depth.pt"
    if Path(output_path).name != canonical_name:
        raise ValueError(
            f"Depth Pro 2 output must use the canonical filename {canonical_name!r}; "
            f"got {Path(output_path).name!r}."
        )

    _dry_load(merged, size, with_fov=with_fov)
    wrapped = wrap_upstream_state_dict(merged, size)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    os.replace(temporary, output)
    print(f"Wrote {output} (with_fov={with_fov})")
    return wrapped


def verify_conversion(output_path: str) -> None:
    """Validate metadata, load via LibreYOLO factory, and run a tiny smoke predict."""
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    ckpt = torch.load(output_path, map_location="cpu", weights_only=True)
    errors = validate_checkpoint_metadata(ckpt, strict=True)
    if errors:
        raise RuntimeError("Metadata validation failed: " + "; ".join(errors))

    model = LibreYOLO(output_path, device="cpu")
    dummy = np.zeros((64, 96, 3), dtype=np.uint8)
    result = model.predict(dummy)
    assert result.depth_map is not None
    assert result.depth_map.data.shape == (64, 96)
    print(
        f"Verified {output_path}: family={model.family}, size={model.size}, "
        f"metric={result.depth_map.metric}, encoding={result.depth_map.encoding}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official Apple Depth Pro 2 weights to LibreYOLO format."
    )
    parser.add_argument("input", help="Path to upstream Depth Pro 2 .safetensors or .pt")
    parser.add_argument("output", help="Path to output LibreDepthPro2{s,b,l,h}-depth.pt")
    parser.add_argument(
        "--fov",
        default=None,
        help="Optional path to companion DepthPro2-focal .safetensors or .pt",
    )
    parser.add_argument(
        "--size",
        choices=("s", "b", "l", "h"),
        default=None,
        help="Override auto-detected variant size",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate metadata and run a CPU smoke forward after writing",
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, fov_path=args.fov, size=args.size)
    if args.verify:
        verify_conversion(args.output)


if __name__ == "__main__":
    main()
