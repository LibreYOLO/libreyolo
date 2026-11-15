"""Convert official RiO-DETRv2 weights to LibreYOLO format.

Source code: RicePasteM/RiO-DETRv2 commit
ab8b22df5d800a927774c8c67e77fd29c0fe6967 (Apache-2.0; preserved as
LibreYOLO/RiO-DETRv2 tag apache-2.0-ab8b22d).  Source weights:
huggingface.co/RicePasteM/RiO-DETRv2 at revision
5f578f57929aded1962ebf8c5584a4771461124f (Apache-2.0):
``dota_1_ss/riov2_hgnetv2_<size>_dota_1_ss.pth`` and
``diorr/riov2_hgnetv2_<size>_diorr.pth``.

Steps:
  1. Read ``ckpt["model"]`` (upstream stores the EMA detector there).
  2. Remap the encoder input projection ``.conv/.norm`` -> ``.0/.1`` and drop
     the training-only DINOv3 distillation projector
     (``encoder.feature_projector``).  No tensor value changes.
  3. Strict-load every converted tensor into the LibreYOLO graph sized to the
     checkpoint's saved anchors (1024 px DOTA, 800 px DIOR-R).
  4. Wrap with v1.0 metadata and save atomically.

Usage::

    python weights/convert_riodetrv2_weights.py riov2_hgnetv2_n_dota_1_ss.pth \\
        weights/LibreRiODETRv2n-obb.pt --size n
    python weights/convert_riodetrv2_weights.py riov2_hgnetv2_n_diorr.pth \\
        weights/LibreRiODETRv2n-obb-diorr.pt --size n
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def _atomic_save(checkpoint: dict, output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        save_checkpoint(checkpoint, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_weights(input_path: str, output_path: str, size: str) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.riodetrv2.convert import (
        convert_upstream,
        is_upstream_state_dict,
    )
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    raw = load_checkpoint(input_path)
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    if not is_upstream_state_dict(state_dict):
        raise ValueError(f"{input_path} is not an upstream RiO-DETRv2 checkpoint")
    detected = LibreRiODETRv2.detect_size(state_dict)
    if detected != size:
        raise ValueError(f"--size={size} conflicts with the checkpoint ({detected})")

    out = {k: v.clone() for k, v in convert_upstream(state_dict).items()}
    nc = LibreRiODETRv2.detect_nb_classes(out)
    wrapper = LibreRiODETRv2(None, size=size, nb_classes=nc, device="cpu")
    wrapper._prepare_model_for_state_dict(out)
    expected = wrapper.model.state_dict()
    missing = sorted(expected.keys() - out.keys())
    unexpected = sorted(out.keys() - expected.keys())
    mismatched = sorted(k for k in expected.keys() & out.keys() if expected[k].shape != out[k].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Converted tensors do not load strictly: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatches={mismatched[:5]}"
        )
    wrapper.model.load_state_dict(out, strict=True)

    checkpoint = wrap_libreyolo_checkpoint(
        out,
        model_family="riodetrv2",
        size=size,
        nc=nc,
        names=LibreRiODETRv2.default_checkpoint_names(nc),
        task="obb",
        imgsz=int(wrapper.input_size),
    )
    validate_checkpoint_metadata(checkpoint)
    _atomic_save(checkpoint, output_path)
    print(f"Saved {output_path} ({len(out)} tensors, nc={nc}, imgsz={wrapper.input_size})")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RiO-DETRv2 weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream RiO-DETRv2 checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=["n", "s", "m", "l", "x"])
    args = parser.parse_args()
    convert_weights(args.input, args.output, args.size)
