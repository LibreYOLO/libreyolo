"""Convert official UniMatch / GMFlow optical flow checkpoints to LibreYOLO format.

Upstream checkpoints (https://github.com/autonomousvision/unimatch,
MODEL_ZOO.md) are ``{"model": state_dict}``. The model code is vendored
unchanged under ``libreyolo/models/unimatch/_vendor``, so conversion only wraps
the tensors in the LibreYOLO v1.0 checkpoint schema: the keys are unchanged and
the load is 0 missing / 0 unexpected.

Sizes and canonical outputs::

    gmflow-scale1-things-e9887eda.pth                         -> LibreUniMatchs-flow.pt
    gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth -> LibreUniMatchl-flow.pt
    gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth            -> LibreUniMatchl-flow-sintel.pt
    gmflow-scale2-regrefine6-kitti15-25b554d7.pth             -> LibreUniMatchl-flow-kitti.pt

Usage::

    python weights/convert_unimatch_weights.py \\
        gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \\
        weights/LibreUniMatchl-flow.pt

Size is detected from the tensors and the variant from the upstream filename;
``--size`` and ``--variant`` override. ``--verify`` reloads the file through
``LibreYOLO`` and runs a smoke forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

_IMGSZ = 640
_VARIANTS = ("sintel", "kitti")


def detect_variant(filename: str) -> str | None:
    """Return the dataset variant encoded in an upstream checkpoint filename."""
    name = Path(filename).name.lower()
    if "sintel" in name:
        return "sintel"
    if "kitti" in name:
        return "kitti"
    return None


def canonical_name(size: str, variant: str | None) -> str:
    return f"LibreUniMatch{size}-flow" + (f"-{variant}" if variant else "") + ".pt"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(
    state_dict: dict,
    *,
    size: str | None = None,
    variant: str | None = None,
    upstream_name: str | None = None,
    upstream_sha256: str | None = None,
) -> dict:
    """Wrap an upstream GMFlow state dict in LibreYOLO checkpoint metadata."""
    add_repo_root_to_path()
    from libreyolo.models.unimatch.model import LibreUniMatch
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    if not LibreUniMatch.can_load(state_dict):
        raise ValueError(
            "This does not look like a UniMatch / GMFlow flow checkpoint "
            "(expected backbone.*, transformer.layers.* and feature_flow_attn.* keys)."
        )
    detected = LibreUniMatch.detect_size(state_dict)
    if size is None:
        size = detected
    if size is None:
        raise ValueError(
            "Unsupported GMFlow graph: LibreYOLO ships scale1 (size s) and "
            "scale2 with regression refinement (size l)."
        )
    if detected is not None and detected != size:
        raise ValueError(f"--size {size} does not match the tensors (size {detected}).")
    if variant is not None and variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS} or None, got {variant!r}")
    if variant is not None and size != "l":
        raise ValueError("Sintel and KITTI variants exist for size l only.")

    # Dry-load: the vendored graph must take every tensor unchanged.
    model = LibreUniMatch(None, size=size, device="cpu")
    res = model.model.load_state_dict(state_dict, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise ValueError(
            f"Key mismatch: missing={res.missing_keys} unexpected={res.unexpected_keys}"
        )

    return wrap_libreyolo_checkpoint(
        dict(state_dict),
        model_family="unimatch",
        size=size,
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=_IMGSZ,
        variant=variant,
        upstream_checkpoint=upstream_name,
        upstream_sha256=upstream_sha256,
    )


def convert_weights(
    input_path: str,
    output_path: str | None = None,
    *,
    size: str | None = None,
    variant: str | None = "auto",
) -> Path:
    print(f"Loading upstream UniMatch weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=False)
    print(f"Found {len(state_dict)} parameter entries")
    if variant == "auto":
        variant = detect_variant(input_path)
    ckpt = convert_state_dict(
        state_dict,
        size=size,
        variant=variant,
        upstream_name=Path(input_path).name,
        upstream_sha256=_sha256(input_path),
    )
    out = Path(output_path) if output_path else Path(input_path).with_name(
        canonical_name(ckpt["size"], variant)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(ckpt, tmp)
    tmp.replace(out)
    print(f"Saved {out} (size={ckpt['size']}, variant={variant})")
    return out


def verify_conversion(converted_path: str | Path) -> None:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    m = LibreYOLO(str(converted_path), device="cpu")
    print(f"  family={m.FAMILY} size={m.size} task={m.task} nc={m.nb_classes} names={m.names}")
    x = torch.zeros(1, 3, 64, 96)
    with torch.no_grad():
        flow = m._forward_pair(x, x)
    print(f"  forward OK, flow shape {tuple(flow.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="Upstream GMFlow checkpoint (.pth)")
    parser.add_argument("output", nargs="?", default=None, help="Output .pt (default: canonical name beside the input)")
    parser.add_argument("--size", choices=["s", "l"], default=None)
    parser.add_argument("--variant", choices=["auto", "none", *_VARIANTS], default="auto")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    variant = None if args.variant == "none" else args.variant
    out = convert_weights(args.input, args.output, size=args.size, variant=variant)
    if args.verify:
        verify_conversion(out)
