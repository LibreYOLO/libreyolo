"""Convert a ptlflow DPFlow checkpoint to the LibreYOLO checkpoint format.

ptlflow publishes DPFlow as Lightning checkpoints (``{"state_dict": ...}``)
on its GitHub release ``weights1``. The tensor names already match
:class:`libreyolo.models.dpflow._vendor.dpflow.DPFlow`, so conversion only
unwraps the state dict, checks it is a complete DPFlow layout, and wraps the
498 unchanged tensors in checkpoint schema v1.0 (``task="flow"``, ``nc=1``,
``names={0: "flow"}``).

The weights are CC BY-NC-SA 4.0 (trained within ptlflow); the license is
recorded in the checkpoint metadata.

Usage::

    python weights/convert_dpflow_weights.py dpflow-things-2012b5d6.ckpt \
        weights/LibreDPFlowm-flow.pt --variant things
    python weights/convert_dpflow_weights.py dpflow-sintel-b44b072c.ckpt \
        weights/LibreDPFlowm-flow-sintel.pt --variant sintel
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    save_checkpoint,
    strip_state_dict_prefix,
    wrap_libreyolo_checkpoint,
)

VARIANTS = ("things", "sintel", "kitti", "spring")
CANONICAL_VARIANT = "things"
WEIGHT_LICENSE = "CC-BY-NC-SA-4.0"
WEIGHT_LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
SOURCE = "https://github.com/hmorimitsu/ptlflow/releases/tag/weights1"
IMGSZ = 960


def output_name(variant: str) -> str:
    """Canonical LibreYOLO filename for a ptlflow DPFlow variant."""
    if variant == CANONICAL_VARIANT:
        return "LibreDPFlowm-flow.pt"
    return f"LibreDPFlowm-flow-{variant}.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_state_dict(raw) -> dict[str, torch.Tensor]:
    """Return the DPFlow tensor dict from a ptlflow checkpoint object."""
    add_repo_root_to_path()
    from libreyolo.models.dpflow.model import LibreDPFlow

    state_dict = extract_state_dict(raw, state_dict_keys=("state_dict", "model"))
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a ptlflow checkpoint with a 'state_dict' mapping.")
    if not LibreDPFlow.can_load(state_dict):
        # Some exports wrap the module once more as ``model.``.
        state_dict = strip_state_dict_prefix(state_dict, "model.")
    if not LibreDPFlow.can_load(state_dict):
        raise ValueError("Checkpoint is not a ptlflow DPFlow state dict.")
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def convert(input_path: str, output_path: str, variant: str, verify: bool = True) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}.")
    src = Path(input_path)
    raw = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = convert_state_dict(raw)
    print(f"Extracted {len(state_dict)} tensors from {src}")

    if verify:
        from libreyolo.models.dpflow._vendor.dpflow import DPFlow

        net = DPFlow()
        result = net.load_state_dict(state_dict, strict=True)
        print("missing:", list(result.missing_keys), "unexpected:", list(result.unexpected_keys))

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="dpflow",
        size="m",
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=IMGSZ,
        variant=variant,
        weight_license=WEIGHT_LICENSE,
        weight_license_url=WEIGHT_LICENSE_URL,
        weight_commercial_use=False,
        source=SOURCE,
        source_file=src.name,
        source_sha256=_sha256(src),
    )
    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.rename(out)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="ptlflow dpflow-<variant>-<hash>.ckpt")
    parser.add_argument("output", nargs="?", help="output .pt (default: canonical name)")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    output = args.output or output_name(args.variant)
    convert(args.input, output, args.variant, verify=not args.no_verify)


if __name__ == "__main__":
    main()
