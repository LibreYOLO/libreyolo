"""Convert a released PP-YOLOE checkpoint to a lean LibreYOLO checkpoint.

The source file is a third-party pickle. This script refuses to deserialize it
until its SHA-256 matches the digest pinned on ``LibrePPYOLOE``, and it keeps
only the model tensors plus LibreYOLO metadata: optimizer state, scaler state,
EMA bookkeeping and any other pickled object in the source file are dropped.

Usage:
    python weights/convert_ppyoloe_weights.py ppyoloe_s_coco.pth weights/LibrePPYOLOEs.pt --size s

``--size`` may be omitted; the size is then inferred from the tensor shapes.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

SOURCE_REVISION = "63de22c404d5740f34f7706c302b37fce3c8fe5d"
SOURCE_REPO = "https://github.com/Deci-AI/super-gradients"
CONVERSION_VERSION = "1"

COCO80_IMAGE_SIZE = 640


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    input_path: str,
    output_path: str,
    size: str | None = None,
    nc: int | None = None,
    allow_unpinned: bool = False,
) -> None:
    add_repo_root_to_path()
    from libreyolo.models.ppyoloe.convert import (
        convert_upstream,
        detect_nb_classes_from_state,
        detect_size_from_state,
        unwrap_ppyoloe_checkpoint,
    )
    from libreyolo.models.ppyoloe.model import LibrePPYOLOE

    source = Path(input_path)
    digest = _sha256(source)
    pinned = LibrePPYOLOE._CHECKPOINT_SHA256
    expected = pinned.get(source.name)
    if expected is None or digest != expected:
        message = (
            f"{source.name} has SHA-256 {digest}, which is not the pinned digest "
            f"for any released PP-YOLOE checkpoint ({sorted(pinned)}). Re-audit "
            "the artifact provenance before converting it."
        )
        if not allow_unpinned:
            raise SystemExit(f"Refusing to deserialize an unpinned file. {message}")
        print(f"WARNING: {message}")

    raw = load_checkpoint(input_path)
    state_dict = convert_upstream(unwrap_ppyoloe_checkpoint(raw))
    print(f"Extracted {len(state_dict)} parameter entries from {input_path}")

    detected_size = detect_size_from_state(state_dict)
    if size is None:
        size = detected_size
        if size is None:
            raise SystemExit(
                "Could not infer the PP-YOLOE size from the tensor shapes. "
                "Pass --size explicitly."
            )
    elif detected_size is not None and detected_size != size:
        raise SystemExit(
            f"--size {size} contradicts the tensor shapes, which say "
            f"'{detected_size}'. Refusing to write a mislabelled checkpoint."
        )

    detected_nc = detect_nb_classes_from_state(state_dict)
    if nc is None:
        nc = detected_nc if detected_nc is not None else 80
    elif detected_nc is not None and detected_nc != nc:
        raise SystemExit(f"--nc {nc} contradicts the head shape ({detected_nc}).")

    from libreyolo import LibrePPYOLOE as _Model

    model = _Model(size=size, nb_classes=nc)
    result = model.model.load_state_dict(state_dict, strict=True)
    print(f"Strict load OK ({result})")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ppyoloe",
        size=size,
        nc=nc,
        task="detect",
        imgsz=COCO80_IMAGE_SIZE,
        supported_tasks=("detect",),
        default_task="detect",
        source_repository=SOURCE_REPO,
        source_revision=SOURCE_REVISION,
        source_filename=source.name,
        source_sha256=digest,
        code_license="Apache-2.0",
        weight_evidence=(
            "Linked from the source provider CDN; not redistributed by LibreYOLO."
        ),
        normalization="mean=[123.675,116.28,103.53] std=[58.395,57.12,57.375] scale=0-255",
        resize_mode="stretch",
        conversion_version=CONVERSION_VERSION,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out} (family=ppyoloe, size={size}, nc={nc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", choices=["s", "m", "l", "x"], default=None)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Convert a file whose digest is not pinned (e.g. your own trained weights).",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.size, args.nc, args.allow_unpinned)
