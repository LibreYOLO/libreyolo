"""Convert upstream YOLO-NAS-R (rotated / OBB) DOTA2 weights to LibreYOLO format.

Upstream source (code, Apache-2.0):
    https://github.com/Deci-AI/super-gradients pull request 2014,
    pinned commit 69141b55c1161d939939a270523a7eca5a645f72

Upstream weights (NOT redistributable -- Deci's separate YOLO-NAS-R licence,
see ``weights/LICENSE_NOTICE.txt``) are downloaded from Deci's CDN:
    https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_r_{s,m,l}_dota2.pth

The rotated head keeps SuperGradients' module names, so this is a metadata
wrap after unwrapping the EMA weights -- no key remapping. The file is a
third-party pickle, so its SHA256 is verified against the pins in
``LibreYOLONAS._DECI_CHECKPOINT_SHA256`` *before* it is unpickled.

Usage:
    python weights/convert_yolonas_obb_weights.py \\
        downloads/yolo_nas_r_s_dota2.pth weights/LibreYOLONASs-obb.pt --size s
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_SIZES = ("s", "m", "l")
_UPSTREAM_FILENAME = "yolo_nas_r_{size}_dota2.pth"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    *,
    allow_unpinned: bool = False,
    verify: bool = False,
) -> None:
    add_repo_root_to_path()
    import torch

    from libreyolo.models.yolonas.model import (
        YOLONAS_OBB_CLASS_NAMES,
        LibreYOLONAS,
    )
    from libreyolo.models.yolonas.utils import unwrap_yolonas_checkpoint

    expected_name = _UPSTREAM_FILENAME.format(size=size)
    actual_name = Path(input_path).name
    if actual_name != expected_name and not allow_unpinned:
        raise SystemExit(
            f"Expected the upstream file {expected_name!r} for --size {size}, got "
            f"{actual_name!r}. Pass --allow-unpinned only if you know why."
        )

    expected_sha = LibreYOLONAS._DECI_CHECKPOINT_SHA256.get(expected_name)
    actual_sha = _sha256(input_path)
    if expected_sha is None:
        if not allow_unpinned:
            raise SystemExit(
                f"No pinned SHA256 for {expected_name!r}; refusing to load."
            )
    elif actual_sha != expected_sha:
        raise SystemExit(
            f"SHA256 mismatch for {actual_name}: expected {expected_sha}, got "
            f"{actual_sha}. Refusing to unpickle a possibly tampered file."
        )
    print(f"SHA256 verified: {actual_sha}")

    # Only now, after the hash gate, do we unpickle the third-party file.
    raw = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = dict(unwrap_yolonas_checkpoint(raw))
    print(f"Extracted {len(state_dict)} parameter entries from {input_path}")

    if not LibreYOLONAS.is_obb_state_dict(state_dict):
        raise SystemExit(
            "This is not a YOLO-NAS-R rotated checkpoint (no heads.*.rot_pred.*). "
            "Use weights/convert_yolonas_weights or the detect/pose path instead."
        )

    detected_size = LibreYOLONAS.detect_size(state_dict)
    if detected_size != size:
        raise SystemExit(
            f"State dict looks like size {detected_size!r}, not {size!r}. "
            "Refusing an ambiguous conversion."
        )
    nc = LibreYOLONAS.detect_nb_classes(state_dict)
    if nc is None:
        raise SystemExit("Could not read the class count from the rotated head.")

    names = (
        dict(enumerate(YOLONAS_OBB_CLASS_NAMES))
        if nc == len(YOLONAS_OBB_CLASS_NAMES)
        else {i: str(i) for i in range(nc)}
    )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="yolonas",
        size=size,
        nc=nc,
        names=names,
        task="obb",
        supported_tasks=("detect", "pose", "obb"),
        default_task="detect",
        imgsz=LibreYOLONAS.OBB_INPUT_SIZES[size],
        provenance={
            "upstream_repo": "https://github.com/Deci-AI/super-gradients",
            "upstream_pr": "https://github.com/Deci-AI/super-gradients/pull/2014",
            "upstream_commit": "69141b55c1161d939939a270523a7eca5a645f72",
            "code_license": "Apache-2.0",
            "weights_license": "YOLO-NAS-R License (Deci.AI) -- non-redistributable",
            "weights_source": (
                "https://d2gjn4b69gu75n.cloudfront.net/models/" + expected_name
            ),
            "weights_sha256": actual_sha,
            "dataset": "DOTA2",
        },
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out} (task=obb, size={size}, nc={nc})")

    if verify:
        from libreyolo import LibreYOLO

        model = LibreYOLO(str(out))
        print(
            f"Verified load: family={model.FAMILY} task={model.task} size={model.size}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Upstream yolo_nas_r_<size>_dota2.pth")
    parser.add_argument("output", help="Output LibreYOLONAS<size>-obb.pt")
    parser.add_argument("--size", required=True, choices=list(_SIZES))
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Skip the filename/SHA256 gate (only for a checkpoint you trust).",
    )
    parser.add_argument("--verify", action="store_true", help="Reload the result.")
    args = parser.parse_args()
    convert_weights(
        args.input,
        args.output,
        args.size,
        allow_unpinned=args.allow_unpinned,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
