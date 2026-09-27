"""Convert the released DEKR-W32-NO-DC checkpoint to LibreYOLO format.

The upstream artifact is served from Deci's public CDN:

    https://d2gjn4b69gu75n.cloudfront.net/models/dekr_w32_no_dc_coco_pose.pth

LibreYOLO links to it rather than mirroring it (no per-artifact redistribution
grant was found), so this script converts a file the user already has.

Usage:
    python weights/convert_dekr_weights.py \
        downloads/dekr_w32_no_dc_coco_pose.pth weights/LibreDEKRw32-pose.pt

The released file is a dict of {net, acc, epoch, optimizer_state_dict,
scaler_state_dict}. Only ``net`` is carried over; optimizer and scaler state are
never written into a public inference checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

# Observed 2026-08-10 at 357,227,441 bytes.
SOURCE_SHA256 = "e5c4797205ddabd5efcebee470ee669c657e6b62f03948d57996e7d9f4022a6b"
SOURCE_URL = (
    "https://d2gjn4b69gu75n.cloudfront.net/models/dekr_w32_no_dc_coco_pose.pth"
)
SOURCE_REVISION = "63de22c404d5740f34f7706c302b37fce3c8fe5d"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(input_path: str, output_path: str, *, allow_digest_mismatch: bool) -> None:
    add_repo_root_to_path()
    from libreyolo.data.pose_metadata import (
        COCO17_FLIP_IDX,
        COCO17_KEYPOINT_NAMES,
        COCO17_OKS_SIGMAS,
        COCO17_SKELETON,
    )
    from libreyolo.models.dekr.model import LibreDEKR
    from libreyolo.models.dekr.nn import LibreDEKRModel
    from libreyolo.models.dekr.utils import strip_module_prefix, unwrap_dekr_checkpoint

    # 1. Verify the digest before the restricted load. A mismatch means this is
    #    not the audited artifact, so refuse rather than silently converting it.
    digest = file_sha256(input_path)
    if digest != SOURCE_SHA256:
        message = (
            f"Digest mismatch for {input_path}: expected {SOURCE_SHA256}, got "
            f"{digest}. This is not the audited DEKR-W32-NO-DC artifact."
        )
        if not allow_digest_mismatch:
            raise SystemExit(f"{message}\nPass --allow-digest-mismatch to override.")
        print(f"WARNING: {message}", file=sys.stderr)

    # 2. Require a dict with `net`; reject arbitrary pickled module objects.
    raw = load_checkpoint(input_path)
    if not isinstance(raw, dict) or "net" not in raw:
        raise SystemExit(
            "Expected a checkpoint dict containing 'net'. Arbitrary pickled "
            "module objects are rejected."
        )

    # 3. Strip exactly one `module.` prefix, dropping optimizer/scaler state.
    state_dict = strip_module_prefix(unwrap_dekr_checkpoint(raw))
    print(f"Extracted {len(state_dict)} parameter entries from {input_path}")

    # 4/5/6. Shape validation decides loadability: K from the heatmap head, a
    #        complete run of K two-channel offset branches, w32 from the branch
    #        widths, and no deformable-convolution state anywhere.
    if not LibreDEKR.can_load(state_dict):
        raise SystemExit(
            "State dict does not match DEKR-W32-NO-DC. The original deformable "
            "DEKR-W32 is a different architecture and is deliberately rejected."
        )
    size = LibreDEKR.detect_size(state_dict)
    num_keypoints = LibreDEKR.detect_num_keypoints(state_dict)
    if size != "w32" or num_keypoints is None:
        raise SystemExit(f"Unsupported DEKR variant (size={size}, K={num_keypoints})")
    print(f"Recognized DEKR-{size.upper()}-NO-DC with {num_keypoints} keypoints")

    # 9. Load strictly into a fresh native model before writing anything.
    model = LibreDEKRModel(num_keypoints=num_keypoints)
    model.load_state_dict(state_dict, strict=True)
    print("Strict load into LibreDEKRModel: OK")

    is_coco = num_keypoints == 17
    # 8. Checkpoint schema v1.0 plus pose and provenance metadata.
    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="dekr",
        size=size,
        nc=1,
        names={0: "person"},
        task="pose",
        supported_tasks=("pose",),
        default_task="pose",
        imgsz=640,
        num_keypoints=num_keypoints,
        keypoint_dim=3,
        keypoint_names=(
            list(COCO17_KEYPOINT_NAMES) if is_coco else None
        ),
        flip_idx=list(COCO17_FLIP_IDX) if is_coco else None,
        skeleton=[list(edge) for edge in COCO17_SKELETON] if is_coco else None,
        oks_sigmas=list(COCO17_OKS_SIGMAS) if is_coco else None,
        variant="no_dc",
        source_url=SOURCE_URL,
        source_revision=SOURCE_REVISION,
        source_sha256=digest,
        source_license=(
            "Apache-2.0 (Deci-AI/super-gradients); no per-artifact "
            "redistribution grant found for the checkpoint, so LibreYOLO links "
            "to the source CDN instead of mirroring it"
        ),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, lean: no optimizer state)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to dekr_w32_no_dc_coco_pose.pth")
    parser.add_argument("output", help="Path to write LibreDEKRw32-pose.pt")
    parser.add_argument(
        "--allow-digest-mismatch",
        action="store_true",
        help="Convert anyway when the source digest is not the audited one",
    )
    args = parser.parse_args()
    convert(
        args.input, args.output, allow_digest_mismatch=args.allow_digest_mismatch
    )
