"""Convert upstream PP-LiteSeg Cityscapes weights to LibreYOLO format.

The four released artifacts share one architecture per backbone, so the tiny /
base distinction is inferred from the tensors while the 50 / 75 recipe is not
inferable at all -- both recipes produce byte-compatible state dicts. The
recipe therefore comes from ``--size`` or from a trusted upstream filename, and
ambiguous input is rejected rather than guessed.

Usage:
    python weights/convert_ppliteseg_weights.py \
        upstream/pp_lite_t_seg50_cityscapes.pth weights/LibrePPLiteSegt50-sem.pt
    python weights/convert_ppliteseg_weights.py in.pth out.pt --size b75

The released checkpoints are trained on Cityscapes and are NON-COMMERCIAL; the
conversion stamps that into the checkpoint metadata so it travels with the file
(see libreyolo/models/ppliteseg/NOTICE).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

SIZES = ("t50", "b50", "t75", "b75")

# SHA-256 of each upstream artifact, recomputed on 2026-08-10. Identical byte
# sizes across the two resolution recipes do not imply identical contents, so
# every digest is pinned separately.
SOURCE_DIGESTS = {
    "t50": "ae9ad0cae645ebdfb8de661e8dbf1c33e08c0d90997954429b467e70e9ca4194",
    "b50": "f7a6769dd37290ee1145c5f1aa2a669b5421591d3afaf8629910614405fe7122",
    "t75": "1fdd809572a1b3168727ed0dea32da287c9917ed0ebbfdf8ecab87a3116733f6",
    "b75": "383bfb69b3ffb643224dbf76047acacd9b48a1564958a20f1aca69c5453dca3e",
}
SOURCE_URLS = {
    size: f"https://d2gjn4b69gu75n.cloudfront.net/models/pp_lite_{size[0]}_seg{size[1:]}_cityscapes.pth"
    for size in SIZES
}
SOURCE_REVISION = "Deci-AI/super-gradients@63de22c404d5740f34f7706c302b37fce3c8fe5d"
STDC_REVISION = "MichaelFan01/STDC-Seg@59ff37fbd693b99972c76fcefe97caa14aeb619f"
CITYSCAPES_LICENSE_URL = "https://www.cityscapes-dataset.com/license/"

_FILENAME_RE = re.compile(r"pp_lite_(?P<backbone>[tb])_seg(?P<recipe>50|75)", re.IGNORECASE)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_from_filename(path: str | Path) -> str | None:
    match = _FILENAME_RE.search(Path(path).name)
    if not match:
        return None
    return f"{match.group('backbone').lower()}{match.group('recipe')}"


def convert(input_path: str, output_path: str, size: str | None = None) -> None:
    add_repo_root_to_path()
    from libreyolo.models.ppliteseg.model import (
        CITYSCAPES_NAMES,
        WEIGHT_LICENSE,
        LibrePPLiteSeg,
    )
    from libreyolo.models.ppliteseg.nn import SIZE_CONFIGS

    resolved = size or size_from_filename(input_path)
    if resolved is None:
        raise SystemExit(
            f"Cannot tell the 50 recipe from the 75 recipe for {input_path!r}: the two "
            "share an architecture and a class count. Pass --size explicitly, or use "
            "the canonical upstream filename."
        )
    if resolved not in SIZES:
        raise SystemExit(f"--size must be one of {SIZES}, got {resolved!r}")

    digest = sha256(input_path)
    expected = SOURCE_DIGESTS[resolved]
    if digest != expected:
        raise SystemExit(
            f"Digest mismatch for {input_path}: got {digest}, expected {expected} for "
            f"size {resolved}. Refusing to deserialize an unverified checkpoint."
        )

    raw = load_checkpoint(input_path)
    if not isinstance(raw, dict) or "net" not in raw:
        raise SystemExit(f"{input_path} has no top-level 'net' key; not an upstream artifact.")
    source_state = raw["net"]
    if not isinstance(source_state, dict):
        raise SystemExit("Upstream 'net' entry is not a state dict.")

    import torch

    non_tensor = [k for k, v in source_state.items() if not torch.is_tensor(v)]
    if non_tensor:
        raise SystemExit(
            f"Upstream state dict holds {len(non_tensor)} non-tensor entries "
            f"(e.g. {non_tensor[:3]}); refusing to repackage arbitrary objects."
        )

    # Strip exactly one DDP prefix, never a chain.
    state_dict = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in source_state.items()
    }
    print(f"Extracted {len(state_dict)} tensors from {input_path}")

    backbone = LibrePPLiteSeg.detect_backbone(state_dict)
    expected_backbone = SIZE_CONFIGS[resolved]["backbone"]
    if backbone != expected_backbone:
        raise SystemExit(
            f"Tensor evidence says the backbone is {backbone!r} but size {resolved!r} "
            f"expects {expected_backbone!r}."
        )
    nc = LibrePPLiteSeg.detect_nb_classes(state_dict)
    if nc is None:
        raise SystemExit("Could not read the class count from the segmentation heads.")

    # Strict load through the production model, auxiliary heads included: the
    # canonical artifact stays trainable.
    model = LibrePPLiteSeg(size=resolved, nb_classes=nc, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    print(f"Strict load OK: size={resolved} backbone={backbone} nc={nc}")

    imgsz_h, imgsz_w = SIZE_CONFIGS[resolved]["imgsz"]
    train_h, train_w = SIZE_CONFIGS[resolved]["train_crop"]
    names = dict(CITYSCAPES_NAMES) if nc == len(CITYSCAPES_NAMES) else None

    wrapped = wrap_libreyolo_checkpoint(
        {key: value.cpu() for key, value in state_dict.items()},
        model_family="ppliteseg",
        size=resolved,
        nc=nc,
        names=names,
        task="semantic",
        supported_tasks=("semantic",),
        default_task="semantic",
        # Legacy square imgsz is the long side; the real rectangle is the pair.
        imgsz=max(imgsz_h, imgsz_w),
        imgsz_h=imgsz_h,
        imgsz_w=imgsz_w,
        train_imgsz_h=train_h,
        train_imgsz_w=train_w,
        normalization="imagenet",
        resize_mode="direct",
        ignore_index=255,
        weight_license=WEIGHT_LICENSE,
        weight_license_url=CITYSCAPES_LICENSE_URL,
        weight_dataset="Cityscapes",
        weight_commercial_use=False,
        source_url=SOURCE_URLS[resolved],
        source_sha256=digest,
        source_revision=SOURCE_REVISION,
        architecture_licenses="Apache-2.0 (super-gradients), MIT (STDC-Seg)",
        stdc_revision=STDC_REVISION,
    )

    out = Path(output_path)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)
    print(f"Wrote {out} (size={resolved}, nc={nc}, imgsz={imgsz_h}x{imgsz_w})")
    print(
        "NOTE: this checkpoint derives from Cityscapes and is restricted to "
        f"NON-COMMERCIAL use ({CITYSCAPES_LICENSE_URL})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", choices=list(SIZES), default=None)
    args = parser.parse_args()
    convert(args.input, args.output, args.size)
