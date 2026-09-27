"""Convert upstream Dome-DETR weights to the LibreYOLO checkpoint format.

Upstream publishes six checkpoints under ``best_ckpts_dome_2026/`` on
https://huggingface.co/RicePasteM/Dome-DETR — three sizes on each of two
datasets. There is no COCO checkpoint for this family, so every canonical
filename carries a dataset suffix:

    python weights/convert_domedetr_weights.py \
        best_ckpts_dome_2026/aitod-s-best.pth \
        weights/LibreDOMEDETRs-aitod.pt --size s --variant aitod

    python weights/convert_domedetr_weights.py \
        best_ckpts_dome_2026/dome-l-visdrone_converted.pth \
        weights/LibreDOMEDETRl-visdrone.pt --size l --variant visdrone

The conversion is a metadata wrap: upstream key names already match the port
(that is deliberate, see ``models/domedetr/encoder.py``), so no tensor is
renamed or reshaped.

Two pieces of metadata are load-bearing rather than cosmetic:

``weight_variant``
    Selects the PAQI query budget (AI-TOD-V2 300..1500, VisDrone 250..500) and
    the decoder depth for L (4 layers on AI-TOD-V2, 6 on VisDrone). Getting it
    wrong builds a different model, not just a different label.

``names``
    The two datasets do not share a taxonomy, so class names must travel with
    the checkpoint. Both are 1-indexed upstream (``remap_mscoco_category:
    False`` means raw COCO category ids are used as labels), which is why the
    class counts are 9 and 12 rather than 8 and 10.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


# AI-TOD-V2, from Chasel-Tsui/mmdet-aitod ``mmdet/datasets/aitodv2.py``.
# Index 0 is unused: annotations are 1-indexed and upstream sets num_classes=9.
AITOD_NAMES = {
    0: "background",
    1: "airplane",
    2: "bridge",
    3: "storage-tank",
    4: "ship",
    5: "swimming-pool",
    6: "vehicle",
    7: "person",
    8: "wind-mill",
}

# VisDrone, from upstream ``tools/dataset/visdrone2coco.py`` (ids 1..10).
# Ids 0 and 11 are the dataset's own non-object buckets.
VISDRONE_NAMES = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VARIANT_NAMES = {"aitod": AITOD_NAMES, "visdrone": VISDRONE_NAMES}

UPSTREAM_MODEL_REPO = "RicePasteM/Dome-DETR"
UPSTREAM_MODEL_REVISION = "530230620d1f3261a267d462989cddf204cc6e10"
WEIGHT_TERMS = (
    "Academic research purposes only (upstream model card); the same card also "
    "claims Apache-2.0. See weights/LICENSE_NOTICE.txt."
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(input_path: str, output_path: str, size: str, variant: str) -> None:
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw, prefer_ema=True)
    print(f"Extracted {len(state_dict)} parameter entries from {input_path}")

    names = VARIANT_NAMES[variant]
    nc = len(names)

    add_repo_root_to_path()
    from libreyolo.models.domedetr.nn import LibreDOMEDETRModel

    model = LibreDOMEDETRModel(config=size, nb_classes=nc, variant=variant)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"missing: {list(result.missing_keys)}")
    print(f"unexpected: {list(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(
            "Refusing to write: upstream keys do not match the port exactly. "
            "Check --size/--variant against the source filename."
        )

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="domedetr",
        size=size,
        nc=nc,
        names=names,
        task="detect",
        imgsz=800,
        supported_tasks=("detect",),
        default_task="detect",
        weight_variant=variant,
        source=(
            f"{UPSTREAM_MODEL_REPO}@{UPSTREAM_MODEL_REVISION}/"
            f"best_ckpts_dome_2026/{Path(input_path).name}"
        ),
        source_sha256=_sha256(input_path),
        license=WEIGHT_TERMS,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_checkpoint(wrapped, tmp)
    tmp.replace(out)  # atomic
    print(f"Wrote {out} (size={size}, variant={variant}, nc={nc})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--size", required=True, choices=["s", "m", "l"])
    p.add_argument("--variant", required=True, choices=["aitod", "visdrone"])
    args = p.parse_args()
    convert(args.input, args.output, args.size, args.variant)
