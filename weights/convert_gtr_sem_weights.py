"""Convert GTR Cityscapes semantic EMA weights to a strict LibreYOLO checkpoint.

Source: Intellindust-AI-Lab/GTR, MIT, revision
782e737efe2e6437ac537fbdcee089673d3376c1; weights from Phoenix8125/GTR
``semseg/gtrsemseg_{s,m,l,x}_cityscapes.pth``. Learned tensors are unchanged.
The weights are trained on Cityscapes, whose terms are non-commercial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def convert(input_path: str, output_path: str, size: str | None = None):
    add_repo_root_to_path()
    from libreyolo.models.gtr import sem
    from libreyolo.models.gtr.model import LibreGTR
    from libreyolo.models.ppliteseg.model import CITYSCAPES_LICENSE_URL, WEIGHT_LICENSE

    state = extract_state_dict(load_checkpoint(input_path), prefer_ema=True)
    detected = LibreGTR.detect_size(state)
    if not (LibreGTR.can_load(state) and sem.is_semantic_state_dict(state)):
        raise ValueError("Not a supported GTR semantic checkpoint")
    if detected is None or (size is not None and size != detected):
        raise ValueError(f"Checkpoint size is {detected}, not {size}")
    nc = LibreGTR.detect_nb_classes(state)
    model = sem.LibreGTRSemModel(detected, nc)
    model.load_state_dict(state, strict=True)
    names = sem.CITYSCAPES_NAMES if nc == len(sem.CITYSCAPES_NAMES) else None
    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="gtr",
        size=detected,
        nc=nc,
        names=names,
        task="semantic",
        # The square training/evaluation window; inference slides it over
        # the native 1024x2048 canvas.
        imgsz=sem.SEM_WINDOW,
        imgsz_h=sem.SEM_WINDOW,
        imgsz_w=2 * sem.SEM_WINDOW,
        weight_license=WEIGHT_LICENSE,
        weight_license_url=CITYSCAPES_LICENSE_URL,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(path)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--size", choices=("s", "m", "l", "x"))
    args = parser.parse_args()
    convert(args.input, args.output, args.size)
