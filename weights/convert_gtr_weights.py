"""Convert GTR detection or instance-segmentation EMA weights to a strict
LibreYOLO checkpoint (``gtr_*_coco.pth`` or ``gtrseg_*_coco.pth``).

Source: Intellindust-AI-Lab/GTR, MIT, revision
782e737efe2e6437ac537fbdcee089673d3376c1. Learned tensors are unchanged.
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
    from libreyolo.models.gtr.model import LibreGTR
    from libreyolo.models.gtr.nn import LibreGTRModel
    from libreyolo.models.gtr.seg import SEG_MASK_DOWNSAMPLE_RATIO

    state = extract_state_dict(load_checkpoint(input_path), prefer_ema=True)
    detected = LibreGTR.detect_size(state)
    if not LibreGTR.can_load(state) or detected is None:
        raise ValueError("Not a supported GTR detection or segmentation checkpoint")
    if size is not None and size != detected:
        raise ValueError(f"Checkpoint size is {detected}, not {size}")
    nc = LibreGTR.detect_nb_classes(state)
    task = LibreGTR.detect_checkpoint_task(state) or "detect"
    model = LibreGTRModel(
        detected,
        nc,
        mask_downsample_ratio=SEG_MASK_DOWNSAMPLE_RATIO if task == "segment" else None,
    )
    model.load_state_dict(state, strict=True)
    checkpoint = wrap_libreyolo_checkpoint(
        state,
        model_family="gtr",
        size=detected,
        nc=nc,
        task=task,
        imgsz=640,
        supported_tasks=LibreGTR.SUPPORTED_TASKS,
        default_task="detect",
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
