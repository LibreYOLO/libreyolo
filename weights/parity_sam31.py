"""SAM 3.1 image parity: LibreSAM31 against Meta's reference implementation.

Runs on a CUDA machine with the reference ``sam3`` package installed from
https://github.com/facebookresearch/sam3 (SAM License; installed and called
as a black box, never vendored) and the pinned ``sam3.1_multiplex.pt``.

Compares, in fp32 on the same images:

* text prompts: the reference image predictor (``Sam3Processor`` text path)
  against ``LibreSAM31.predict(text=...)``; instances are matched by mask IoU
  and the script reports matched counts, score deltas and mask IoU;
* point and box prompts: the reference instance-interactive predictor against
  ``LibreSAM31.predict(points=/bboxes=)``; reports mask IoU and IoU-score delta.

A SAM 3 baseline runs the same comparison with ``--checkpoint sam3.pt`` and a
snapshot converted from it, which shows the gap the existing SAM 3
integration already has against the reference.

Usage:
    python weights/parity_sam31.py --checkpoint sam3.1_multiplex.pt \\
        --snapshot /path/to/LibreSAM31large --images a.jpg b.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEXT_PROMPTS = ("person", "building", "tree", "car")
SCORE_FLOOR = 0.3


class ReferenceSAM31:
    """Meta's reference SAM 3 image model, called through its public API.

    The reference image builder does not understand the SAM 3.1 key names and
    loads that file non-strictly (the interactive head stays random), so the
    checkpoint is first renamed into the SAM 3 layout with step 1 of
    ``convert_sam31_weights.py``; only video-memory keys remain missing. The
    reference enables a global bf16 autocast when built and casts to bf16
    internally, so it cannot run in fp32.
    """

    def __init__(self, checkpoint: str, score_floor: float):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        from convert_sam31_weights import load_original_checkpoint, sam31_to_sam3_layout

        state = load_original_checkpoint(checkpoint)
        if any(k.startswith("tracker.model.") for k in state):
            renamed = Path(tempfile.mkdtemp()) / "sam31_as_sam3_layout.pt"
            torch.save({"model": sam31_to_sam3_layout(state)}, renamed)
            checkpoint = str(renamed)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            enable_inst_interactivity=True,
            device="cuda",
        ).eval()
        self.processor = Sam3Processor(
            self.model, device="cuda", confidence_threshold=score_floor
        )

    @torch.no_grad()
    def text(self, image: Image.Image, prompt: str):
        state = self.processor.set_image(image)
        out = self.processor.set_text_prompt(prompt=prompt, state=state)
        masks = out["masks"].squeeze(1).cpu().numpy().astype(bool)
        return masks, out["scores"].float().cpu().numpy()

    @torch.no_grad()
    def visual(self, image: Image.Image, points=None, labels=None, bboxes=None):
        state = self.processor.set_image(image)
        kwargs = {"multimask_output": False}
        if points is not None:
            kwargs["point_coords"] = np.array([points], dtype=np.float32)
            kwargs["point_labels"] = np.array(labels)
        if bboxes is not None:
            kwargs["box"] = np.array(bboxes, dtype=np.float32)
        masks, scores, _ = self.model.predict_inst(state, **kwargs)
        return masks[0].astype(bool), float(scores[0])


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def load_libre(snapshot: str):
    """LibreSAM31 on a local snapshot copy, through the public factory."""
    from libreyolo import LibreSAM

    work = Path(tempfile.mkdtemp(prefix="sam31-parity-"))
    target = work / "weights" / "LibreSAM31large"
    target.parent.mkdir(parents=True)
    os.symlink(Path(snapshot).resolve(), target)
    if not (target / ".libreyolo_snapshot_complete").exists():
        (target / ".libreyolo_snapshot_complete").write_text("{}\n")
    cwd = os.getcwd()
    os.chdir(work)
    try:
        model = LibreSAM("sam3.1", device="cuda")
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
    return model


def libre_text(model, image: Image.Image, prompt: str):
    with torch.autocast("cuda", enabled=False):
        r = model.predict(image, text=prompt, conf=SCORE_FLOOR, max_det=300)
    if r.masks is None:
        return np.zeros((0, *image.size[::-1]), bool), np.zeros(0)
    return r.masks.data.cpu().numpy().astype(bool), r.boxes.conf.cpu().numpy()


def libre_visual(model, image: Image.Image, **prompt):
    with torch.autocast("cuda", enabled=False):
        r = model.predict(image, multimask=False, **prompt)
    if r.masks is None:  # empty mask: nothing to keep
        return np.zeros(image.size[::-1], bool), 0.0
    return r.masks.data.cpu().numpy().astype(bool)[0], float(r.boxes.conf[0])


def match_instances(ref_masks, ref_scores, our_masks, our_scores):
    """Greedy one-to-one matching by mask IoU, highest reference score first."""
    used: set[int] = set()
    pairs = []
    for i in np.argsort(-ref_scores):
        best, best_j = 0.0, -1
        for j in range(len(our_scores)):
            if j in used:
                continue
            iou = mask_iou(ref_masks[i], our_masks[j])
            if iou > best:
                best, best_j = iou, j
        if best_j >= 0 and best >= 0.5:
            used.add(best_j)
            pairs.append((best, abs(float(ref_scores[i]) - float(our_scores[best_j]))))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", default="sam31_parity.json")
    parser.add_argument(
        "--reference-precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="bf16 keeps the global autocast the reference enables when built",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    reference = ReferenceSAM31(args.checkpoint, score_floor=SCORE_FLOOR)
    ref_ctx = (
        torch.autocast("cuda", enabled=False)
        if args.reference_precision == "fp32"
        else torch.autocast("cuda", dtype=torch.bfloat16)
    )
    ours = load_libre(args.snapshot)

    report = {"reference_precision": args.reference_precision, "text": [], "visual": []}
    for path in args.images:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        for prompt in TEXT_PROMPTS:
            with ref_ctx:
                ref_masks, ref_scores = reference.text(image, prompt)
            our_masks, our_scores = libre_text(ours, image, prompt)
            pairs = match_instances(ref_masks, ref_scores, our_masks, our_scores)
            report["text"].append(
                {
                    "image": Path(path).name,
                    "prompt": prompt,
                    "reference_instances": int(len(ref_scores)),
                    "libre_instances": int(len(our_scores)),
                    "matched": len(pairs),
                    "min_mask_iou": min((p[0] for p in pairs), default=None),
                    "max_score_delta": max((p[1] for p in pairs), default=None),
                }
            )
        visual_prompts = {
            "point_center": {"points": [w // 2, h // 2], "labels": [1]},
            "point_left": {"points": [w // 4, h // 2], "labels": [1]},
            "box_center": {"bboxes": [w // 4, h // 4, 3 * w // 4, 3 * h // 4]},
        }
        for name, prompt in visual_prompts.items():
            with ref_ctx:
                ref_mask, ref_score = reference.visual(image, **prompt)
            our_mask, our_score = libre_visual(ours, image, **prompt)
            report["visual"].append(
                {
                    "image": Path(path).name,
                    "prompt": name,
                    "mask_iou": mask_iou(ref_mask, our_mask),
                    "iou_score_reference": ref_score,
                    "iou_score_libre": our_score,
                    "score_delta": abs(ref_score - our_score),
                }
            )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
