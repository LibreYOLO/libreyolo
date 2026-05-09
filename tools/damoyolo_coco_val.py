"""Run COCO val 2017 with a ported DAMO-YOLO checkpoint and report mAP.

Usage:
    python tools/damoyolo_coco_val.py \\
        --weights /path/to/damoyolo_tinynasL20_T_420.pth \\
        --coco /Users/xuban.ceccon/datasets/coco \\
        --size t \\
        --device cuda

Targets reproduction of upstream's stated mAP (DAMO-YOLO-T = 42.0,
DAMO-YOLO-T* = 43.6) on COCO val 2017.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

# Make the repo importable when running from the worktree.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libreyolo.models.damoyolo.builder import build_damoyolo  # noqa: E402
from libreyolo.models.damoyolo.utils import (  # noqa: E402
    postprocess_predictions,
    preprocess_image,
)
from libreyolo.validation.coco_evaluator import COCOEvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("damoyolo-val")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--coco", required=True, type=Path, help="COCO root containing annotations/ and images/")
    p.add_argument("--split", default="val2017")
    p.add_argument("--size", default="t", choices=["t", "s"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input-size", type=int, default=640)
    p.add_argument("--conf-thres", type=float, default=0.05)
    p.add_argument("--iou-thres", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all)")
    p.add_argument("--batch", type=int, default=1, help="Batch size for inference")
    p.add_argument("--save-json", type=Path, default=None)
    return p.parse_args()


def load_upstream_checkpoint(model: torch.nn.Module, path: Path, device: str) -> None:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    # upstream checkpoints sometimes have a "module." prefix from DDP
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log.warning("Missing keys: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        log.warning("Unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])
    if not missing and not unexpected:
        log.info("State dict loaded cleanly (%d tensors)", len(sd))


def main() -> int:
    args = parse_args()

    # COCO API
    from pycocotools.coco import COCO

    ann_file = args.coco / "annotations" / f"instances_{args.split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(ann_file)
    log.info("Loading COCO annotations: %s", ann_file)
    coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    log.info("Evaluating on %d images", len(img_ids))

    # Build model + load weights
    log.info("Building DAMO-YOLO size=%s", args.size)
    model = build_damoyolo(size=args.size, num_classes=80)
    load_upstream_checkpoint(model, args.weights, device=args.device)
    model.to(args.device).eval()
    log.info("Switching RepConv branches to deploy mode")
    model.switch_to_deploy()

    # Build label_id mapping (contiguous 0-79 → COCO category_id 1-90)
    cat_ids = sorted(coco_gt.getCatIds())  # COCO uses 1-90 with gaps
    if len(cat_ids) != 80:
        log.warning("Got %d COCO categories (expected 80)", len(cat_ids))
    label_to_category_id = {i: cid for i, cid in enumerate(cat_ids)}

    img_dir = args.coco / "images" / args.split
    if not img_dir.exists():
        # Fallback to the layout the README dump shows
        img_dir = args.coco / args.split
    if not img_dir.exists():
        raise FileNotFoundError(f"COCO image dir not found near {args.coco}")

    evaluator = COCOEvaluator(coco_gt, iou_type="bbox", label_to_category_id=label_to_category_id)
    input_size = (args.input_size, args.input_size)
    t0 = time.time()
    with torch.no_grad():
        # Process in mini-batches: stretch resize → fixed (H, W) means we
        # can stack tensors directly. Original sizes are tracked per image
        # so postprocess can rescale back.
        with tqdm(total=len(img_ids), desc="val") as pbar:
            for start in range(0, len(img_ids), args.batch):
                batch_ids = img_ids[start : start + args.batch]
                tensors = []
                orig_sizes = []
                for img_id in batch_ids:
                    info = coco_gt.loadImgs(img_id)[0]
                    img_path = img_dir / info["file_name"]
                    t, (ow, oh) = preprocess_image(img_path, input_size=input_size)
                    tensors.append(t)
                    orig_sizes.append((ow, oh))
                batch = torch.stack(tensors, dim=0).to(args.device)
                cls_scores, boxes = model(batch)
                preds_list = postprocess_predictions(
                    cls_scores,
                    boxes,
                    orig_sizes=orig_sizes,
                    input_size=input_size,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    max_det=args.max_det,
                )
                for img_id, preds in zip(batch_ids, preds_list):
                    evaluator.update(preds, image_id=img_id)
                pbar.update(len(batch_ids))
    dt = time.time() - t0
    log.info("Inference complete (%.1fs, %.1f img/s)", dt, len(img_ids) / max(dt, 1e-6))

    log.info("Computing COCO metrics")
    metrics = evaluator.compute(save_json=str(args.save_json) if args.save_json else None)
    log.info("---- DAMO-YOLO size=%s on %s ----", args.size, args.split)
    for k in ("mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100"):
        log.info("  %-12s %.4f", k, metrics[k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
