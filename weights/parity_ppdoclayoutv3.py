"""Parity check: LibrePPDocLayoutV3 vs the Apache-2.0 PyTorch reference.

The reference is ``PPDocLayoutV3ForObjectDetection`` from ``transformers``
(v5.22.1 used for the recorded numbers) loaded from the official
``PaddlePaddle/PP-DocLayoutV3_safetensors`` snapshot with eager attention.
Needs ``transformers`` installed; LibreYOLO itself never imports it.

Two gates, both in eval mode, fp32:

1. Raw head outputs on the SAME preprocessed tensor (each image plus one
   random tensor): ``pred_logits``, ``pred_boxes``, ``order_logits`` and the
   last-layer masks. Target: ``max_abs_diff == 0``.
2. Decoded regions on real pages: the reference image processor's
   ``post_process_object_detection`` vs ``model.predict``, both at
   ``conf=0.5`` and on the same tensor. Labels, scores, boxes and reading order
   must be identical. A second, informational comparison runs each side with
   its own preprocessing (the reference resizes with torchvision, LibreYOLO
   with cv2 like the original Paddle pipeline).

Usage::

    python weights/parity_ppdoclayoutv3.py \
        --checkpoint weights/LibrePPDocLayoutV3l.pt \
        --upstream path/to/PP-DocLayoutV3_safetensors \
        --images page1.jpg page2.png page3.png --device cuda
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream", required=True, help="HF snapshot directory")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    from _conversion_utils import add_repo_root_to_path

    add_repo_root_to_path()
    from PIL import Image
    from transformers import AutoImageProcessor, PPDocLayoutV3ForObjectDetection

    from libreyolo import LibreYOLO
    from libreyolo.models.ppdoclayoutv3.utils import preprocess_image

    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ref = PPDocLayoutV3ForObjectDetection.from_pretrained(
        args.upstream, attn_implementation="eager"
    ).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.upstream)
    model = LibreYOLO(args.checkpoint, device=str(device))
    net = model.model.eval()

    failures = 0
    inputs = []
    for path in args.images:
        tensor, _, size, _ = preprocess_image(path, input_size=800)
        inputs.append((path, tensor, size))
    torch.manual_seed(0)
    inputs.append(("random", torch.rand(1, 3, 800, 800), (800, 800)))

    print("== Gate 1: raw head outputs, same input tensor ==")
    for name, tensor, _ in inputs:
        x = tensor.to(device)
        with torch.no_grad():
            r = ref(pixel_values=x)
            o = net(x, return_masks=True)
        diffs = {
            "pred_logits": _max_diff(r.logits, o["pred_logits"]),
            "pred_boxes": _max_diff(r.pred_boxes, o["pred_boxes"]),
            "order_logits": _max_diff(r.order_logits, o["order_logits"]),
            "masks": _max_diff(r.out_masks, o["pred_masks"]),
        }
        ok = all(v == 0.0 for v in diffs.values())
        failures += not ok
        print(f"{'OK ' if ok else 'BAD'} {name}: " + ", ".join(f"{k}={v:.3g}" for k, v in diffs.items()))

    print(f"== Gate 2: decoded regions at conf={args.conf} ==")
    for name, tensor, size in inputs[:-1]:
        width, height = size
        x = tensor.to(device)
        with torch.no_grad():
            r = ref(pixel_values=x)
        upstream = processor.post_process_object_detection(
            r, threshold=args.conf, target_sizes=[(height, width)]
        )[0]
        ours = model(name, conf=args.conf)
        up_labels = upstream["labels"].cpu().numpy()
        up_scores = upstream["scores"].cpu().numpy()
        up_boxes = upstream["boxes"].cpu().numpy()
        same = (
            len(up_labels) == len(ours)
            and np.array_equal(up_labels, ours.boxes.cls.cpu().numpy().astype(np.int64))
            and np.array_equal(
                np.asarray(ours.reading_order), np.arange(1, len(up_labels) + 1)
            )
        )
        score_diff = box_diff = float("nan")
        if same and len(up_labels):
            score_diff = float(np.abs(up_scores - ours.boxes.conf.cpu().numpy()).max())
            box_diff = float(np.abs(up_boxes - ours.boxes.xyxy.cpu().numpy()).max())
            same = score_diff == 0.0 and box_diff == 0.0
        failures += not same
        print(
            f"{'OK ' if same else 'BAD'} {name}: regions={len(up_labels)}/{len(ours)} "
            f"max_score_diff={score_diff:.3g} max_box_diff={box_diff:.3g}"
        )

        # Informational: each side with its own preprocessing.
        image = Image.open(name).convert("RGB")
        batch = processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            r_own = ref(**batch)
        up_own = processor.post_process_object_detection(
            r_own, threshold=args.conf, target_sizes=[(height, width)]
        )[0]
        own_labels = up_own["labels"].cpu().numpy()
        match = len(own_labels) == len(ours) and np.array_equal(
            own_labels, ours.boxes.cls.cpu().numpy().astype(np.int64)
        )
        detail = ""
        if match and len(own_labels):
            detail = " max_box_diff_px={:.3g}".format(
                float(np.abs(up_own["boxes"].cpu().numpy() - ours.boxes.xyxy.cpu().numpy()).max())
            )
        print(
            f"    own preprocessing: regions={len(own_labels)}/{len(ours)} "
            f"same labels+order={match}{detail}"
        )

    if failures:
        raise SystemExit(f"{failures} parity check(s) failed")
    print("All parity checks passed.")


if __name__ == "__main__":
    main()
