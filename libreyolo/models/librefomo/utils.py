"""LibreFOMO preprocessing, decoding, and point metrics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int) -> tuple[np.ndarray, float]:
    pil_img = Image.fromarray(img_rgb_hwc).resize((input_size, input_size), Image.Resampling.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    arr = np.asarray(img.convert("RGB"))
    chw, ratio = preprocess_numpy(arr, input_size)
    return torch.from_numpy(chw).unsqueeze(0), img, img.size, ratio


def decode_points_from_logits(
    pred_logits: torch.Tensor,
    conf_threshold: float = 0.5,
    nms_radius: int = 1,
    max_points: int | None = None,
    class_offset: int = 1,
) -> list[torch.Tensor]:
    """Decode point peaks from ``(B, C, H, W)`` logits into grid-space x/y rows."""
    probs = F.softmax(pred_logits, dim=1)
    obj_probs, cls_idx = probs[:, class_offset:].max(dim=1)
    cls_idx = cls_idx + class_offset
    batch_points = []
    bsz, h, w = obj_probs.shape

    for b in range(bsz):
        prob = obj_probs[b]
        ys, xs = torch.where(prob > conf_threshold)
        if ys.numel() == 0:
            batch_points.append(torch.zeros((0, 4), dtype=pred_logits.dtype, device=pred_logits.device))
            continue

        scores = prob[ys, xs]
        order = torch.argsort(scores, descending=True)
        suppressed = torch.zeros((h, w), dtype=torch.bool, device=prob.device)
        kept = []
        for idx in order:
            y = int(ys[idx])
            x = int(xs[idx])
            if suppressed[y, x]:
                continue
            kept.append(torch.stack((xs[idx].float(), ys[idx].float(), cls_idx[b, y, x].float(), scores[idx].float())))
            y0 = max(0, y - nms_radius)
            y1 = min(h, y + nms_radius + 1)
            x0 = max(0, x - nms_radius)
            x1 = min(w, x + nms_radius + 1)
            suppressed[y0:y1, x0:x1] = True
            if max_points is not None and len(kept) >= max_points:
                break
        batch_points.append(torch.stack(kept) if kept else torch.zeros((0, 4), dtype=pred_logits.dtype, device=pred_logits.device))
    return batch_points


def postprocess(
    output: torch.Tensor,
    conf_thres: float,
    input_size: int,
    original_size: Tuple[int, int],
    nms_radius: int = 1,
    max_det: int = 300,
) -> dict:
    points = decode_points_from_logits(output, conf_threshold=conf_thres, nms_radius=nms_radius, max_points=max_det)[0]
    if points.numel() == 0:
        return {
            "points": torch.zeros((0, 2), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "classes": torch.zeros((0,), dtype=torch.float32),
            "num_detections": 0,
        }

    grid_h, grid_w = output.shape[-2:]
    orig_w, orig_h = original_size
    scale_x = orig_w / grid_w
    scale_y = orig_h / grid_h
    xy = points[:, :2].clone()
    xy[:, 0] = (xy[:, 0] + 0.5) * scale_x
    xy[:, 1] = (xy[:, 1] + 0.5) * scale_y
    return {
        "points": xy.float(),
        "scores": points[:, 3].float(),
        "classes": (points[:, 2] - 1).float(),
        "num_detections": int(points.shape[0]),
    }


def point_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
