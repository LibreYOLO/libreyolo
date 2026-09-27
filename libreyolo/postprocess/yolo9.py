"""YOLO9 postprocessing (detect / pose / OBB).

Moved verbatim from ``libreyolo/models/yolo9/utils.py``, which re-exports
everything here for backward compatibility.
"""

from __future__ import annotations

from . import obb_ops
from ..utils.lazy import lazy_module

from typing import Dict, Tuple, Union


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")


_YOLO9_MAX_NMS_CANDIDATES = 30000
_YOLO9_OBB_MAX_NMS_CANDIDATES = 1200
_YOLO9_OBB_PREFILTER_CANDIDATES = _YOLO9_OBB_MAX_NMS_CANDIDATES

ImageSize = Union[int, Tuple[int, int]]


def _input_size_hw(input_size: ImageSize) -> Tuple[int, int]:
    if isinstance(input_size, (tuple, list)):
        if len(input_size) != 2:
            raise ValueError(f"input_size must be int or (height, width), got {input_size}")
        h, w = int(input_size[0]), int(input_size[1])
    else:
        h = w = int(input_size)
    if h <= 0 or w <= 0:
        raise ValueError(f"input_size values must be positive, got {(h, w)}")
    return h, w


def _nms_keep_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_thres: float,
    max_det: int,
) -> torch.Tensor:
    # Imported here rather than at module scope: this module must stay
    # importable without torchvision for the torch-free ONNX path, which
    # reaches NMS through _batched_nms_numpy in backends/base.py instead.
    from torchvision.ops import batched_nms

    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Drop non-finite rows — batched_nms is undefined on NaN/Inf inputs.
    finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        valid_indices = torch.where(finite_mask)[0]
        if len(valid_indices) == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        boxes = boxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]
    else:
        valid_indices = None

    # Shift to non-negative coords — batched_nms's class-offset trick uses
    # (boxes.max() + 1) and only separates classes when all coords are
    # non-negative. Translation-invariant for IoU.
    nms_boxes = boxes - boxes.min().clamp(max=0)
    keep = batched_nms(nms_boxes, scores, class_ids, iou_thres)
    if len(keep) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    if len(keep) > max_det:
        _, order = torch.topk(scores[keep], max_det)
        keep = keep[order]

    # Map back to original indices when we filtered non-finite rows above.
    if valid_indices is not None:
        keep = valid_indices[keep]
    return keep


# Rotated-box geometry and NMS are task-level, not family-level: they moved to
# ``libreyolo.postprocess.obb_ops`` so every OBB family shares one exact
# implementation. The private aliases stay for backward compatibility.
_xywhr_to_corners = obb_ops.xywhr_to_corners
_xywhr_to_xyxy = obb_ops.xywhr_to_xyxy
_rotated_nms_keep_indices = obb_ops.rotated_nms_keep_indices


def _obb_prefilter_keep_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    max_det: int,
) -> torch.Tensor:
    """Cheaply bound candidates before exact rotated NMS without suppressing boxes."""
    if scores.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=scores.device)

    limit = max(max_det, _YOLO9_OBB_PREFILTER_CANDIDATES)
    if scores.numel() <= limit:
        return torch.arange(scores.numel(), dtype=torch.long, device=scores.device)

    del boxes, class_ids
    return torch.topk(scores, min(limit, scores.numel())).indices


def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: ImageSize = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
    letterbox_pad: str | None = None,
) -> Dict:
    """
    Postprocess YOLOv9 model outputs to get final detections.

    Args:
        output: Model output dictionary with 'predictions' key
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size as int or (height, width).
        original_size: Original image size (width, height) for scaling
        max_det: Maximum number of detections to return (default: 300)

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    input_h, input_w = _input_size_hw(input_size)
    predictions = output["predictions"]  # (batch, 4+nc, total_anchors)
    is_obb = bool(output.get("obb", False))

    if predictions.dim() == 3:
        pred = predictions[0]  # (4+nc, total_anchors)
    else:
        pred = predictions

    # Transpose to (total_anchors, 4+nc)
    pred = pred.transpose(0, 1)

    boxes_input = pred[:, :4]  # xyxy format in model input pixels
    if is_obb:
        angles_input = pred[:, 4]
        scores = pred[:, 5:]  # class scores (already sigmoid applied in model)
    else:
        angles_input = None
        scores = pred[:, 4:]  # class scores (already sigmoid applied in model)

    # Detection uses multi-label selection: every class whose score exceeds
    # conf_thres yields a detection for that anchor, matching the port source
    # MultimediaTechLab/YOLO (yolo/utils/bounding_box_utils.py::bbox_nms, which
    # selects candidates via torch.where(cls_dist > min_confidence)). At the low
    # conf thresholds used for COCO evaluation this recovers ~0.7 mAP over
    # best-class-only selection.
    keypoints_all = output.get("keypoints")
    if keypoints_all is not None:
        keypoints_all = keypoints_all[0] if keypoints_all.dim() == 4 else keypoints_all

    if is_obb:
        max_scores, class_ids = torch.max(scores, dim=1)
        mask = max_scores > conf_thres
        if not mask.any():
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "obb": [],
                "num_detections": 0,
            }
        boxes_input = boxes_input[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        angles_input = angles_input[mask]

        wh = (boxes_input[:, 2:4] - boxes_input[:, 0:2]).clamp_min(0)
        centers = (boxes_input[:, 0:2] + boxes_input[:, 2:4]) / 2
        xywhr = torch.cat((centers, wh, angles_input[:, None]), dim=1)

        if original_size is not None:
            if letterbox:
                from ..preprocess.letterbox import letterbox_geometry

                orig_w, orig_h = original_size
                # xywhr is cx,cy,w,h,angle. Undo pad on the center only;
                # scale w/h by ratio. Going through the enclosing AABB would
                # destroy width/height whenever angle != 0.
                ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(
                    orig_h, orig_w, input_h, input_w, letterbox_pad
                )
                xywhr[:, 0] = (xywhr[:, 0] - pad_left) / ratio
                xywhr[:, 1] = (xywhr[:, 1] - pad_top) / ratio
                xywhr[:, 2] = xywhr[:, 2] / ratio
                xywhr[:, 3] = xywhr[:, 3] / ratio
            else:
                scale_x = original_size[0] / input_w
                scale_y = original_size[1] / input_h
                xywhr[:, [0, 2]] *= scale_x
                xywhr[:, [1, 3]] *= scale_y
            xywhr[:, 0].clamp_(0, original_size[0])
            xywhr[:, 1].clamp_(0, original_size[1])

        boxes = _xywhr_to_xyxy(xywhr)
        if original_size is not None:
            boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
            boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])

        widths = xywhr[:, 2]
        heights = xywhr[:, 3]
        valid = (widths > 0) & (heights > 0)
        if not valid.any():
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "obb": [],
                "num_detections": 0,
            }
        if not valid.all():
            xywhr = xywhr[valid]
            boxes = boxes[valid]
            max_scores = max_scores[valid]
            class_ids = class_ids[valid]

        max_nms = max(max_det, _YOLO9_OBB_MAX_NMS_CANDIDATES)
        if max_scores.numel() > max_nms:
            top = torch.topk(max_scores, max_nms).indices
            xywhr = xywhr[top]
            boxes = boxes[top]
            max_scores = max_scores[top]
            class_ids = class_ids[top]

        pre_keep = _obb_prefilter_keep_indices(boxes, max_scores, class_ids, max_det)
        if pre_keep.numel() != max_scores.numel():
            xywhr = xywhr[pre_keep]
            boxes = boxes[pre_keep]
            max_scores = max_scores[pre_keep]
            class_ids = class_ids[pre_keep]

        keep = _rotated_nms_keep_indices(xywhr, max_scores, class_ids, iou_thres, max_det)
        if len(keep) == 0:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "obb": [],
                "num_detections": 0,
            }
        boxes = boxes[keep]
        scores_out = max_scores[keep]
        classes_out = class_ids[keep]
        obb_out = torch.cat(
            (xywhr[keep], scores_out[:, None], classes_out[:, None].float()),
            dim=1,
        )
        return {
            "boxes": boxes.detach().cpu().numpy().tolist(),
            "scores": scores_out.detach().cpu().numpy().tolist(),
            "classes": classes_out.detach().cpu().numpy().tolist(),
            "obb": obb_out.detach().cpu().numpy().tolist(),
            "num_detections": len(boxes),
        }

    anchor_idx, class_ids = (scores > conf_thres).nonzero(as_tuple=True)
    if anchor_idx.numel() == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
    boxes_input = boxes_input[anchor_idx]
    keypoints = keypoints_all[anchor_idx].clone() if keypoints_all is not None else None
    max_scores = scores[anchor_idx, class_ids]
    max_nms = max(max_det, _YOLO9_MAX_NMS_CANDIDATES)
    if max_scores.numel() > max_nms:
        keep = torch.topk(max_scores, max_nms).indices
        boxes_input = boxes_input[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        max_scores = max_scores[keep]
        class_ids = class_ids[keep]
    boxes = boxes_input.clone()

    if original_size is not None:
        if letterbox:
            from ..preprocess.letterbox import letterbox_geometry, unletterbox_xyxy

            orig_w, orig_h = original_size
            boxes = unletterbox_xyxy(
                boxes, orig_w, orig_h, input_h, input_w, pad=letterbox_pad
            )
            if keypoints is not None:
                _ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(
                    orig_h, orig_w, input_h, input_w, letterbox_pad
                )
                keypoints[..., 0] = (keypoints[..., 0] - pad_left) / _ratio
                keypoints[..., 1] = (keypoints[..., 1] - pad_top) / _ratio
        else:
            scale_x = original_size[0] / input_w
            scale_y = original_size[1] / input_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            if keypoints is not None:
                keypoints[..., 0] *= scale_x
                keypoints[..., 1] *= scale_y

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])
        if keypoints is not None:
            keypoints[..., 0] = torch.clamp(keypoints[..., 0], 0, original_size[0])
            keypoints[..., 1] = torch.clamp(keypoints[..., 1], 0, original_size[1])

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 0) & (heights > 0)
    if not valid.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
    if not valid.all():
        boxes = boxes[valid]
        boxes_input = boxes_input[valid]
        max_scores = max_scores[valid]
        class_ids = class_ids[valid]
        if keypoints is not None:
            keypoints = keypoints[valid]

    keep = _nms_keep_indices(boxes, max_scores, class_ids, iou_thres, max_det)
    if len(keep) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[keep]
    scores_out = max_scores[keep]
    classes_out = class_ids[keep]
    keypoints_out = keypoints[keep] if keypoints is not None else None

    result = {
        "boxes": boxes.detach().cpu().numpy().tolist(),
        "scores": scores_out.detach().cpu().numpy().tolist(),
        "classes": classes_out.detach().cpu().numpy().tolist(),
        "num_detections": len(boxes),
    }
    if keypoints_out is not None:
        result["keypoints"] = keypoints_out.detach().cpu()

    return result
