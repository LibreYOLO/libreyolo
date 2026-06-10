"""RF-DETR postprocessing (DETR-style top-K, no NMS; multi-task heads).

Behavior matches upstream RF-DETR (https://github.com/roboflow/rf-detr) so
weights load and produce numerically equivalent detections.

Moved verbatim from ``libreyolo/models/rfdetr/utils.py``, which re-exports it
for backward compatibility.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict

from ..data.obb import scale_xywhr
from ..utils.general import cxcywh_to_xyxy


def postprocess(
    outputs: Dict[str, torch.Tensor], target_sizes: torch.Tensor, num_select: int = 300
) -> List[Dict[str, torch.Tensor]]:
    """
    Postprocess RF-DETR outputs to get final detections.

    This matches the original rfdetr PostProcess class exactly:
    1. Apply sigmoid to logits
    2. Select top-K scores across all (queries × classes)
    3. Convert boxes from cxcywh to xyxy
    4. Scale boxes to original image coordinates

    No NMS is applied - just top-K selection (same as original).

    Args:
        outputs: Model output dictionary with 'pred_logits' and 'pred_boxes'
        target_sizes: Tensor of shape (batch_size, 2) with (height, width) for each image
        num_select: Number of top detections to select (default: 300)

    Returns:
        List of dictionaries, one per image, each containing:
            - scores: Tensor of shape (num_select,) with confidence scores
            - labels: Tensor of shape (num_select,) with class IDs
            - boxes: Tensor of shape (num_select, 4) in xyxy format
    """
    out_logits = outputs["pred_logits"]  # (B, num_queries, num_classes)
    out_bbox = outputs["pred_boxes"]  # (B, num_queries, 4) in cxcywh [0, 1]
    out_masks = outputs.get("pred_masks")  # (B, num_queries, Hm, Wm) or None
    out_keypoints = outputs.get("pred_keypoints")  # (B, num_queries, K, 3) or None
    out_angles = outputs.get("pred_angles")  # (B, num_queries, 1) or None

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()

    # Top-K across all (queries × classes)
    batch_size = out_logits.shape[0]
    num_classes = out_logits.shape[2]

    topk_values, topk_indexes = torch.topk(prob.view(batch_size, -1), num_select, dim=1)

    scores = topk_values

    topk_boxes = topk_indexes // num_classes  # Which query
    labels = topk_indexes % num_classes  # Which class

    boxes = cxcywh_to_xyxy(out_bbox)

    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    obb = None
    if out_angles is not None:
        obb_cxcywh = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        obb_angles = torch.gather(out_angles, 1, topk_boxes.unsqueeze(-1)).squeeze(-1)

    # Scale from relative [0, 1] to absolute [0, height/width] coordinates.
    # RF-DETR resizes rectangular images directly to a square canvas, so OBBs
    # must be transformed through corners before refitting xywhr.
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    if out_angles is not None:
        obb_rel = torch.cat((obb_cxcywh, obb_angles.unsqueeze(-1)), dim=-1)
        obb_rows = []
        for batch_idx in range(batch_size):
            scaled = scale_xywhr(
                obb_rel[batch_idx].detach().cpu().numpy(),
                float(img_w[batch_idx].detach().cpu().item()),
                float(img_h[batch_idx].detach().cpu().item()),
            )
            obb_rows.append(
                torch.as_tensor(
                    scaled,
                    dtype=obb_cxcywh.dtype,
                    device=obb_cxcywh.device,
                )
            )
        obb_xywhr = torch.stack(obb_rows, dim=0)
        obb = torch.cat(
            (
                obb_xywhr,
                scores.unsqueeze(-1),
                labels.to(dtype=obb_xywhr.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    results = []
    for i in range(batch_size):
        res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
        if obb is not None:
            res_i["obb"] = obb[i]

        if out_masks is not None:
            # Gather masks for top-K queries
            k_idx = topk_boxes[i]
            masks_i = torch.gather(
                out_masks[i],
                0,
                k_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
            )  # (K, Hm, Wm)

            # Resize to original image size
            h, w = target_sizes[i].tolist()
            masks_i = F.interpolate(
                masks_i.unsqueeze(1),
                size=(int(h), int(w)),
                mode="bilinear",
                align_corners=False,
            )  # (K, 1, H, W)

            # Threshold at 0.0 in logit space (= 0.5 probability)
            res_i["masks"] = (masks_i[:, 0] > 0.0).bool()  # (K, H, W)

        if out_keypoints is not None:
            k_idx = topk_boxes[i]
            keypoints_i = out_keypoints[i][k_idx].clone()
            h, w = target_sizes[i].tolist()
            keypoints_i[..., 0] = keypoints_i[..., 0] * float(w)
            keypoints_i[..., 1] = keypoints_i[..., 1] * float(h)
            keypoints_i[..., 2] = keypoints_i[..., 2].sigmoid()
            res_i["keypoints"] = keypoints_i

        results.append(res_i)

    return results
