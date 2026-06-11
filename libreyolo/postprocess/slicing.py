"""Per-image slicing of batched model outputs.

Shared by the validators and the batched predict path: a batched forward
output goes in, and one image's view comes out with the batch dimension
kept (``[i : i + 1]``), so the family's existing batch-1 postprocess
accepts it unchanged.
"""

from typing import Any

import torch


def slice_batch_outputs(output: Any, batch_idx: int) -> Any:
    """Extract predictions for a single image from batched model output.

    Tensors are sliced to ``[batch_idx : batch_idx + 1]`` (batch dim kept);
    dicts are sliced value-wise (one nested-dict level deep); lists and
    tuples recurse so nested list-of-tensor outputs (e.g. PICODET's
    per-level ``(List[cls_scores], List[bbox_preds])``) are sliced too.
    Without the recursion every per-image postprocess would get the full
    batch's tensors and ``[0]``-indexing would yield the first image's
    detections for every image in the batch. Non-tensor leaves (flags,
    scalars, ``None``) pass through unchanged.

    Contract note: list/tuple values INSIDE a dict pass through unsliced
    (matching the long-standing validation behavior this was extracted
    from). Today no family postprocess reads per-image data from such keys
    (yolo9 ``raw_outputs``, yolonas ``raw_predictions`` are unread); a new
    family that does must keep every read key a tensor or nested dict of
    tensors, or override its slicing.
    """
    if isinstance(output, dict):
        sliced = {}
        for key, value in output.items():
            if isinstance(value, dict):
                sliced[key] = {
                    k: v[batch_idx : batch_idx + 1]
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in value.items()
                }
            elif isinstance(value, torch.Tensor):
                sliced[key] = value[batch_idx : batch_idx + 1]
            else:
                sliced[key] = value
        return sliced
    elif isinstance(output, torch.Tensor):
        return output[batch_idx : batch_idx + 1]
    elif isinstance(output, (list, tuple)):
        return type(output)(slice_batch_outputs(p, batch_idx) for p in output)
    else:
        return output
