"""RF-DETR PyTorch boxes are clipped to the image, like every exported backend."""

from __future__ import annotations

import pytest
import torch

from libreyolo.postprocess.rfdetr import postprocess

pytestmark = pytest.mark.unit


def test_boxes_past_the_image_edge_are_clipped():
    logits = torch.full((1, 2, 3), -10.0)
    logits[0, 0, 1] = 5.0
    logits[0, 1, 2] = 4.0
    # cxcywh in [0, 1]: one box sticks out on the left and top, one on the
    # right and bottom.
    boxes = torch.tensor([[[0.05, 0.10, 0.20, 0.40], [0.95, 0.90, 0.20, 0.40]]])
    out = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes},
        target_sizes=torch.tensor([[200, 100]]),  # (height, width)
        num_select=2,
    )[0]

    by_label = {int(label): box.tolist() for label, box in zip(out["labels"], out["boxes"])}
    assert by_label[1] == pytest.approx([0.0, 0.0, 15.0, 60.0])
    assert by_label[2] == pytest.approx([85.0, 140.0, 100.0, 200.0])
