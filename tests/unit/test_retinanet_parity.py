"""Exact raw-head parity against torchvision's official RetinaNet weights.

Set ``LIBREYOLO_RETINANET_CHECKPOINT_DIR`` to a directory containing the two
official checkpoint filenames, or set ``LIBREYOLO_RETINANET_ACCEPTANCE=1`` to
let the pinned torchvision weight enums download them. The implementation is
BSD-3-Clause; checkpoint redistribution terms are documented separately.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import resnet50
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import (
    RetinaNet,
    RetinaNetHead,
    RetinaNet_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import pil_to_tensor

from libreyolo.models.retinanet.nn import LibreRetinaNetModel
from libreyolo.postprocess.retinanet import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80


CHECKPOINT_DIR = os.environ.get("LIBREYOLO_RETINANET_CHECKPOINT_DIR")
ACCEPTANCE = os.environ.get("LIBREYOLO_RETINANET_ACCEPTANCE") == "1"

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.retinanet,
    pytest.mark.skipif(
        not CHECKPOINT_DIR and not ACCEPTANCE,
        reason=(
            "set LIBREYOLO_RETINANET_CHECKPOINT_DIR or LIBREYOLO_RETINANET_ACCEPTANCE=1"
        ),
    ),
]


CASES = {
    "r50": (
        "retinanet_resnet50_fpn_coco-eeacb38b.pth",
        RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    ),
    "r50v2": (
        "retinanet_resnet50_fpn_v2_coco-5905b1c5.pth",
        RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
}


def _build_reference(size: str) -> RetinaNet:
    v2 = size == "r50v2"
    backbone = resnet50(
        weights=None,
        norm_layer=torch.nn.BatchNorm2d if v2 else FrozenBatchNorm2d,
    )
    backbone = _resnet_fpn_extractor(
        backbone,
        trainable_layers=5,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(2048 if v2 else 256, 256),
    )
    if v2:
        head = RetinaNetHead(
            256,
            9,
            91,
            norm_layer=partial(torch.nn.GroupNorm, 32),
        )
        model = RetinaNet(backbone, 91, head=head)
    else:
        model = RetinaNet(backbone, 91)
        overwrite_eps(model, 0.0)
    return model.eval()


def _load_state(size: str) -> dict:
    filename, weights = CASES[size]
    if CHECKPOINT_DIR:
        path = Path(CHECKPOINT_DIR) / filename
        if not path.is_file():
            pytest.fail(f"missing official checkpoint: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)
    return weights.get_state_dict(progress=True, check_hash=True)


def _image_tensor() -> torch.Tensor:
    path = Path(__file__).parents[2] / "tests" / "data" / "parkour.jpg"
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float() / 255.0


def _maximum_difference(expected: dict, actual: dict) -> float:
    maximum = 0.0
    assert expected.keys() == actual.keys()
    for key in expected:
        assert expected[key].shape == actual[key].shape
        if expected[key].numel():
            maximum = max(
                maximum,
                float((expected[key] - actual[key]).abs().max().item()),
            )
    return maximum


@pytest.mark.parametrize("size", ["r50", "r50v2"])
def test_official_raw_head_parity(size):
    state_dict = _load_state(size)
    reference = _build_reference(size)
    ours = LibreRetinaNetModel(size=size, num_classes=91).eval()
    reference.load_state_dict(state_dict, strict=True)
    ours.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference.to(device)
    ours.to(device)
    image = _image_tensor().to(device)

    # One upstream transform owns resize, normalization, and padding. Both
    # graphs then receive the identical pre-resized tensor, so this comparison
    # measures model math rather than image-transform drift.
    transformed, _ = reference.transform([image], None)
    input_tensor = transformed.tensors
    with torch.inference_mode():
        expected_features = list(reference.backbone(input_tensor).values())
        expected_heads = reference.head(expected_features)
        actual_heads, actual_features = ours.forward_head(input_tensor)
        expected_detections = reference([image])[0]
        decoded_output = ours(input_tensor)

    feature_diff = max(
        float((expected - actual).abs().max().item())
        for expected, actual in zip(expected_features, actual_features)
    )
    head_diff = _maximum_difference(expected_heads, actual_heads)
    print(
        f"size={size} device={device.type} "
        f"feature_max_abs_diff={feature_diff} head_max_abs_diff={head_diff}"
    )
    assert feature_diff == 0.0
    assert head_diff == 0.0

    original_size = (int(image.shape[-1]), int(image.shape[-2]))
    actual_detections = postprocess(
        decoded_output,
        conf_thres=0.05,
        iou_thres=0.5,
        original_size=original_size,
        input_size=800,
        max_det=300,
    )
    mapped_labels = torch.full_like(expected_detections["labels"], -1)
    for source, target in COCO91_TO_COCO80.items():
        mapped_labels[expected_detections["labels"] == source] = target
    valid = mapped_labels >= 0
    expected_boxes = expected_detections["boxes"][valid].cpu().numpy()
    expected_scores = expected_detections["scores"][valid].cpu().numpy()
    expected_classes = mapped_labels[valid].cpu().numpy()
    np.testing.assert_array_equal(actual_detections["boxes"], expected_boxes)
    np.testing.assert_array_equal(actual_detections["scores"], expected_scores)
    np.testing.assert_array_equal(actual_detections["classes"], expected_classes)
    print(
        f"size={size} final_detections={len(expected_boxes)} "
        "box_max_abs_diff=0.0 score_max_abs_diff=0.0 classes_equal=True"
    )
