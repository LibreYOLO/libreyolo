import numpy as np
import pytest
import torch

from libreyolo.common.utils import draw_boxes, preprocess_image

pytestmark = pytest.mark.unit


def test_preprocess_image_produces_tensor_and_preserves_original_size():
    img = np.zeros((8, 6, 3), dtype=np.uint8)

    tensor, original_img, original_size = preprocess_image(img, input_size=32)

    assert tensor.shape == (1, 3, 32, 32)
    assert original_size == (6, 8)
    assert torch.is_floating_point(tensor)
    assert tensor.min() >= 0 and tensor.max() <= 1
    annotated = draw_boxes(original_img, boxes=[[0, 0, 1, 1]], scores=[0.9], classes=[0])
    assert annotated.size == original_img.size
