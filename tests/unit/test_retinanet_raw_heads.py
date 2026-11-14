"""RetinaNet raw-head export contract: host anchors and box decode vs the in-graph decode."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _RetinaNetRawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import decode_raw_heads, raw_head_output_names

pytestmark = pytest.mark.unit

_LAYOUT = "retinanet_raw_heads_v1"


def _net(num_classes: int, seed: int = 0):
    from libreyolo.models.retinanet.nn import LibreRetinaNetModel

    torch.manual_seed(seed)
    net = LibreRetinaNetModel("r50", num_classes).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():  # spread deltas and logits so boxes and scores both move
        head = net.head
        for conv in (head.regression_head.bbox_reg, head.classification_head.cls_logits):
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.01)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator) * 0.5)
    return net


def test_raw_wrapper_output_contract():
    wrapper = _RetinaNetRawHeadsWrapper(_net(5)).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 256, 256))
    assert wrapper.output_layout == _LAYOUT and wrapper.strides == (8, 16, 32, 64, 128)
    assert wrapper.output_names == raw_head_output_names(_LAYOUT, (8, 16, 32, 64, 128))
    assert wrapper.class_index_map is None and len(wrapper.anchor_sizes) == 5
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 9 * 4, "cls": 9 * 5}[part]
        assert tuple(output.shape) == (1, channels, -(-256 // stride), -(-256 // stride))


@pytest.mark.parametrize(("num_classes", "size"), [(5, 256), (91, 320)])
def test_host_decode_matches_in_graph_decode(num_classes, size):
    net = _net(num_classes)
    wrapper = _RetinaNetRawHeadsWrapper(net).eval()
    sample = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = net(sample).numpy()  # (B, N, 4 + classes), COCO-91 -> 80 mapped
        raw = [t.numpy() for t in wrapper(sample)]
    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        anchor_sizes=wrapper.anchor_sizes,
        aspect_ratios=wrapper.aspect_ratios,
        class_index_map=wrapper.class_index_map,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    np.testing.assert_allclose(decoded[..., :4], expected[..., :4], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(decoded[..., 4:], expected[..., 4:], atol=1e-5)


def test_fixed_canvas_preprocessing_fits_the_image_into_one_square():
    from libreyolo.models.retinanet.utils import preprocess_numpy
    from libreyolo.postprocess.retinanet import resize_geometry

    image = np.random.default_rng(0).integers(0, 256, (240, 400, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(image, 320, fixed_canvas=True)
    assert chw.shape == (3, 320, 320) and ratio == pytest.approx(0.8)
    resized_h, resized_w, scale_x, scale_y = resize_geometry((400, 240), 320, fixed_canvas=True)
    assert (resized_h, resized_w) == (192, 320)
    dynamic, _ = preprocess_numpy(image, 320)
    assert dynamic.shape[1:] != (320, 320)  # upstream min/max resize stays the default
