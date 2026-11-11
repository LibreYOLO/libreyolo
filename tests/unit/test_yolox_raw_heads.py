"""YOLOX raw-head export contract: graph cut, output windows and host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _YOLOXRawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import (
    decode_raw_heads,
    raw_head_output_names,
    unwindow_raw_heads,
)

pytestmark = pytest.mark.unit

_LAYOUT = "yolox_raw_heads_v1"


def _build(size: str = "n", nb_classes: int = 5, seed: int = 0):
    from libreyolo import LibreYOLOX

    torch.manual_seed(seed)
    model = LibreYOLOX(None, size=size, nb_classes=nb_classes, device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Non-trivial BatchNorm statistics; offsets, log sizes and logits
        # around zero so boxes and scores are both exercised.
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        head = net.head
        for conv in (*head.reg_preds, *head.obj_preds, *head.cls_preds):
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model, net


def _in_graph_decode(net, sample):
    net.head.export = True
    try:
        with torch.no_grad():
            return net(sample).numpy()  # (B, N, 5 + nc)
    finally:
        net.head.export = False


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _YOLOXRawHeadsWrapper(net, output_windows={}).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (8, 16, 32)
    assert wrapper.reg_max is None and wrapper.grid_cell_offset == 0.0
    assert wrapper.output_names == raw_head_output_names(_LAYOUT)
    assert wrapper.output_names[:3] == ["p3_reg", "p3_obj", "p3_cls"]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 4, "obj": 1, "cls": 5}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_raw_wrapper_rejects_other_tasks():
    _, net = _build()
    with pytest.raises(ValueError, match="detect only"):
        _YOLOXRawHeadsWrapper(net, "segment")


@pytest.mark.parametrize(("size", "hw"), [("n", (160, 160)), ("n", (192, 320)), ("s", (128, 128))])
@pytest.mark.parametrize("windowed", [False, True])
def test_host_decode_matches_in_graph_decode(size, hw, windowed):
    _, net = _build(size)
    windows = {"obj": (-8.0, 8.0), "cls": (-8.0, 8.0)} if windowed else {}
    wrapper = _YOLOXRawHeadsWrapper(net, output_windows=windows).eval()
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7)) * 255
    expected = _in_graph_decode(net, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        grid_cell_offset=wrapper.grid_cell_offset,
        output_windows=wrapper.output_windows,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    if windowed:
        # The windows clip only logits beyond +-8; the random head keeps
        # every objectness and class logit inside them.
        restored = unwindow_raw_heads(_LAYOUT, raw, windows)
        logits = [r for name, r in zip(wrapper.output_names, restored) if not name.endswith("reg")]
        assert max(float(np.abs(r).max()) for r in logits) < 8.0
    np.testing.assert_allclose(decoded[..., :4], expected[..., :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[..., 4:], expected[..., 4:], atol=1e-5)


def test_decode_rejects_wrong_output_count_and_strides():
    _, net = _build()
    wrapper = _YOLOXRawHeadsWrapper(net, output_windows={}).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 64, 64))]
    with pytest.raises(ValueError, match="expects 9 outputs"):
        decode_raw_heads(_LAYOUT, raw[:-1])
    with pytest.raises(ValueError, match="do not match strides"):
        decode_raw_heads(_LAYOUT, raw, strides=(8, 16, 16))
