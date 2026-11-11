"""Darknet (YOLOv3/v4) raw-head export contract: split head convs, output windows and host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _darknet_raw_heads_wrapper
from libreyolo.postprocess.yolonas_raw import (
    decode_raw_heads,
    raw_head_output_names,
    unwindow_raw_heads,
)

pytestmark = pytest.mark.unit

_LAYOUT = "darknet_raw_heads_v1"


def _build(size: str = "t", nb_classes: int = 5, seed: int = 0, family: str = "yolo4"):
    from libreyolo import LibreYOLO3, LibreYOLO4

    torch.manual_seed(seed)
    cls = {"yolo3": LibreYOLO3, "yolo4": LibreYOLO4}[family]
    model = cls(None, size=size, nb_classes=nb_classes, device="cpu")
    holder = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for spec in holder.net.detection_specs():
            conv = holder.net.layers[spec.layer_index - 1].conv
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model, holder


def _in_graph_decode(holder, sample):
    from libreyolo.models.darknet.export import DarknetExportWrapper

    with torch.no_grad():
        return DarknetExportWrapper(holder).eval()(sample).numpy()  # (B, 4 + nc, N)


def test_raw_wrapper_output_contract():
    _, holder = _build(nb_classes=5)
    wrapper = _darknet_raw_heads_wrapper(holder).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (32, 16)  # yolov4-tiny heads, in cfg order
    assert wrapper.reg_max is None and wrapper.grid_cell_offset == 0.0
    assert wrapper.output_names == raw_head_output_names(_LAYOUT, (32, 16))
    assert wrapper.output_names[:3] == ["p5_reg", "p5_obj", "p5_cls"]
    assert [len(level) for level in wrapper.anchors] == [3, 3]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 12, "obj": 3, "cls": 15}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_raw_wrapper_rejects_other_tasks():
    _, holder = _build()
    with pytest.raises(ValueError, match="detect only"):
        _darknet_raw_heads_wrapper(holder, "segment")


@pytest.mark.parametrize(
    ("family", "size", "hw"),
    [("yolo4", "t", (160, 160)), ("yolo4", "t", (192, 320)), ("yolo4", "b", (128, 128)),
     ("yolo3", "t", (160, 160))],
)
@pytest.mark.parametrize("windowed", [False, True])
def test_host_decode_matches_in_graph_decode(family, size, hw, windowed):
    _, holder = _build(size, family=family)
    wrapper = _darknet_raw_heads_wrapper(holder).eval()
    if windowed:
        wrapper.output_windows = {"obj": (-8.0, 8.0), "cls": (-8.0, 8.0)}
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7))
    expected = _in_graph_decode(holder, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        anchors=wrapper.anchors,
        scale_x_y=wrapper.scale_x_y,
        output_windows=wrapper.output_windows,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    if windowed:
        restored = unwindow_raw_heads(_LAYOUT, raw, wrapper.output_windows)
        logits = [r for name, r in zip(wrapper.output_names, restored) if not name.endswith("reg")]
        assert max(float(np.abs(r).max()) for r in logits) < 8.0
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)


def test_decode_needs_anchors_per_level():
    _, holder = _build()
    wrapper = _darknet_raw_heads_wrapper(holder).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 64, 64))]
    with pytest.raises(ValueError, match="needs anchors for 2 levels"):
        decode_raw_heads(_LAYOUT, raw, strides=wrapper.strides, anchors=wrapper.anchors[:1])



@pytest.mark.parametrize("family", ["yolo3", "yolo4"])
def test_batchnorms_are_fused_without_changing_outputs(family):
    from libreyolo.models.darknet.blocks import DarknetConv

    _, holder = _build(family=family, seed=3)
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():  # non-trivial statistics, so the fold is exercised
        for module in holder.net.modules():
            if isinstance(module, DarknetConv) and module.bn is not None:
                n = module.bn.num_features
                module.bn.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.bn.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
                module.bn.weight.copy_(torch.rand(n, generator=generator) + 0.5)
                module.bn.bias.copy_(torch.randn(n, generator=generator) * 0.1)
    wrapper = _darknet_raw_heads_wrapper(holder).eval()

    assert not any(
        isinstance(m, DarknetConv) and m.bn is not None for m in wrapper.modules()
    )
    assert any(
        isinstance(m, DarknetConv) and m.bn is not None for m in holder.net.modules()
    )
    sample = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(9))
    expected = _in_graph_decode(holder, sample)
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(sample)]
    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        anchors=wrapper.anchors,
        scale_x_y=wrapper.scale_x_y,
    )
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-4)
