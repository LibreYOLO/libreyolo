"""Darknet (YOLOv2/v3/v4) raw-head export contract: split head convs, output windows and host decode parity."""

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
    from libreyolo import LibreYOLO2, LibreYOLO3, LibreYOLO4

    torch.manual_seed(seed)
    cls = {"yolo2": LibreYOLO2, "yolo3": LibreYOLO3, "yolo4": LibreYOLO4}[family]
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
     ("yolo3", "t", (160, 160)), ("yolo2", "t", (160, 160)), ("yolo2", "b", (192, 320))],
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
        wrapper.output_layout,
        raw,
        strides=wrapper.strides,
        anchors=wrapper.anchors,
        scale_x_y=wrapper.scale_x_y,
        output_windows=wrapper.output_windows,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    if windowed:
        restored = unwindow_raw_heads(wrapper.output_layout, raw, wrapper.output_windows)
        logits = [r for name, r in zip(wrapper.output_names, restored) if not name.endswith("reg")]
        assert max(float(np.abs(r).max()) for r in logits) < 8.0
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)


@pytest.mark.parametrize(("size", "reorgs"), [("t", 0), ("b", 1)])
def test_region_wrapper_output_contract(size, reorgs):
    from libreyolo.export.exporter import _ReorgConv

    _, holder = _build(size=size, family="yolo2")
    wrapper = _darknet_raw_heads_wrapper(holder).eval()
    (spec,) = holder.net.detection_specs()
    assert spec.kind == "region"
    assert wrapper.output_layout == "darknet_region_raw_heads_v1"
    assert wrapper.strides == (32,)
    assert wrapper.output_names == ["p5_reg", "p5_obj", "p5_cls"]
    # [region] anchors are in grid cells; the wrapper carries them in pixels.
    assert wrapper.anchors == [[[w * 32, h * 32] for w, h in spec.anchors]]
    assert sum(isinstance(layer, _ReorgConv) for layer in wrapper.model.layers) == reorgs
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 128, 160))
    count = len(spec.anchors)
    assert [tuple(o.shape) for o in outputs] == [(1, 4 * count, 4, 5), (1, count, 4, 5), (1, 5 * count, 4, 5)]


@pytest.mark.parametrize("stride", [2, 3])
def test_reorg_conv_matches_darknet_reorg_exactly(stride):
    from libreyolo.export.exporter import _ReorgConv
    from libreyolo.models.darknet.blocks import Reorg

    reorg = Reorg(stride)
    x = torch.randn(2, 6, 4 * stride, 5 * stride, generator=torch.Generator().manual_seed(3))
    conv = _ReorgConv(reorg, 6)
    with torch.no_grad():
        expected, actual = reorg(x), conv(x)
    assert torch.equal(actual, expected)
    assert set(conv.conv.weight.unique().tolist()) == {0.0, 1.0}
    assert int(conv.conv.weight.sum()) == 6 * stride * stride  # one input cell per output channel


def test_region_classes_get_a_softmax_on_the_host():
    _, holder = _build(size="t", family="yolo2")
    wrapper = _darknet_raw_heads_wrapper(holder).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 96, 96))]
    raw[1] = np.full_like(raw[1], 50.0)  # objectness ~1: the scores are the class distribution
    (decoded,) = decode_raw_heads(
        wrapper.output_layout, raw, strides=wrapper.strides, anchors=wrapper.anchors
    )
    np.testing.assert_allclose(decoded[0, 4:].sum(axis=0), 1.0, atol=1e-5)


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


@pytest.mark.parametrize("stride", [1, 2])
def test_local_as_convs_matches_darknet_local_layer(stride):
    """YOLOv1's [local] layer (an Einsum in ONNX) as per-position convolutions: same tensor."""
    from libreyolo.export.exporter import _LocalAsConvs
    from libreyolo.models.darknet.blocks import DarknetLocal

    out_side = 5 if stride == 1 else 3
    local = DarknetLocal(4, 6, 3, stride=stride, pad=1, activation="leaky", out_h=out_side, out_w=out_side)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        local.weight.copy_(torch.randn(local.weight.shape, generator=generator))
        local.bias.copy_(torch.randn(local.bias.shape, generator=generator))
    x = torch.randn(1, 4, 5, 5, generator=generator)
    with torch.no_grad():
        expected, actual = local(x), _LocalAsConvs(local)(x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_local_layers_to_convs_replaces_only_local_layers():
    from libreyolo.export.exporter import _LocalAsConvs, _local_layers_to_convs
    from libreyolo.models.darknet.blocks import DarknetLocal

    net = torch.nn.Module()
    net.layers = torch.nn.ModuleList(
        [torch.nn.Identity(), DarknetLocal(2, 3, 3, stride=1, pad=1, activation="linear", out_h=2, out_w=2)]
    )
    assert _local_layers_to_convs(net) == 1
    assert isinstance(net.layers[0], torch.nn.Identity)
    assert isinstance(net.layers[1], _LocalAsConvs)
    assert _local_layers_to_convs(net) == 0


def test_yolo1_batchnorm_fold_keeps_the_decoded_output():
    """Darknet norms folded into the convs (F53 on the Metis compiler): same decoded YOLOv1 output."""
    import copy

    from libreyolo import LibreYOLO1
    from libreyolo.export.exporter import _fuse_darknet_batchnorms
    from libreyolo.models.darknet.export import DarknetExportWrapper

    torch.manual_seed(0)
    model = LibreYOLO1(None, size="t", device="cpu")  # the [connected] head is sized for its 20 VOC classes
    with torch.no_grad():
        for module in model.model.net.modules():
            if hasattr(module, "running_var"):
                module.running_mean.uniform_(-0.2, 0.2)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.1, 0.1)
    reference = DarknetExportWrapper(model.model).eval()
    prepared = copy.deepcopy(reference).eval()
    assert _fuse_darknet_batchnorms(prepared.net) > 0
    x = torch.rand(1, 3, 448, 448, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected, actual = reference(x), prepared(x)
    assert actual.shape == expected.shape
    # Pixel-scale boxes: compare against the output's own scale (float rounding of the fold).
    assert float((actual - expected).abs().max()) <= 1e-4 * float(expected.abs().max()) + 1e-5


def test_yolo1_stretch_preprocessing_accepts_a_size_pair():
    """Export calibration and verify pass (h, w); the stretch resize must take it like an int."""
    from libreyolo.models.darknet.preprocess import preprocess_numpy_stretch

    image = np.random.default_rng(2).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    by_int, ratio_int = preprocess_numpy_stretch(image, 64)
    by_pair, ratio_pair = preprocess_numpy_stretch(image, (64, 64))
    np.testing.assert_array_equal(by_pair, by_int)
    assert ratio_pair == ratio_int == 1.0 and by_pair.shape == (3, 64, 64)
