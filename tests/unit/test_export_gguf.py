"""GGUF export: BatchNorm folding, RepConv fusion and the file round trip."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.unit

gguf = pytest.importorskip("gguf", reason="GGUF export needs the gguf extra")


def _randomize_bn(module: nn.Module, seed: int = 0) -> None:
    """Non-trivial BatchNorm statistics so folding is actually exercised."""
    gen = torch.Generator().manual_seed(seed)
    for bn in module.modules():
        if isinstance(bn, nn.BatchNorm2d):
            n = bn.num_features
            with torch.no_grad():
                bn.weight.copy_(torch.rand(n, generator=gen) + 0.5)
                bn.bias.copy_(torch.randn(n, generator=gen) * 0.1)
                bn.running_mean.copy_(torch.randn(n, generator=gen) * 0.1)
                bn.running_var.copy_(torch.rand(n, generator=gen) + 0.5)


def _build_yolo9(nb_classes: int = 3):
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(model_path=None, size="t", nb_classes=nb_classes, device="cpu")
    _randomize_bn(model.model)
    model.model.eval()
    return model


def test_folded_conv_matches_unfolded():
    from libreyolo.export.gguf import fold_conv_bn
    from libreyolo.models.yolo9.nn import Conv

    torch.manual_seed(0)
    block = Conv(8, 16, 3, stride=2, activation=False).eval()
    _randomize_bn(block)
    assert block.bn.eps == pytest.approx(1e-3)

    weight, bias = fold_conv_bn(block.conv, block.bn)
    x = torch.randn(2, 8, 17, 19)
    folded = torch.nn.functional.conv2d(
        x, weight, bias, block.conv.stride, block.conv.padding
    )
    torch.testing.assert_close(folded, block(x), rtol=1e-5, atol=1e-5)


def test_fused_repconv_matches_two_branches():
    from libreyolo.export.gguf import fuse_repconv
    from libreyolo.models.yolo9.nn import RepConv

    torch.manual_seed(1)
    block = RepConv(8, 8, 3).eval()
    _randomize_bn(block, seed=1)

    weight, bias = fuse_repconv(block)
    x = torch.randn(1, 8, 13, 13)
    fused = torch.nn.functional.silu(torch.nn.functional.conv2d(x, weight, bias, 1, 1))
    torch.testing.assert_close(fused, block(x), rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = _build_yolo9()
    path = tmp_path_factory.mktemp("gguf") / "LibreYOLO9t.gguf"
    out = model.export(format="gguf", output_path=str(path), imgsz=64)
    return model, out


def _reader_tensors(path):
    reader = gguf.GGUFReader(path)
    return reader, {t.name: t for t in reader.tensors}


def _field(reader, key):
    return reader.fields[key].contents()


def test_gguf_round_trip_metadata(exported):
    model, path = exported
    reader, tensors = _reader_tensors(path)

    assert _field(reader, "general.architecture") == "yolov9"
    assert _field(reader, "general.file_type") == 0
    assert _field(reader, "cv.task") == "detect"
    assert _field(reader, "cv.num_classes") == 3
    assert _field(reader, "cv.labels") == [model.names[i] for i in range(3)]
    assert _field(reader, "cv.input.width") == 64
    assert _field(reader, "cv.pre.pad") == "top_left"
    assert _field(reader, "cv.pre.resize") == "cv2_linear"
    assert _field(reader, "cv.post.kind") == "dfl"
    assert _field(reader, "yolov9.reg_max") == 16
    assert list(_field(reader, "yolov9.strides")) == [8, 16, 32]
    assert reader.fields["yolov9.strides"].types == [
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.UINT32,
    ]
    assert _field(reader, "yolov9.variant") == "t"
    assert _field(reader, "yolov9.repconv_fused") is True

    assert not any(name.startswith("aux") for name in tensors)
    assert not any(".bn." in name for name in tensors)
    # RepConv 1x1 branches (bottleneck.N.conv1.conv2) are fused away.
    assert not any(".conv1.conv2." in name for name in tensors)
    assert any(name.endswith("bottleneck.0.conv1.conv1.conv.weight") for name in tensors)
    stem = tensors["backbone.conv0.conv.weight"]
    assert list(stem.shape) == [3, 3, 3, 16]  # ggml ne order: W, H, I, O
    assert stem.tensor_type == gguf.GGMLQuantizationType.F32
    assert tensors["head.anchor_convs.0.2.weight"].shape[2] == 64 // 4  # groups=4


def _substitute_folded(nn_model: nn.Module, tensors) -> nn.Module:
    """Rebuild the network from GGUF tensors: plain convs, no BN, no RepConv.

    Every weight of the result comes from ``tensors``; ``nn_model`` only
    provides the module structure, so pass a differently initialized model.
    """
    from libreyolo.models.yolo9.nn import Conv, RepConv

    fused = copy.deepcopy(nn_model)
    rep_paths = [p for p, m in fused.named_modules() if isinstance(m, RepConv)]
    conv_paths = {p for p, m in fused.named_modules() if isinstance(m, Conv)}

    def _conv_from(path: str, ref: nn.Conv2d) -> nn.Conv2d:
        weight = torch.from_numpy(np.array(tensors[f"{path}.weight"].data)).float()
        bias = torch.from_numpy(np.array(tensors[f"{path}.bias"].data)).float()
        conv = nn.Conv2d(
            weight.shape[1] * ref.groups,
            weight.shape[0],
            weight.shape[-2:],
            stride=ref.stride,
            padding=(weight.shape[-2] // 2, weight.shape[-1] // 2),
            groups=ref.groups,
        )
        with torch.no_grad():
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
        return conv

    for path, module in list(fused.named_modules()):
        if any(path.startswith(f"{rep}.") for rep in rep_paths):
            continue
        if isinstance(module, RepConv):
            module.conv1 = nn.Sequential(_conv_from(f"{path}.conv1.conv", module.conv1.conv))
            module.forward = (lambda m: lambda x: m.act(m.conv1(x)))(module)
        elif isinstance(module, Conv):
            ref = module.conv
            module.conv = _conv_from(f"{path}.conv", ref)
            module.conv.padding = ref.padding
            module.bn = nn.Identity()
        elif isinstance(module, nn.Conv2d) and path.rsplit(".", 1)[0] not in conv_paths:
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(np.array(tensors[f"{path}.weight"].data)))
                module.bias.copy_(torch.from_numpy(np.array(tensors[f"{path}.bias"].data)))
    return fused.eval()


def test_gguf_tensors_reproduce_model_output(exported):
    model, path = exported
    _, tensors = _reader_tensors(path)
    nn_model = model.model
    nn_model.eval()
    torch.manual_seed(123)
    skeleton = _build_yolo9().model
    with torch.no_grad():
        for tower in skeleton.head.class_convs:
            tower[-1].bias.normal_()
    fused = _substitute_folded(skeleton, tensors)
    fused.head.export = True
    nn_model.head.export = True
    try:
        x = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            ref = nn_model(x)
            got = fused(x)
    finally:
        nn_model.head.export = False
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_gguf_fp16_weights_and_f32_biases(tmp_path):
    model = _build_yolo9()
    out = model.export(format="gguf", output_path=str(tmp_path / "m.gguf"), imgsz=64, half=True)
    reader, tensors = _reader_tensors(out)
    assert _field(reader, "general.file_type") == 1
    for name, tensor in tensors.items():
        expected = (
            gguf.GGMLQuantizationType.F16
            if name.endswith(".weight")
            else gguf.GGMLQuantizationType.F32
        )
        assert tensor.tensor_type == expected, name
    assert model.model.backbone.conv0.conv.weight.dtype == torch.float32


def test_gguf_support_is_yolo9_detect_only():
    from libreyolo.export.support import get_support

    assert get_support("yolo9", "detect", "gguf").tier == "available"
    for family in ("yolox", "rfdetr", "yolo9_e2e", "dfine"):
        assert get_support(family, "detect", "gguf").tier == "blocked"
