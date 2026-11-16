"""Hybrid Axelera exports (support tier "hybrid"): RT-DETR's CNN parts on Metis, its transformer on the host.

The fake SDK from test_axelera_backend plays the compiler and the device (it
runs each compiled part's FP32 ONNX), so the chain is checked end to end
without Axelera's packages.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.test_axelera_backend import _dataset, _image, _install_fake_sdk

pytestmark = pytest.mark.unit
pytest.importorskip("onnxruntime")

OUTPUTS = ("pred_logits", "pred_boxes")
IMGSZ = 160


@pytest.fixture(scope="module")
def sdk():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _install_fake_sdk(monkeypatch)


def _rtdetr():
    from libreyolo.models.rtdetr.model import LibreRTDETR

    return LibreRTDETR._from_scratch(size="r18", device="cpu", seed=0)


def test_rtdetr_axelera_is_a_hybrid_int8_row():
    from libreyolo.export.support import get_support, precision_support

    entry = get_support("rtdetr", "detect", "axelera")
    assert entry.tier == "hybrid"
    assert "host CPU" in entry.reason and "onnxruntime" in entry.constraint
    assert precision_support("rtdetr", "detect", "axelera") == (True, True)


def test_unfrozen_batchnorms_keep_the_outputs():
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.models.rtdetr.backbone import FrozenBatchNorm2d

    torch.manual_seed(0)
    norm = FrozenBatchNorm2d(4)
    norm.weight.uniform_(0.5, 2.0)
    norm.bias.normal_()
    norm.running_mean.normal_()
    norm.running_var.uniform_(0.5, 2.0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Sequential(norm)).eval()
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = net(x)
        _unfrozen_batchnorms(net)
        actual = net(x)
    assert type(net[1][0]) is torch.nn.BatchNorm2d and not net[1][0].training
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _assert_same_queries(actual, expected, tol):
    """Decoder outputs match up to the order of the queries.

    Query selection is a TopK over the encoder scores: in a random-init model
    two near-tied scores can swap under float noise, and the decoder is
    permutation-equivariant over its queries, so the outputs swap rows too.
    """
    a = np.concatenate([np.asarray(v, dtype=np.float64)[0] for v in actual], axis=-1)
    e = np.concatenate([np.asarray(v, dtype=np.float64)[0] for v in expected], axis=-1)
    assert a.shape == e.shape
    distance = (np.abs(a[:, None, :] - e[None, :, :]) - tol * (1 + np.abs(e[None, :, :]))).max(-1)
    assert (distance.min(axis=1) <= 0).all() and (distance.min(axis=0) <= 0).all()
    assert (np.diagonal(distance) <= 0).mean() >= 0.95


@pytest.mark.parametrize("split", ["backbone", "neck"])
def test_rtdetr_hybrid_parts_reproduce_the_native_model(split):
    from libreyolo.export.exporter import (
        _DETRBackboneWrapper,
        _DETRHostTailWrapper,
        _detr_neck_parts,
        _unfrozen_batchnorms,
    )

    net = _rtdetr().model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        if split == "backbone":
            actual = _DETRHostTailWrapper(copy_, OUTPUTS).eval()(*features)
        else:
            samples = [torch.rand(1, 3, IMGSZ, IMGSZ) for _ in range(2)]
            pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, samples)
            assert min(pack.scales) == 1.0
            assert all(float(k).hex().startswith("0x1.0000000000000p") for k in pack.scales)
            canvas = pack(*features)[0]
            assert canvas.shape == (1, 3 * 256, IMGSZ // 8, IMGSZ // 8)
            actual = decoder(*neck(canvas))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )
    # The caller's model keeps its backbone and its own input projections.
    assert net.backbone is not None
    assert isinstance(net.encoder.input_proj[0], torch.nn.Sequential)
    assert isinstance(net.decoder.input_proj[0], torch.nn.Sequential)


def _export(model, tmp_path, name):
    return Path(
        model.export(
            format="axelera",
            data=str(_dataset(tmp_path)),
            imgsz=IMGSZ,
            output_path=str(tmp_path / f"{name}_axelera_model"),
        )
    )


@pytest.mark.parametrize("split", ["neck", "backbone"])
def test_rtdetr_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch, split):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import axelera

    if split == "backbone":
        monkeypatch.setitem(
            axelera.AXELERA_SIZE_RECIPES, ("rtdetr", "detect", "r18"), {"hybrid_split": "backbone"}
        )
    model = _rtdetr()
    artifact = _export(model, tmp_path, f"rtdetr_{split}")
    assert not list(tmp_path.glob("*.export_intermediate*"))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "graph_outputs_v1"
    assert meta["graph_outputs"] == meta["output_names"] == list(OUTPUTS)
    assert meta["hybrid"]["split"] == split
    if split == "neck":
        assert len(meta["hybrid"]["canvas_scales"]) == 3
    kinds = ["axm" if "axm" in part else "onnx" for part in meta["parts"]]
    if split == "neck":
        assert kinds == ["axm", "onnx", "axm", "onnx"]
        assert meta["parts"][2]["input"] == "canvas"
        assert sorted(p.name for p in artifact.glob("*.axm")) == [
            "rtdetr_neck.axm",
            "rtdetr_neck_part2.axm",
        ]
    else:
        assert kinds == ["axm", "onnx"]
    assert meta["parts"][0]["input"] == "images"
    assert sorted(meta["parts"][0]["outputs"]) == ["feat_s16", "feat_s32", "feat_s8"]
    assert meta["parts"][-1]["outputs"] == list(OUTPUTS)

    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend" and backend.model_family == "rtdetr"
    image = _image(shape=(IMGSZ, IMGSZ, 3))
    blob = model._preprocess(image, input_size=IMGSZ)[0].numpy()
    actual = backend._run_inference(blob)
    with torch.no_grad():
        expected = model.model.eval()(torch.from_numpy(blob))
    _assert_same_queries(actual, [expected[name].numpy() for name in OUTPUTS], 1e-3)
    assert backend.chain_timing["images"] == 1
    assert backend.chain_timing["aipu_ms"] > 0 and backend.chain_timing["host_ms"] > 0
    result = backend(image, conf=0.0)
    assert len(result.boxes) > 0


def test_rtdetr_hybrid_calibration_takes_square_sizes(sdk):
    from libreyolo.export.exporter import AxeleraExporter

    model = _rtdetr()
    preprocess = AxeleraExporter(model)._calibration_preprocess()
    chw, _ = preprocess(_image(shape=(48, 64, 3)), (IMGSZ, IMGSZ))
    assert chw.shape == (3, IMGSZ, IMGSZ) and chw.dtype == np.float32
    with pytest.raises(ValueError, match="square"):
        preprocess(_image(), (IMGSZ, 2 * IMGSZ))


def test_unknown_hybrid_split_is_rejected(sdk, tmp_path, monkeypatch):
    from libreyolo.export import axelera

    monkeypatch.setitem(
        axelera.AXELERA_SIZE_RECIPES, ("rtdetr", "detect", "r18"), {"hybrid_split": "decoder"}
    )
    with pytest.raises(ValueError, match="hybrid split"):
        _export(_rtdetr(), tmp_path, "bad")


def test_unmeasured_rtdetr_sizes_are_not_available_yet():
    from libreyolo.export.support import BLOCKED_SIZES, UNMEASURED_SIZES

    unmeasured = UNMEASURED_SIZES.get(("rtdetr", "detect", "axelera"), {})
    assert "r18" not in unmeasured
    assert not set(unmeasured) & set(BLOCKED_SIZES.get(("rtdetr", "detect", "axelera"), {}))


def test_hybrid_neck_part_compiles_per_tensor(sdk, tmp_path, monkeypatch):
    """Depthwise backbones compile per channel; the neck part (graph input into a depthwise conv) stays per tensor."""
    import yaml

    from libreyolo.export import axelera

    monkeypatch.setitem(
        axelera.AXELERA_SIZE_RECIPES,
        ("rtdetr", "detect", "r18"),
        {"quantization_scheme": "hybrid_per_tensor_per_channel"},
    )
    artifact = _export(_rtdetr(), tmp_path, "rtdetr_dw")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["neck_quantization_scheme"] == "per_tensor_min_max"
    first = (artifact / "compiler_config_final.toml").read_text()
    second = (artifact / "compiler_config_final.part2.toml").read_text()
    assert "hybrid_per_tensor_per_channel" in first
    assert "per_tensor_min_max" in second and "hybrid_per_tensor_per_channel" not in second


def test_rtdetrv2_is_hybrid_and_its_parts_reproduce_the_native_model():
    from libreyolo.export.exporter import _AXELERA_HYBRID_OUTPUTS, _DETRBackboneWrapper, _detr_neck_parts
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.export.support import get_support
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    assert get_support("rtdetrv2", "detect", "axelera").tier == "hybrid"
    assert _AXELERA_HYBRID_OUTPUTS[("rtdetrv2", "detect")] == OUTPUTS
    net = LibreRTDETRv2._from_scratch(size="r18", device="cpu", seed=0).model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, [torch.rand(1, 3, IMGSZ, IMGSZ)])
        actual = decoder(*neck(pack(*features)[0]))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


def test_detr_hybrid_is_backbone_only_and_reproduces_the_native_model(sdk, tmp_path, monkeypatch):
    from libreyolo.export import axelera
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _VanillaDETRBackboneWrapper,
        _VanillaDETRHostTailWrapper,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.detr.model import LibreDETR

    assert get_support("detr", "detect", "axelera").tier == "hybrid"
    model = LibreDETR._from_scratch(size="r50", device="cpu", seed=0)
    net = model.model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        assert not any(type(m).__name__ == "FrozenBatchNorm2d" for m in copy_.modules())
        features = _VanillaDETRBackboneWrapper(copy_).eval()(x)
        actual = _VanillaDETRHostTailWrapper(copy_, OUTPUTS).eval()(*features)
    for name, value in zip(OUTPUTS, actual):
        torch.testing.assert_close(value, expected[name], atol=1e-4, rtol=1e-4)
    assert AxeleraExporter(model)._hybrid_split() == "backbone"
    monkeypatch.setitem(axelera.AXELERA_SIZE_RECIPES, ("detr", "detect", "r50"), {"hybrid_split": "neck"})
    with pytest.raises(ValueError, match="no convolutional neck"):
        AxeleraExporter(model)._hybrid_split()


@pytest.mark.parametrize("family", ["dfine", "deim", "rtdetrv4"])
def test_dfine_line_is_hybrid_and_its_parts_reproduce_the_native_model(family):
    from libreyolo.export.exporter import _AXELERA_HYBRID_OUTPUTS, _DETRBackboneWrapper, _detr_neck_parts
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.export.support import get_support
    from libreyolo.models.deim.model import LibreDEIM
    from libreyolo.models.dfine.model import LibreDFINE

    assert get_support(family, "detect", "axelera").tier == "hybrid"
    assert _AXELERA_HYBRID_OUTPUTS[(family, "detect")] == OUTPUTS
    from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

    cls = {"dfine": LibreDFINE, "deim": LibreDEIM, "rtdetrv4": LibreRTDETRv4}[family]
    net = cls._from_scratch(size="s" if family == "rtdetrv4" else "n", device="cpu", seed=0).model.eval()
    # D-FINE-n has two levels (strides 16 and 32): 320 px gives 500 positions for its 300 queries.
    size = 320
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, [torch.rand(1, 3, size, size)])
        actual = decoder(*neck(pack(*features)[0]))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


def test_faster_rcnn_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.faster_rcnn.model import LibreFasterRCNN

    assert get_support("faster_rcnn", "detect", "axelera").tier == "hybrid"
    monkeypatch.setitem(support.UNMEASURED_SIZES, ("faster_rcnn", "detect", "axelera"), {})
    model = LibreFasterRCNN._from_scratch(size="n", device="cpu", seed=0)
    data = str(_dataset(tmp_path))
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=320,
            data=data,
            output_path=str(tmp_path / "frcnn_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["fpn0", "fpn1"]
    assert meta["parts"][-1]["outputs"] == ["boxes", "scores", "labels"]
    backend = LibreYOLO(str(artifact))
    image = _image(shape=(240, 320, 3))
    result = backend(image, conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4
    with pytest.raises(ValueError, match="canvas"):
        model.export(
            format="axelera", imgsz=256, data=data, output_path=str(tmp_path / "bad_axelera_model")
        )


def test_hybrid_neck_scheme_is_per_family_with_a_recipe_override(sdk, monkeypatch):
    from libreyolo.export import axelera
    from libreyolo.export.exporter import AxeleraExporter
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    assert AxeleraExporter(_rtdetr())._hybrid_neck_scheme() == "per_tensor_min_max"
    v2 = LibreRTDETRv2._from_scratch(size="r18", device="cpu", seed=0)
    # r18 keeps the min-max neck it was measured with (a per-size recipe).
    assert AxeleraExporter(v2)._hybrid_neck_scheme() == "per_tensor_min_max"
    monkeypatch.setitem(axelera.AXELERA_SIZE_RECIPES, ("rtdetrv2", "detect", "r18"), {})
    assert AxeleraExporter(v2)._hybrid_neck_scheme() == "per_tensor_histogram"


def test_faster_rcnn_measured_sizes_leave_the_unmeasured_list():
    from libreyolo.export.support import UNMEASURED_SIZES

    assert ("faster_rcnn", "detect", "axelera") not in UNMEASURED_SIZES


def test_deimv2_small_sizes_take_the_encoder_split_and_reproduce_the_native_model():
    from libreyolo.export.exporter import AxeleraExporter, _detr_conv_encoder_parts, _unfrozen_batchnorms
    from libreyolo.export.support import BLOCKED_SIZES, get_support
    from libreyolo.models.deimv2.model import LibreDEIMv2

    assert get_support("deimv2", "detect", "axelera").tier == "hybrid"
    assert "DINOv3" in BLOCKED_SIZES[("deimv2", "detect", "axelera")]["s"]
    model = LibreDEIMv2._from_scratch(size="atto", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "encoder"
    net = model.model.eval()
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        expected = net(x)
        head, tail = _detr_conv_encoder_parts(_unfrozen_batchnorms(copy.deepcopy(net).eval()), OUTPUTS)
        memory = head(x)
        assert len(memory) == len(head.output_names) == 2
        actual = tail(*memory)
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@pytest.mark.parametrize("task", ["segment", "detect"])
def test_mask_rcnn_hybrid_chain_matches_the_export_wrapper(task):
    """Mask R-CNN backbone-split hybrid in FP32: the Metis part (normalization, backbone, FPN)
    then the host step (pool level, RPN, RoI box and mask heads, mask paste) reproduce
    MaskRCNNExportWrapper at the canvas."""
    import copy

    from libreyolo.export.exporter import (
        _AXELERA_HYBRID_OUTPUTS,
        _HYBRID_BACKBONE_WRAPPERS,
        _MaskRCNNHostTailWrapper,
        _RCNNBackboneWrapper,
    )
    from libreyolo.models.mask_rcnn.nn import LibreMaskRCNNModel, MaskRCNNExportWrapper

    assert _HYBRID_BACKBONE_WRAPPERS["mask_rcnn"] == ("_RCNNBackboneWrapper", "_MaskRCNNHostTailWrapper")
    outputs = _AXELERA_HYBRID_OUTPUTS[("mask_rcnn", task)]
    torch.manual_seed(0)
    net = LibreMaskRCNNModel().eval()
    net.transform.min_size, net.transform.max_size = (256,), 256
    net.roi_heads.score_thresh = 0.0
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = MaskRCNNExportWrapper(copy.deepcopy(net), include_masks=task == "segment").eval()(x)
        levels = _RCNNBackboneWrapper(net).eval()(x)
        got = _MaskRCNNHostTailWrapper(net, outputs, canvas=(256, 256)).eval()(*levels)
    assert len(got) == len(want) == len(outputs)
    assert got[0].shape[0] == want[0].shape[0] > 0
    # Random-init ResNet-50-FPN v2: float accumulation moves box corners by < 0.01 px (of 256).
    torch.testing.assert_close(got[0], want[0], atol=0.05, rtol=0)
    torch.testing.assert_close(got[1], want[1], atol=1e-3, rtol=0)
    assert torch.equal(got[2], want[2])
    if task == "segment":
        assert got[3].shape == want[3].shape and got[3].shape[-2:] == (256, 256)
        torch.testing.assert_close(got[3], want[3], atol=1e-2, rtol=0)


def test_mask_rcnn_hybrid_segment_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.mask_rcnn.model import LibreMaskRCNN

    assert get_support("mask_rcnn", "segment", "axelera").tier == "hybrid"
    # r50 is measured for both tasks (docs/axelera.md).
    assert ("mask_rcnn", "segment", "axelera") not in support.UNMEASURED_SIZES
    assert ("mask_rcnn", "detect", "axelera") not in support.UNMEASURED_SIZES
    model = LibreMaskRCNN._from_scratch(size="r50", task="segment", device="cpu", seed=0)
    # A small canvas keeps the test light; the export takes the model's min_size as its canvas.
    model.model.transform.min_size, model.model.transform.max_size = (256,), 256
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=256,
            data=str(_dataset(tmp_path)),
            output_path=str(tmp_path / "maskrcnn_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["fpn0", "fpn1", "fpn2", "fpn3"]
    assert meta["parts"][-1]["outputs"] == ["boxes", "scores", "labels", "masks"]
    backend = LibreYOLO(str(artifact))
    result = backend(_image(shape=(240, 320, 3)), conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4
    if len(result.boxes):
        assert tuple(result.masks.data.shape[-2:]) == (240, 320)


def test_ppdoclayout_hybrid_chain_matches_the_export_wrapper():
    """PP-DocLayoutV3 backbone-split hybrid in FP32: the Metis part (HGNetV2 and the encoder
    input projections) then the host step reproduce PPDocLayoutV3ExportWrapper."""
    import copy

    from libreyolo.export.exporter import (
        _AXELERA_HYBRID_OUTPUTS,
        _DocLayoutBackboneWrapper,
        _DocLayoutHostTailWrapper,
    )
    from libreyolo.models.ppdoclayoutv3.nn import PPDocLayoutV3ExportWrapper, PPDocLayoutV3Model

    outputs = _AXELERA_HYBRID_OUTPUTS[("ppdoclayoutv3", "detect")]
    torch.manual_seed(0)
    net = PPDocLayoutV3Model().eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = PPDocLayoutV3ExportWrapper(copy.deepcopy(net)).eval()(x)
        head = _DocLayoutBackboneWrapper(net).eval()
        levels = head(x)
        got = _DocLayoutHostTailWrapper(net, outputs).eval()(*levels)
    assert head.output_names == ("feat_s4", "proj_s8", "proj_s16", "proj_s32")
    assert net.backbone is not None and net.encoder_input_proj is not None
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


def test_ppdoclayout_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.ppdoclayoutv3.model import LibrePPDocLayoutV3

    assert get_support("ppdoclayoutv3", "detect", "axelera").tier == "hybrid"
    assert ("ppdoclayoutv3", "detect", "axelera") not in support.UNMEASURED_SIZES
    model = LibrePPDocLayoutV3._from_scratch(size="l", device="cpu", seed=0)
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=256,
            data=str(_dataset(tmp_path)),
            output_path=str(tmp_path / "doclayout_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["feat_s4", "proj_s16", "proj_s32", "proj_s8"]
    assert meta["parts"][-1]["outputs"] == ["pred_logits", "pred_boxes", "order_votes"]
    result = LibreYOLO(str(artifact))(_image(shape=(240, 320, 3)), conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4


def test_deformable_detr_hybrid_is_backbone_only_and_reproduces_the_native_model():
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _DeformableDETRBackboneWrapper,
        _DeformableDETRHostTailWrapper,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.deformable_detr.model import LibreDeformableDETR

    assert get_support("deformable_detr", "detect", "axelera").tier == "hybrid"
    model = LibreDeformableDETR._from_scratch(size="r50", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        actual = _DeformableDETRHostTailWrapper(copy_, OUTPUTS)(*_DeformableDETRBackboneWrapper(copy_)(x))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )
