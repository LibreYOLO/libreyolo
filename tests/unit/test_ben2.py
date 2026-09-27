"""Offline contract tests for the BEN2 matte family."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def _export_stub():
    return SimpleNamespace(
        task="matte",
        device=torch.device("cpu"),
        _get_model_name=lambda: "ben2",
        _get_input_size=lambda: 1024,
    )


@pytest.mark.external_data
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("BEN2_LIBRE_CHECKPOINT"),
    reason="set BEN2_LIBRE_CHECKPOINT to a converted LibreBEN2b-matte.pt",
)
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_ben2_trained_export_raw_parity(tmp_path, format):
    if format == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    from libreyolo import LibreYOLO

    checkpoint = os.environ["BEN2_LIBRE_CHECKPOINT"]
    model = LibreYOLO(checkpoint, device="cpu")
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "matte8"
        / "images"
        / "furry_blob.png"
    )
    tensor, _, _, _ = model._preprocess(str(fixture))
    with torch.inference_mode():
        expected = model.model(tensor).numpy()

    artifact = model.export(
        format=format,
        imgsz=1024,
        dynamic=False,
        simplify=False,
        output_path=str(tmp_path / f"ben2-matte.{format}"),
    )
    if format == "onnx":
        import onnx

        graph = onnx.load(artifact, load_external_data=False).graph
        assert [output.name for output in graph.output] == ["matte"]
    actual = LibreYOLO(artifact, device="cpu")._run_inference(tensor.numpy())[0]
    if format == "torchscript":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=4e-3)


@pytest.mark.external_data
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("BEN2_LIBRE_CHECKPOINT"),
    reason="set BEN2_LIBRE_CHECKPOINT to a converted LibreBEN2b-matte.pt",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP16")
@pytest.mark.parametrize("format", ["onnx", "torchscript"])
def test_ben2_trained_fp16_export_smoke(tmp_path, format):
    if format == "onnx":
        onnx = pytest.importorskip("onnx")

    from libreyolo import LibreYOLO

    model = LibreYOLO(os.environ["BEN2_LIBRE_CHECKPOINT"], device="cuda")
    artifact = model.export(
        format=format,
        imgsz=1024,
        dynamic=False,
        simplify=False,
        half=True,
        device="cuda",
        output_path=str(tmp_path / f"ben2-matte-fp16.{format}"),
    )
    assert Path(artifact).is_file()
    if format == "onnx":
        graph = onnx.load(artifact, load_external_data=False).graph
        assert graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    else:
        scripted = torch.jit.load(artifact, map_location="cuda").eval()
        tensor = torch.zeros(1, 3, 1024, 1024, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            output = scripted(tensor)
        assert output.dtype == torch.float16


def _synthetic_ben2_state_dict() -> dict:
    return {
        "backbone.patch_embed.proj.weight": torch.zeros(128, 3, 4, 4),
        "multifieldcrossatt.attention.4.out_proj.weight": torch.zeros(128, 128),
        "dec_blk4.sal_conv.weight": torch.zeros(1, 128, 1, 1),
        "insmask_head.6.weight": torch.zeros(128, 384, 3, 3),
    }


def test_ben2_checkpoint_detection_and_filename():
    from libreyolo.models.ben2 import LibreBEN2

    state_dict = _synthetic_ben2_state_dict()
    assert LibreBEN2.can_load(state_dict) is True
    assert LibreBEN2.detect_size(state_dict) == "b"
    assert LibreBEN2.detect_nb_classes(state_dict) == 1
    assert LibreBEN2.detect_checkpoint_task(state_dict) == "matte"
    assert LibreBEN2.detect_size_from_filename("LibreBEN2b-matte.pt") == "b"
    assert LibreBEN2.detect_size_from_filename("LibreBEN2b.pt") is None


def test_ben2_export_rejects_non_batch_one():
    from libreyolo.export.exporter import OnnxExporter

    with pytest.raises(ValueError, match=r"batch-1.*batch=2"):
        OnnxExporter(_export_stub())(batch=2, dynamic=False)


@pytest.mark.parametrize("imgsz", [640, (1024, 640)])
def test_ben2_export_rejects_non_native_imgsz(imgsz):
    from libreyolo.export.exporter import OnnxExporter

    with pytest.raises(ValueError, match=r"fixed native resolution 1024x1024"):
        OnnxExporter(_export_stub())._resolve_params(
            output_path="ben2.onnx",
            imgsz=imgsz,
            device="cpu",
            half=False,
            int8=False,
        )


def test_ben2_rejects_other_matte_and_generic_state_dicts():
    from libreyolo.models.ben2 import LibreBEN2
    from libreyolo.models.birefnet import LibreBiRefNet
    from libreyolo.models.feynobg import LibreFeyNobg

    ben2 = _synthetic_ben2_state_dict()
    birefnet = {
        "bb.patch_embed.proj.weight": torch.zeros(192, 3, 4, 4),
        "squeeze_module.0.conv_in.weight": torch.zeros(1),
        "decoder.ipt_blk5.conv1.weight": torch.zeros(1),
        "decoder.gdt_convs_attn_4.0.weight": torch.zeros(1),
    }
    assert LibreBEN2.can_load(birefnet) is False
    assert LibreBiRefNet.can_load(ben2) is False
    assert LibreFeyNobg.can_load(ben2) is False
    assert LibreBEN2.can_load({"model.0.weight": torch.zeros(1)}) is False


def test_ben2_raw_checkpoint_autoconverts(tmp_path):
    from libreyolo.models.autoconvert import autoconvert_upstream_checkpoint
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    source = tmp_path / "ben2-base.pt"
    torch.save(_synthetic_ben2_state_dict(), source)

    converted = autoconvert_upstream_checkpoint(str(source))

    assert converted is not None
    converted_path = Path(converted)
    assert converted_path.name == "ben2-base-LibreBEN2b-matte.pt"
    checkpoint = torch.load(converted_path, map_location="cpu", weights_only=False)
    assert validate_checkpoint_metadata(checkpoint, strict=False) == []
    assert checkpoint["model_family"] == "ben2"
    assert checkpoint["size"] == "b"
    assert checkpoint["task"] == "matte"
    assert checkpoint["nc"] == 1
    assert checkpoint["names"] == {0: "matte"}
    assert checkpoint["imgsz"] == 1024


def test_ben2_patch_rearrangement_round_trip():
    from libreyolo.models.ben2.nn import _image_to_patches, _patches_to_image

    image = torch.arange(2 * 3 * 8 * 10).reshape(2, 3, 8, 10)
    patches = _image_to_patches(image)
    assert patches.shape == (8, 3, 4, 5)
    assert torch.equal(_patches_to_image(patches), image)


def test_ben2_preprocess_matches_lanczos_contract():
    from libreyolo.models.ben2.utils import preprocess_numpy

    source = np.zeros((5, 7, 3), dtype=np.uint8)
    source[:, 3:] = (255, 100, 20)
    actual, ratio = preprocess_numpy(source, input_size=11)
    resized = (
        np.asarray(
            Image.fromarray(source, mode="RGB").resize(
                (11, 11), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    expected = (
        resized - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected.transpose(2, 0, 1))
    assert ratio == 1.0


def test_ben2_postprocess_applies_sigmoid_once():
    from libreyolo.postprocess.ben2 import postprocess

    logits = torch.tensor([[[[0.0, 2.0], [-2.0, 0.0]]]])
    matte = postprocess(logits, original_size=(2, 2))["matte"]
    torch.testing.assert_close(matte, torch.sigmoid(logits)[0, 0])
