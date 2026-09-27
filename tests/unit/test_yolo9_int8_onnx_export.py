"""YOLO9 ONNX INT8 export smoke tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.onnx, pytest.mark.export_backend]


_needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None
    or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx/onnxruntime not installed",
)


def _calibration_yaml(tmp_path):
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for idx in range(2):
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{idx}.jpg")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {tmp_path.as_posix()}",
                "train: images/train",
                "val: images/train",
                "nc: 2",
                "names:",
                "  0: object",
                "  1: other",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


@_needs_onnx
def test_yolo9_detect_onnx_int8_export_loads_and_predicts(tmp_path):
    import onnx
    import onnxruntime as ort

    from libreyolo import LibreYOLO, LibreYOLO9

    data_yaml = _calibration_yaml(tmp_path)

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    for block in model.model.head.cv3:
        convs = [m for m in block.modules() if isinstance(m, torch.nn.Conv2d)]
        convs[-1].bias.data.fill_(4.0)
    fp32_path = tmp_path / "LibreYOLO9t.onnx"
    int8_path = tmp_path / "LibreYOLO9t_int8.onnx"

    model.export(
        "onnx",
        output_path=str(fp32_path),
        imgsz=64,
        simplify=False,
        dynamic=False,
    )
    exported_int8 = model.export(
        "onnx",
        output_path=str(int8_path),
        imgsz=64,
        simplify=False,
        dynamic=False,
        int8=True,
        data=str(data_yaml),
    )

    assert exported_int8 == str(int8_path)
    assert int8_path.stat().st_size < fp32_path.stat().st_size

    proto = onnx.load(str(int8_path))
    metadata = {p.key: p.value for p in proto.metadata_props}
    assert metadata["model_family"] == "yolo9"
    assert metadata["task"] == "detect"
    assert metadata["precision"] == "int8"

    input_type = proto.graph.input[0].type.tensor_type.elem_type
    assert input_type == onnx.TensorProto.FLOAT

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"images": np.zeros((1, 3, 64, 64), dtype=np.float32)})
    assert outs[0].shape == (1, 6, 84)
    assert float(outs[0][0, 4:, :].max()) > 0.25

    loaded = LibreYOLO(str(int8_path), device="cpu")
    result = loaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.0, imgsz=64)
    assert result.boxes is not None


def _quantized_conv_names(graph):
    """Conv nodes whose weight input comes from a DequantizeLinear node."""
    producers = {output: node for node in graph.node for output in node.output}
    return {
        node.name
        for node in graph.node
        if node.op_type == "Conv"
        and getattr(producers.get(node.input[1]), "op_type", None)
        == "DequantizeLinear"
    }


@_needs_onnx
def test_yolo9_onnx_int8_keeps_the_first_conv_and_head_float(tmp_path):
    """The family's float layers stay float, as in ``model.quantize()``.

    Quantized class-logit convs saturate at the calibrated maximum: with a
    maximum of 0 every score reads exactly sigmoid(0) = 0.5.
    """
    import onnx

    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    path = model.export(
        "onnx",
        output_path=str(tmp_path / "int8.onnx"),
        imgsz=64,
        simplify=False,
        dynamic=False,
        int8=True,
        data=str(_calibration_yaml(tmp_path)),
    )

    graph = onnx.load(path).graph
    convs = {node.name for node in graph.node if node.op_type == "Conv"}
    quantized = _quantized_conv_names(graph)
    float_scopes = ("/head/", "/backbone/conv0/")
    head_convs = {name for name in convs if any(s in name for s in float_scopes)}
    assert head_convs and not head_convs & quantized
    assert quantized == convs - head_convs

    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node.op_type)
    for node in graph.node:
        if node.name in head_convs and "/head/" in node.name:
            # Head outputs (class logits, box distributions) stay unclipped.
            assert "QuantizeLinear" not in consumers.get(node.output[0], []), node.name


@_needs_onnx
def test_yolo9_onnx_int8_explicit_nodes_to_exclude_replace_the_default(tmp_path):
    import onnx

    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    path = model.export(
        "onnx",
        output_path=str(tmp_path / "int8.onnx"),
        imgsz=64,
        simplify=False,
        dynamic=False,
        int8=True,
        data=str(_calibration_yaml(tmp_path)),
        nodes_to_exclude=[],
    )

    graph = onnx.load(path).graph
    convs = {node.name for node in graph.node if node.op_type == "Conv"}
    assert _quantized_conv_names(graph) == convs


@_needs_onnx
def test_yolo9_onnx_int8_export_with_dynamic_batch(tmp_path):
    from libreyolo import LibreYOLO, LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    path = model.export(
        "onnx",
        output_path=str(tmp_path / "int8.onnx"),
        imgsz=64,
        simplify=False,
        dynamic=True,
        int8=True,
        data=str(_calibration_yaml(tmp_path)),
    )

    backend = LibreYOLO(path, device="cpu")
    outputs = backend._run_inference(np.zeros((2, 3, 64, 64), dtype=np.float32))
    assert outputs[0].shape[0] == 2
