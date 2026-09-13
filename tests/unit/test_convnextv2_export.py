"""ConvNeXt V2 export metadata and runtime parity with local random tensors."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreConvNeXtV2, LibreYOLO

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_export_roundtrip(tmp_path, fmt):
    if fmt == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    torch.manual_seed(19)
    model = LibreConvNeXtV2(nb_classes=7, device="cpu")
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if name.endswith(("grn.gamma", "grn.beta")):
                parameter.normal_(std=0.1)
    path = model.export(
        format=fmt,
        output_path=str(tmp_path / f"model.{fmt}"),
        imgsz=224,
        dynamic=False,
        simplify=False,
    )
    backend = LibreYOLO(str(path), device="cpu")
    assert backend.task == "classify"
    assert backend.names == model.names
    image = Image.fromarray(
        np.random.default_rng(4).integers(0, 256, (277, 389, 3), dtype=np.uint8)
    )
    tensor = model._preprocess(image)[0]
    with torch.inference_mode():
        native_logits = model.model(tensor).numpy()
        if fmt == "onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            exported_logits = session.run(
                None, {session.get_inputs()[0].name: tensor.numpy()}
            )[0]
        else:
            exported_logits = torch.jit.load(str(path), map_location="cpu")(
                tensor
            ).numpy()
    np.testing.assert_allclose(exported_logits, native_logits, rtol=1e-4, atol=1e-5)
    expected = model.predict(image)[0].probs
    actual = backend.predict(image)[0].probs
    np.testing.assert_allclose(
        np.asarray(actual.data), np.asarray(expected.data), rtol=1e-4, atol=1e-6
    )
    assert actual.top1 == expected.top1
