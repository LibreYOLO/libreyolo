"""LibreEfficientNetV2 ONNX export: single logits output, numerics match eager.

EfficientNetV2 uses TF-SAME padding (runtime F.pad) and SiLU; this guards the
known export footgun (the memory-efficient swish op) by asserting a clean
round-trip through onnxruntime.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.onnx]


def test_export_onnx_round_trip_matches_eager():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    from libreyolo import LibreEfficientNetV2

    eager = LibreEfficientNetV2(size="b0", nb_classes=1000, device="cpu")
    eager.model.eval()
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    with torch.no_grad():
        eager_out = eager.model(torch.from_numpy(img)).numpy()
    assert eager_out.shape == (1, 1000)

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "effv2b0.onnx")
        path = eager.export(format="onnx", imgsz=224, half=False, output_path=out)
        assert os.path.exists(path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        assert len(sess.get_outputs()) == 1  # single logits tensor
        ort_out = sess.run(None, {sess.get_inputs()[0].name: img})[0]

    assert ort_out.shape == (1, 1000)
    np.testing.assert_allclose(ort_out, eager_out, rtol=1e-4, atol=1e-4)


def test_same_padding_survives_dynamic_batch_onnx_simplify(tmp_path):
    # export() defaults to a dynamic batch axis plus onnxsim. Traced SAME-pad
    # shape arithmetic used to be folded into wrong Conv pads there.
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    onnxsim = pytest.importorskip("onnxsim")

    from libreyolo.models.efficientnetv2.nn import Conv2dSame

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        *(Conv2dSame(c, 4, 3, stride=2) for c in (3, 4, 4))
    ).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = net(x).numpy()

    path = tmp_path / "same.onnx"
    torch.onnx.export(
        net,
        x,
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    simplified, ok = onnxsim.simplify(onnx.load(str(path)))
    assert ok
    sess = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = sess.run(None, {"images": x.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
