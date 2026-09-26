"""FP16 exports of the flagship families: loadable, and the model untouched.

``export(half=True)`` casts a model to float16 to trace it. The caller's
model must come back bit-exact (casting back with ``float()`` keeps the fp16
rounding), and the exported graph must load and predict through LibreYOLO,
whose preprocessing produces float32 inputs.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def _build_yolo9():
    from libreyolo import LibreYOLO9

    return LibreYOLO9(model_path=None, size="t", nb_classes=3, device="cpu"), 64


def _build_rfdetr():
    pytest.importorskip(
        "transformers", reason="RF-DETR needs the rfdetr extra", exc_type=ImportError
    )
    from libreyolo import LibreRFDETR

    # ``{}`` is RF-DETR's random-init convention; ``None`` downloads weights.
    return LibreRFDETR({}, size="n", nb_classes=3, device="cpu"), 128


_BUILDERS = {"yolo9": _build_yolo9, "rfdetr": _build_rfdetr}


@pytest.fixture(scope="module", params=sorted(_BUILDERS))
def half_export(request, tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model, imgsz = _BUILDERS[request.param]()
    model.model.eval()
    before = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    tensor = torch.rand(1, 3, imgsz, imgsz)
    from libreyolo.export.exporter import OnnxExporter

    with OnnxExporter(model)._model_context(
        "cpu", False, False, 1, (imgsz, imgsz)
    ) as (wrapped, _), torch.no_grad():
        native = wrapped(tensor)
    if isinstance(native, torch.Tensor):
        native = (native,)
    native = [output.detach().numpy() for output in native]
    artifact = model.export(
        format="onnx",
        half=True,
        imgsz=imgsz,
        simplify=False,
        output_path=str(tmp_path_factory.mktemp(request.param) / "half.onnx"),
    )
    return request.param, model, before, artifact, tensor, native


def test_half_export_leaves_the_model_bit_exact(half_export):
    _, model, before, *_ = half_export
    after = model.model.state_dict()

    assert after.keys() == before.keys()
    for key, tensor in before.items():
        assert after[key].dtype == tensor.dtype, key
        assert torch.equal(after[key], tensor), key


def test_half_onnx_export_loads_and_predicts(half_export):
    import onnxruntime as ort

    from libreyolo import LibreYOLO

    family, _, _, artifact, tensor, native = half_export
    session = ort.InferenceSession(artifact, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float16)"

    backend = LibreYOLO(artifact, device="cpu")
    outputs = backend._run_inference(tensor.numpy())

    assert len(outputs) == len(native)
    for output, expected in zip(outputs, native):
        assert output.dtype == np.float32
        assert output.shape == expected.shape
        assert np.isfinite(output).all()
        if family == "yolo9":
            # Dense head: row-aligned with the fp32 graph. RF-DETR's in-graph
            # top-k reorders near-tied random-init queries under fp16.
            close = np.isclose(output, expected, rtol=2e-2, atol=5e-2)
            assert float(close.mean()) > 0.95
    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    result = backend.predict(image, conf=0.0)
    assert result.orig_shape == (48, 64)
