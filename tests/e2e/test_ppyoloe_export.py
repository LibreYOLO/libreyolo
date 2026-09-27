"""PP-YOLOE export: raw graph parity, metadata, and public detection parity.

Every advertised row in ``docs/export_support.md`` is executed here rather than
inferred from an ONNX success: TorchScript, ONNX, OpenVINO and a real TensorRT
engine build each run through ``LibreYOLO(artifact)`` and are compared to eager
PyTorch on batch 1 and batch 2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.e2e.conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.ppyoloe, pytest.mark.export_backend]

WEIGHTS = "LibrePPYOLOEs.pt"
IMGSZ = 640


def _native(weights):
    from libreyolo import LibreYOLO

    model = LibreYOLO(weights, device="cpu")
    model.model.eval()
    return model


def _split_outputs(outputs):
    arrays = [np.asarray(o) for o in outputs]
    boxes = next(a for a in arrays if a.shape[-1] == 4)
    scores = next(a for a in arrays if a.shape[-1] != 4)
    return boxes, scores


def _eager(model, x):
    with torch.no_grad():
        (boxes, scores), _ = model.model(x)
    return boxes.numpy(), scores.numpy()


def _export(model, tmp_path, fmt, **kwargs):
    # dynamic=True adds a batch axis only; the spatial canvas stays fixed at
    # 640, which is what docs/export_support.md advertises. The batch gate
    # needs it: a batch-1-only graph cannot be fed two images.
    kwargs.setdefault("dynamic", True)
    return model.export(
        format=fmt,
        output_path=str(tmp_path / f"LibrePPYOLOEs.{fmt}"),
        imgsz=IMGSZ,
        half=False,
        **kwargs,
    )


def _raw_probe(image, batch):
    """Batch of preprocessed real pixels.

    Pure noise is a poor probe for this head: the DFL bins saturate and the
    decoded distances blow up, so absolute box differences there say more about
    float accumulation order than about the exported graph. The second image in
    the batch is inverted so a graph that broadcasts one row cannot pass.
    """
    from libreyolo.models.ppyoloe.utils import preprocess_image

    tensor, _, _, _ = preprocess_image(image, input_size=IMGSZ)
    if batch == 1:
        return tensor
    return torch.cat([tensor, -tensor], dim=0)


# Boxes are only compared where the anchor carries real signal. Below this
# score the DFL distribution is near-uniform and the decoded distance runs far
# off canvas, so a max-abs comparison over all 8400 anchors measures float
# accumulation order on predictions postprocessing discards, not graph fidelity.
RAW_BOX_SCORE_FLOOR = 0.05


def _assert_raw_parity(model, backend, image, box_atol, score_atol, batches=(1, 2)):
    for batch in batches:
        x = _raw_probe(image, batch)
        want_boxes, want_scores = _eager(model, x)
        got_boxes, got_scores = _split_outputs(backend._run_inference(x.numpy()))
        assert got_boxes.shape == want_boxes.shape
        assert got_scores.shape == want_scores.shape
        # Scores are compared everywhere: they are bounded and meaningful.
        assert np.abs(got_scores - want_scores).max() <= score_atol
        keep = want_scores.max(axis=-1) > RAW_BOX_SCORE_FLOOR
        assert keep.sum() > 100, "probe carries too little signal to gate on"
        assert np.abs(got_boxes[keep] - want_boxes[keep]).max() <= box_atol


def _assert_public_parity(model, backend, image, box_atol, score_atol):
    # A single image returns one Results holding every detection.
    want = model.predict(image, conf=0.5)
    got = backend.predict(image, conf=0.5)
    assert len(want.boxes) > 1, "parity on a single detection is too weak a check"
    want_boxes = np.asarray(want.boxes.xyxy)
    got_boxes = np.asarray(got.boxes.xyxy)
    assert got_boxes.shape == want_boxes.shape
    assert sorted(np.asarray(got.boxes.cls).tolist()) == sorted(
        np.asarray(want.boxes.cls).tolist()
    )
    if want_boxes.size:
        order_want = np.argsort(np.asarray(want.boxes.conf))
        order_got = np.argsort(np.asarray(got.boxes.conf))
        assert np.abs(want_boxes[order_want] - got_boxes[order_got]).max() <= box_atol
        assert (
            np.abs(
                np.asarray(want.boxes.conf)[order_want]
                - np.asarray(got.boxes.conf)[order_got]
            ).max()
            <= score_atol
        )


@pytest.mark.torchscript
def test_torchscript_is_bit_exact(tmp_path, sample_image):
    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    model = _native(weights)
    artifact = _export(model, tmp_path, "torchscript")
    backend = LibreYOLO(str(artifact), device="cpu")

    assert backend.model_family == "ppyoloe"
    assert backend.task == "detect"
    assert backend.imgsz == IMGSZ

    _assert_raw_parity(model, backend, sample_image, box_atol=0.0, score_atol=0.0)
    _assert_public_parity(model, backend, sample_image, box_atol=0.0, score_atol=0.0)


@pytest.mark.onnx
def test_onnx_raw_contract_and_parity(tmp_path, sample_image):
    import onnx

    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    model = _native(weights)
    artifact = _export(model, tmp_path, "onnx", simplify=False)

    graph = onnx.load(str(artifact)).graph
    # Raw contract: boxes + sigmoid scores, no NMS and no objectness column.
    assert [o.name for o in graph.output] == ["boxes", "scores"]

    backend = LibreYOLO(str(artifact), device="cpu")
    assert backend.model_family == "ppyoloe"
    assert backend.task == "detect"
    assert backend.imgsz == IMGSZ
    assert backend.names[0] == "person"

    _assert_raw_parity(model, backend, sample_image, box_atol=1e-2, score_atol=1e-5)
    _assert_public_parity(model, backend, sample_image, box_atol=1e-2, score_atol=1e-5)


@pytest.mark.onnx
def test_onnx_rejects_a_shape_it_was_not_exported_for(tmp_path):
    """Only a fixed 640 spatial canvas is advertised, so 640x960 must not pass."""
    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    model = _native(weights)
    artifact = _export(model, tmp_path, "onnx", simplify=False, dynamic=False)
    backend = LibreYOLO(str(artifact), device="cpu")
    with pytest.raises(Exception):
        backend._run_inference(np.zeros((1, 3, 640, 960), dtype=np.float32))


@pytest.mark.openvino
def test_openvino_parity(tmp_path, sample_image):
    pytest.importorskip("openvino")
    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    model = _native(weights)
    artifact = _export(model, tmp_path, "openvino")
    backend = LibreYOLO(str(artifact), device="cpu")

    assert backend.model_family == "ppyoloe"
    _assert_raw_parity(model, backend, sample_image, box_atol=5.0, score_atol=1e-2)
    _assert_public_parity(model, backend, sample_image, box_atol=0.5, score_atol=1e-3)


@pytest.mark.tensorrt
def test_tensorrt_fp32_engine_parity(tmp_path, sample_image):
    """Build and run a real engine; an ONNX parse alone is not TensorRT support.

    Fixed batch 1 on purpose. An engine built with a dynamic batch profile
    reproduces batch 1 fine but diverges materially at batch 2 (raw scores off
    by ~0.24, and one extra detection survives NMS), so TensorRT is advertised
    at fixed batch 1 only. OpenVINO carries the batch-1-and-2 compiled-backend
    coverage.
    """
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT engine build needs a CUDA device")
    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    model = _native(weights)
    artifact = _export(model, tmp_path, "engine", dynamic=False)
    backend = LibreYOLO(str(artifact))

    assert backend.model_family == "ppyoloe"
    _assert_raw_parity(
        model, backend, sample_image, box_atol=2.0, score_atol=1e-2, batches=(1,)
    )
    _assert_public_parity(model, backend, sample_image, box_atol=0.5, score_atol=1e-2)

