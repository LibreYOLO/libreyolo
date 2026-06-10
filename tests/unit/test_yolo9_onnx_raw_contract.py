"""Unit-tier guard for the raw YOLO9 ONNX export contract.

Weight-free twin of ``tests/e2e/test_sam3dbody_contract.py`` (see that module's
docstring for the full story: the SAM3DBody-cpp C++ engine consumes the default
YOLO9 ONNX export and depends on its exact output structure). Everything that
consumer relies on is structural — output shape, coordinate convention, score
range, opset, absence of embedded NMS — so a freshly initialized model guards
it on every push without downloading weights. Only the real-weights
``LibreYOLO("LibreYOLO9t.pt")`` resolution path is left to the e2e twin.
"""

import os
import tempfile

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.yolo9]

IMGSZ = 640
OPSET = 12
BOX_COLUMNS = 4
NUM_CLASSES = 80
EXPECTED_COLUMNS = BOX_COLUMNS + NUM_CLASSES  # 84
EXPECTED_ANCHORS = 8400  # 80*80 + 40*40 + 20*20 at imgsz 640


def test_default_onnx_export_is_raw_pre_nms_xyxy():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from libreyolo import LibreYOLO9

    model = LibreYOLO9(model_path=None, size="t", device="cpu")

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "yolo9_t_raw.onnx")
        path = model.export(
            format="onnx", imgsz=IMGSZ, opset=OPSET, half=False, output_path=out
        )
        assert os.path.exists(path)

        proto = onnx.load(path)
        default_domain_opsets = {
            imp.domain: imp.version
            for imp in proto.opset_import
            if imp.domain in ("", "ai.onnx")
        }
        assert set(default_domain_opsets.values()) == {OPSET}

        # The default export must not embed NMS — that is opt-in via nms=True.
        metadata = {p.key: p.value for p in proto.metadata_props}
        assert metadata.get("nms", "false").lower() != "true"

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        assert inp.name == "images"
        assert inp.type == "tensor(float)"
        assert inp.shape[1:] == [3, IMGSZ, IMGSZ]

        rng = np.random.RandomState(0)
        for image in (
            np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32),
            rng.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32),
        ):
            first = sess.run(None, {inp.name: image})[0]
            # Raw head: [1, 84, 8400]. An embedded-NMS graph would emit
            # [1, max_det, 6] instead, so both dims guard against it.
            assert first.dtype == np.float32
            assert first.shape == (1, EXPECTED_COLUMNS, EXPECTED_ANCHORS)

            # Boxes are x1,y1,x2,y2 in letterboxed pixel space. The DFL decode
            # guarantees x2 > x1 / y2 > y1 for every anchor regardless of
            # weights, so this holds for a fresh model too — a cx,cy,w,h
            # layout would violate it for anchors right of their own width.
            x1, y1, x2, y2 = first[0, 0], first[0, 1], first[0, 2], first[0, 3]
            assert bool(np.all(x2 > x1))
            assert bool(np.all(y2 > y1))
            assert float(x2.max()) > 50.0  # pixel space, not normalized
            assert float(np.abs(first[0, :BOX_COLUMNS]).max()) < 2 * IMGSZ

            scores = first[0, BOX_COLUMNS:, :]
            assert float(scores.min()) >= 0.0
            assert float(scores.max()) <= 1.0
