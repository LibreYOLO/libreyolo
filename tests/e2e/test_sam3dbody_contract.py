"""Consumer contract test: SAM3DBody-cpp raw YOLO9 ONNX export.

Why this test exists
--------------------
SAM3DBody-cpp — https://github.com/AmmarkoV/SAM3DBody-cpp — is a C++ 3D human
body estimation engine (ONNX Runtime, no Python at inference time) that uses
LibreYOLO for its person bounding boxes. Its author announced the switch in
the LibreYOLO v1.2.0 release thread on r/computervision (June 2026):
https://www.reddit.com/r/computervision/comments/1tt6pl8/ — making it the
first known external C++ consumer of our ONNX export. We want that
integration to keep working across releases; this module pins the export
behavior their code depends on.

How they consume LibreYOLO (as of June 2026, two places)
--------------------------------------------------------
1. ``tools/export_libreyolo.py`` in their repo runs, once and offline:

       model = LibreYOLO("LibreYOLO9t.pt")
       model.export(format="onnx", imgsz=640, opset=12)

   libreyolo is installed UNPINNED (plain ``pip install libreyolo``), so every
   re-export tracks our latest PyPI release — at the time of writing no
   version constraint protects them, which is why this test exists.

2. A pre-exported ``libreyolo9.onnx`` published on their HuggingFace model
   repo (AmmarkoV/SAM3DBody-cpp-onnx-models). Their binaries auto-prefer any
   ``libreyolo*.onnx`` found in their ``onnx/`` directory. This frozen
   artifact is immune to our changes, but anyone re-running the export tool
   in (1) replaces it with whatever we currently ship.

What their C++ requires
-----------------------
Their parser (``parse_yolov9_output``) performs its own NMS and reads only the
FIRST model output, which must be raw pre-NMS detections shaped
``[1, 4+nc, N]`` (or ``[1, N, 4+nc]``): 4 box coordinates in letterboxed
input-pixel space followed by per-class probabilities, with no objectness
column and no keypoints. Their export tool's shape check is print-only — it
never fails — so a contract change on our side would corrupt their detections
silently rather than fail loudly.

These tests pin every observable element of that contract. If one fails, the
change breaks downstream raw-output consumers: gate it behind a non-default
flag (as embedded NMS is, via ``nms=True``) or coordinate a major release.

Known consumer quirk, documented rather than pinned as correct: the
SAM3DBody-cpp parser currently decodes columns 0-3 as ``cx,cy,w,h`` while this
export emits ``x1,y1,x2,y2``. The xyxy convention is asserted below because the
library's own ONNX backend and the embedded-NMS wrapper both rely on it.
"""

import numpy as np
import onnx
import pytest

from libreyolo import LibreYOLO

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.onnx,
]

CONSUMER_WEIGHTS = "LibreYOLO9t.pt"
IMGSZ = 640
OPSET = 12
BOX_COLUMNS = 4
NUM_CLASSES = 80
EXPECTED_COLUMNS = BOX_COLUMNS + NUM_CLASSES  # 84
EXPECTED_ANCHORS = 8400  # 80*80 + 40*40 + 20*20 at imgsz 640


@pytest.fixture(scope="module")
def consumer_export(tmp_path_factory):
    """Run the consumer's exact export call once for the whole module."""
    out_dir = tmp_path_factory.mktemp("sam3dbody_contract")
    model = LibreYOLO(CONSUMER_WEIGHTS)
    produced = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        output_path=str(out_dir / "libreyolo9.onnx"),
    )
    return produced


@pytest.fixture(scope="module")
def ort_session(consumer_export):
    import onnxruntime as ort

    return ort.InferenceSession(
        consumer_export, providers=["CPUExecutionProvider"]
    )


@pytest.fixture(scope="module")
def first_output(ort_session):
    """First output for a dummy batch, exactly as the C++ caller requests it."""
    input_name = ort_session.get_inputs()[0].name
    dummy = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
    return ort_session.run(None, {input_name: dummy})[0]


class TestSAM3DBodyExportContract:
    def test_opset_12_is_accepted_and_kept(self, consumer_export):
        model = onnx.load(consumer_export)
        default_domain_opsets = {
            imp.domain: imp.version
            for imp in model.opset_import
            if imp.domain in ("", "ai.onnx")
        }
        assert default_domain_opsets, "model declares no default-domain opset"
        assert set(default_domain_opsets.values()) == {OPSET}

    def test_input_is_640_chw_float(self, ort_session):
        inp = ort_session.get_inputs()[0]
        assert inp.name == "images"
        assert inp.type == "tensor(float)"
        assert inp.shape[1:] == [3, IMGSZ, IMGSZ]

    def test_first_output_is_raw_pre_nms(self, first_output):
        # Raw head: [1, 84, 8400]. An embedded-NMS graph would instead emit
        # [1, max_det, 6], so anchor count and column count both guard it.
        assert first_output.dtype == np.float32
        assert first_output.shape == (1, EXPECTED_COLUMNS, EXPECTED_ANCHORS)

    def test_no_embedded_nms_in_default_export(self, consumer_export):
        model = onnx.load(consumer_export)
        metadata = {p.key: p.value for p in model.metadata_props}
        assert metadata.get("nms", "false").lower() != "true"

    def test_box_columns_are_xyxy_pixel_space(self, ort_session):
        # x2 > x1 and y2 > y1 must hold for every anchor — true only for
        # corner coordinates (a cx,cy,w,h layout violates it for anchors whose
        # center exceeds their size). Checked on two inputs so the property
        # cannot pass by accident of one activation pattern.
        input_name = ort_session.get_inputs()[0].name
        rng = np.random.RandomState(0)
        for image in (
            np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32),
            rng.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32),
        ):
            out = ort_session.run(None, {input_name: image})[0][0]
            x1, y1, x2, y2 = out[0], out[1], out[2], out[3]
            assert bool(np.all(x2 > x1))
            assert bool(np.all(y2 > y1))
            # Letterboxed pixel space, not normalized to [0, 1].
            assert float(x2.max()) > 50.0
            assert float(np.abs(out[:4]).max()) < 2 * IMGSZ

    def test_class_columns_are_probabilities(self, first_output):
        scores = first_output[0, BOX_COLUMNS:, :]
        assert scores.shape[0] == NUM_CLASSES
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0
