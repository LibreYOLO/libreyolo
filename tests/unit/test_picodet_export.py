"""PICODET export tests: ONNX + TorchScript shape and numerics."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.picodet]


def test_export_onnx_round_trip_matches_eager():
    """Exported ONNX model must produce numerically identical detections to
    the eager-mode model on the same input.

    This catches export-time bugs that don't surface in shape-only tests:
    silent dtype casts, opset-version regressions, and tracer specialization
    on dynamic axes.
    """
    onnx = pytest.importorskip("onnx")  # noqa: F841
    pytest.importorskip("onnxruntime")

    from libreyolo import LibrePICODET, LibreYOLO

    eager = LibrePICODET(size="s", nb_classes=80, device="cpu")

    # Run eager forward in export mode
    eager.model.head.export = True
    eager.model.eval()
    rng = np.random.default_rng(0)
    img_np = rng.standard_normal((1, 3, 320, 320), dtype=np.float32)
    with torch.no_grad():
        eager_out = eager.model(torch.from_numpy(img_np)).numpy()
    eager.model.head.export = False  # leave it as we found it

    # Round-trip through ONNX
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "picodet_s.onnx")
        path = eager.export(format="onnx", imgsz=320, half=False, output_path=out)
        assert os.path.exists(path)

        m = LibreYOLO(path)
        # Hit the underlying session directly to compare raw outputs:
        sess = m.session  # type: ignore[attr-defined]
        ort_out = sess.run(None, {sess.get_inputs()[0].name: img_np})[0]

    # Bit-equivalent check is too strict (ONNX may upgrade ops, fuse, etc.).
    # Use a tight tolerance instead.
    np.testing.assert_allclose(ort_out, eager_out, rtol=1e-4, atol=1e-4)


def test_export_torchscript_runs():
    from libreyolo import LibrePICODET

    m = LibrePICODET(size="s", nb_classes=80, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "picodet_s.torchscript")
        path = m.export(format="torchscript", imgsz=320, half=False, output_path=out)
        assert os.path.exists(path)

        ts = torch.jit.load(path)
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            y = ts(x)
        # Single fused output: (1, N, 4 + nc) = (1, 2125, 84) at imgsz=320.
        assert y.shape == (1, 2125, 84), f"unexpected shape {y.shape}"
