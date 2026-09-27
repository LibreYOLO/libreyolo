"""Real-weight exported runtime parity; no downloads in the unit gate."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.external_data]


@pytest.mark.parametrize("format", ["onnx", "torchscript"])
@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_gtr_export(format, size, tmp_path, monkeypatch):
    from libreyolo import LibreYOLO

    path = Path(os.environ.get("GTR_CHECKPOINTS", "")) / f"gtr_{size}_coco.pth"
    if not path.is_file():
        pytest.skip("Set GTR_CHECKPOINTS to local upstream weights")
    pytest.importorskip("onnxruntime")
    image = Path(__file__).parents[1] / "fixtures/dog.jpg"
    model = LibreYOLO(str(path), device="cpu")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.chdir(tmp_path)
    try:
        artifact = model.export(format=format, imgsz=640, dynamic=False, simplify=False)
        if format == "onnx":
            import onnx

            graph = onnx.load(str(artifact))
            assert len(graph.graph.node) < 5000  # token recurrence must remain a Loop
            assert sum(node.op_type == "Loop" for node in graph.graph.node) == 12
        backend = LibreYOLO(str(artifact), device="cpu")
        expected = model.predict(str(image), imgsz=640, conf=0.25, max_det=300)
        actual = backend.predict(str(image), conf=0.25, max_det=300)
        assert len(expected.boxes) > 0
        np.testing.assert_array_equal(actual.boxes.cls, expected.boxes.cls)
        np.testing.assert_allclose(
            actual.boxes.xyxy, expected.boxes.xyxy, atol=0.05, rtol=1e-4
        )
        np.testing.assert_allclose(
            actual.boxes.conf, expected.boxes.conf, atol=1e-5, rtol=1e-3
        )
    finally:
        torch.set_num_threads(old_threads)
