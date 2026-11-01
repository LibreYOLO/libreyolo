"""TorchScript exports of the YOLO9 family run on any device.

A traced graph keeps the device of every tensor it creates. The YOLO9 head
builds its anchor grid in forward, so an export traced on CPU (the TorchScript
default) failed on CUDA with "Expected all tensors to be on the same device".
The exporter now pins the grid as constants for the fixed export canvas, and
``torch.jit.load(map_location=...)`` moves constants with the weights.
"""

from __future__ import annotations

import re

import pytest
import torch

import libreyolo

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("half", [False, True], ids=["fp32", "fp16"])
@pytest.mark.parametrize("cls_name", ["LibreYOLO9", "LibreYOLO9E2E", "LibreYOLO9P2"])
def test_traced_graph_bakes_no_device(cls_name, half, tmp_path):
    model = getattr(libreyolo, cls_name)(None, size="t", device="cpu")
    path = model.export(
        format="torchscript",
        output_path=str(tmp_path / "m.torchscript"),
        imgsz=320,
        half=half,
    )

    graph = str(torch.jit.load(path, map_location="cpu").inlined_graph)
    assert not re.search(r'Device = prim::Constant\[value="', graph)
    # The live model goes back to shape-derived grids after the export.
    assert model.model.head._frozen_grid is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device")
def test_cpu_traced_export_runs_on_another_device(tmp_path):
    model = libreyolo.LibreYOLO9(None, size="t", device="cpu")
    path = model.export(
        format="torchscript", output_path=str(tmp_path / "m.torchscript"), imgsz=320
    )
    x = torch.randn(1, 3, 320, 320)

    on_cpu = torch.jit.load(path, map_location="cpu")(x)
    on_mps = torch.jit.load(path, map_location="mps")(x.to("mps"))

    first = lambda out: out[0] if isinstance(out, (tuple, list)) else out  # noqa: E731
    assert torch.allclose(first(on_cpu), first(on_mps).cpu(), atol=1e-2)
