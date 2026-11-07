"""Gated parity against the released InstantHMR ONNX graphs.

Set ``LIBREYOLO_INSTANTHMR_ONNX`` to a folder holding upstream's
``instanthmr-<size>.onnx`` files and ``mhr-body.onnx`` (Hugging Face
``momolesang/InstantHMR``), and ``LIBREYOLO_INSTANTHMR_WEIGHTS`` to a folder
holding the converted ``LibreInstantHMR<size>-mesh[-commercial].pt`` files.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

RELEASES = [
    ("n", "nano", ""),
    ("t", "tiny", ""),
    ("s", "small", ""),
    ("m", "medium", ""),
    ("m", "medium-commercial", "-commercial"),
]


@pytest.mark.parametrize("size,upstream,variant", RELEASES)
def test_converted_weights_match_released_graphs(size, upstream, variant):
    onnx_dir = os.environ.get("LIBREYOLO_INSTANTHMR_ONNX")
    weights_dir = os.environ.get("LIBREYOLO_INSTANTHMR_WEIGHTS")
    if not onnx_dir or not weights_dir:
        pytest.skip("set LIBREYOLO_INSTANTHMR_ONNX and LIBREYOLO_INSTANTHMR_WEIGHTS")
    ort = pytest.importorskip("onnxruntime")
    from libreyolo import LibreInstantHMR

    network = Path(onnx_dir) / f"instanthmr-{upstream}.onnx"
    body = Path(onnx_dir) / "mhr-body.onnx"
    weights = Path(weights_dir) / f"LibreInstantHMR{size}-mesh{variant}.pt"
    model = LibreInstantHMR(str(weights), size=size, device="cpu", person_detector=None)

    g = torch.Generator().manual_seed(7)
    image = torch.randn(4, 3, 288, 288, generator=g)
    cond = torch.randn(4, 3, generator=g) * 0.3
    names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
    net = ort.InferenceSession(str(network), providers=["CPUExecutionProvider"])
    want = dict(
        zip(names, net.run(names, {"image": image.numpy(), "cliff_cond": cond.numpy()}))
    )
    rig = ort.InferenceSession(str(body), providers=["CPUExecutionProvider"])
    want["joints_3d"], want["vertices"] = rig.run(
        ["joints_3d", "vertices"],
        {"mhr_params": want["mhr_params"], "shape_params": want["shape_params"]},
    )
    with torch.no_grad():
        out = model.model(image, cond)
        joints_3d, vertices = model.model.body(out["mhr_params"], out["shape_params"])
    got = {**{k: out[k] for k in names}, "joints_3d": joints_3d, "vertices": vertices}
    for name in want:
        np.testing.assert_allclose(
            got[name].numpy(), want[name], atol=5e-4, err_msg=name
        )
