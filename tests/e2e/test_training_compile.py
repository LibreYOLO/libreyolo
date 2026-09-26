"""``train(compile=...)`` on CUDA through the public API, for both flagships.

Every compile failure mode degrades to eager with a warning, so a run that
merely completes proves nothing: these tests assert that the compiler was
built, that no fallback warning fired and that Dynamo compiled frames and
never hit its recompile limit. The recipes are the families' defaults
(YOLO9 with its PGI auxiliary branch; RF-DETR with per-batch multi-scale and
``nbs=16`` accumulation), plus compiler-managed CUDA graphs without
accumulation. No throughput or accuracy claim is made here.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests.e2e.conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("compile_train_data")
    rng = np.random.default_rng(0)
    for split, count in (("train", 16), ("valid", 4)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / split / "images" / f"{i:03d}.jpg")
            rows = []
            for _ in range(rng.integers(2, 6)):
                cx, cy = rng.uniform(0.2, 0.8, 2)
                w, h = rng.uniform(0.05, 0.2, 2)
                rows.append(f"{rng.integers(0, 2)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (root / split / "labels" / f"{i:03d}.txt").write_text("\n".join(rows))
    (root / "data.yaml").write_text(
        yaml.dump(
            {"path": str(root), "train": "train/images", "val": "valid/images",
             "nc": 2, "names": ["a", "b"]}
        )
    )
    return root / "data.yaml"


CASES = [
    pytest.param("LibreYOLO9t.pt", {"imgsz": 320, "batch": 4}, False, id="yolo9-pgi"),
    pytest.param("LibreRFDETRn.pt", {"batch": 4}, False, id="rfdetr-multiscale-accum"),
    pytest.param(
        "LibreYOLO9t.pt", {"imgsz": 320, "batch": 4, "nbs": 4}, True, id="yolo9-graphs"
    ),
    pytest.param(
        "LibreRFDETRn.pt",
        {"batch": 4, "nbs": 4, "multi_scale": False},
        True,
        id="rfdetr-graphs",
    ),
]


@pytest.mark.parametrize("weights,extra,cuda_graph", CASES)
def test_compiled_training_engages_and_saves_eager_checkpoints(
    weights, extra, cuda_graph, dataset, tmp_path, caplog
):
    from torch._dynamo.utils import counters

    import libreyolo

    torch._dynamo.reset()
    counters.clear()
    model = libreyolo.LibreYOLO(require_test_weights(weights), device="cuda")
    with caplog.at_level(logging.INFO):
        result = model.train(
            data=str(dataset),
            epochs=2,
            device=0,
            workers=0,
            seed=0,
            project=str(tmp_path),
            name="compiled",
            exist_ok=True,
            eval_interval=1,
            compile=True,
            cuda_graph=cuda_graph,
            **extra,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("compiling the training network" in m for m in messages), messages
    assert not any("training runs eager" in m for m in messages), messages
    assert not any("recompile_limit" in m for m in messages), messages
    assert any(f"CUDA graph replay={cuda_graph}" in m for m in messages), messages
    assert counters["frames"]["ok"] > 0
    assert all(np.isfinite(result["epoch_losses"]))
    assert len(result["val_metrics"]) == 2

    checkpoint = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        if isinstance(checkpoint.get(key), dict):
            assert not any("_orig_mod" in name for name in checkpoint[key])
    reloaded = libreyolo.LibreYOLO(str(result["last_checkpoint"]), device="cuda")
    assert reloaded.predict(np.zeros((320, 320, 3), dtype=np.uint8)) is not None
