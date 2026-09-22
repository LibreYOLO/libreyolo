"""Opt-in CUDA compiler lifecycle tests through the public training API.

These use real flagship weights and a generated detection dataset. They check
compiler execution, eager validation, optimizer/EMA checkpoint integrity and
reload. They do not establish a throughput or final-accuracy claim.
"""

import gc

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests.e2e.conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.extended_training,
    pytest.mark.slow,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor"),
]


@pytest.fixture
def compile_dataset(tmp_path):
    for split, count in (("train", 8), ("val", 2)):
        images = tmp_path / "images" / split
        labels = tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(count):
            image = np.full((192, 192, 3), 32 + index, dtype=np.uint8)
            image[48:144, 48:144] = (240, 90, 40)
            Image.fromarray(image).save(images / f"{index}.png")
            (labels / f"{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    path = tmp_path / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["square"],
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "weight,imgsz",
    [
        pytest.param("LibreYOLO9t.pt", 128, marks=pytest.mark.yolo9),
        pytest.param("LibreRFDETRn.pt", 384, marks=pytest.mark.rfdetr),
    ],
)
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_compiled_training_validation_and_reload(
    weight,
    imgsz,
    cuda_graph,
    compile_dataset,
    tmp_path,
    monkeypatch,
):
    from libreyolo import LibreYOLO
    from libreyolo.training import compile as compile_module

    managers = []
    build_manager = compile_module.build_compile_manager

    def remember_manager(trainer):
        manager = build_manager(trainer)
        managers.append((trainer, manager))
        return manager

    monkeypatch.setattr(compile_module, "build_compile_manager", remember_manager)
    model = LibreYOLO(require_test_weights(weight), device="cuda")
    result = model.train(
        data=str(compile_dataset),
        epochs=2,
        batch=2,
        nbs=2,
        imgsz=imgsz,
        device="cuda",
        workers=0,
        amp=True,
        compile=True,
        cuda_graph=cuda_graph,
        project=str(tmp_path / "run"),
        name="compiled",
        exist_ok=True,
        eval_interval=1,
        save_period=1,
        save_plots=False,
        seed=313,
        **(
            {"multi_scale": False, "crop_resize_prob": 0.0}
            if "RFDETR" in weight
            else {}
        ),
    )
    assert len(managers) == 1
    trainer, manager = managers[0]
    assert manager is not None and manager.started and not manager.disabled
    assert manager.cuda_graph is cuda_graph
    assert trainer._cuda_graph_manager is None
    assert manager.spec.network.module is trainer.model
    assert np.isfinite(result["final_loss"])
    checkpoint = trainer.save_dir / "weights" / "last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["ema_updates"] > 0
    for key in ("model", "train_model", "ema"):
        assert set(state[key]) == set(trainer.model.state_dict())
        assert all("_orig_mod" not in name for name in state[key])
        assert all(
            torch.isfinite(t).all()
            for t in state[key].values()
            if t.is_floating_point()
        )
    optimizer_params = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert optimizer_params == {
        id(p) for p in trainer.model.parameters() if p.requires_grad
    }
    reloaded = LibreYOLO(str(checkpoint), device="cuda")
    predictions = reloaded.predict(
        np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz
    )
    assert predictions is not None
    managers.clear()
    del reloaded, model, trainer, manager, state
    gc.collect()
    torch.cuda.empty_cache()
