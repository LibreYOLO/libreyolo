"""CPU contracts for the GTR depth task (random weights, no downloads)."""

import random

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.depth import LibreGTRDepthModel, is_depth_state_dict
from libreyolo.models.gtr.nn import LibreGTRModel
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


@pytest.fixture(scope="module")
def depth_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = LibreGTRDepthModel("s")
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=1,
        names={0: "depth"},
        task="depth",
        imgsz=640,
    )
    path = tmp_path_factory.mktemp("gtr_depth") / "LibreGTRs-depth.pt"
    torch.save(ckpt, path)
    return path, model


def _make_depth_yaml(root, n_images=2, size=96):
    for split in ("train", "val"):
        img_dir = root / "images" / split
        depth_dir = root / "depths" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_images):
            arr = np.zeros((size, size + 32, 3), dtype=np.uint8)
            arr[:, : size // 2] = (200, 40, 40)
            arr[:, size // 2 :] = (40, 40, 200)
            Image.fromarray(arr).save(img_dir / f"img{i}.jpg")
            depth = np.full((size, size + 32), 8.0)
            depth[:, : size // 2] = 2.0
            encoded = (depth * 256.0).round().astype(np.uint16)
            Image.fromarray(encoded).save(depth_dir / f"img{i}.png")
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val"}
        )
    )
    return path


def test_forward_is_inverse_of_metric_depth_with_in_graph_normalization():
    torch.manual_seed(0)
    model = LibreGTRDepthModel("s").eval()
    x = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        inverse = model(x)
        metric = model.forward_metric(x)
    assert inverse.shape == (1, 1, 160, 160)
    assert (metric > 0).all()
    torch.testing.assert_close(inverse[:, 0], metric.reciprocal())
    # Normalization buffers are not part of the checkpoint.
    assert "pixel_mean" not in model.state_dict()
    with pytest.raises(ValueError, match="square"):
        model(torch.rand(1, 3, 160, 192))


def test_state_dict_routing_and_size_detection():
    depth = LibreGTRDepthModel("s").state_dict()
    detect = LibreGTRModel("s").state_dict()
    assert is_depth_state_dict(depth) and not is_depth_state_dict(detect)
    assert LibreGTR.can_load(depth) and LibreGTR.can_load(detect)
    assert LibreGTR.detect_checkpoint_task(depth) == "depth"
    assert LibreGTR.detect_checkpoint_task(detect) is None
    assert LibreGTR.detect_nb_classes(depth) == 1
    assert LibreGTR.detect_size(depth) == "s"
    # L and X share the backbone width; the MLP ratio tells them apart.
    for size, rows in (("l", 2048), ("x", 3072)):
        fake = {
            "backbone.backbone._model.blocks.0.attn.q_proj.weight": torch.zeros(1, 384),
            "backbone.backbone._model.blocks.0.mlp.gate_proj.weight": torch.zeros(
                rows, 384
            ),
        }
        assert LibreGTR.detect_size(fake) == size
    for other in ("LibreDFINE", "LibreEC", "LibreDEIM"):
        import libreyolo

        assert not getattr(libreyolo, other).can_load(depth)


def test_download_urls_route_depth_repositories():
    assert LibreGTR.get_download_url("LibreGTRm-depth.pt") == (
        "https://huggingface.co/LibreYOLO/LibreGTRm-depth/resolve/main/"
        "LibreGTRm-depth.pt"
    )
    assert "/LibreGTRs/resolve/74193dc" in LibreGTR.get_download_url("LibreGTRs.pt")
    assert LibreGTR.get_download_url("LibreGTRs-pose.pt") is None


def test_factory_loads_depth_checkpoint_and_predicts(depth_checkpoint):
    path, reference = depth_checkpoint
    model = LibreYOLO(str(path), device="cpu")
    assert isinstance(model, LibreGTR)
    assert (model.task, model.size, model.nb_classes) == ("depth", "s", 1)
    assert model.names == {0: "depth"}
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.model.state_dict()[key], rtol=0, atol=0)
    image = np.random.default_rng(0).integers(0, 255, (90, 150, 3), dtype=np.uint8)
    result = model.predict(image, imgsz=160)[0]
    depth = np.asarray(result.depth_map.data)
    assert depth.shape == (90, 150)
    assert np.isfinite(depth).all() and (depth > 0).all()
    assert result.boxes is None or len(result.boxes) == 0
    with torch.no_grad():
        x, *_ = model._preprocess(image, input_size=160)
        expected = model.model(x)
    # Native preprocessing is a square stretch to [0, 1] RGB.
    assert x.shape == (1, 3, 160, 160) and 0 <= float(x.min()) <= float(x.max()) <= 1
    assert expected.shape == (1, 1, 160, 160)


def test_task_mismatch_is_rejected(depth_checkpoint, tmp_path):
    path, _ = depth_checkpoint
    with pytest.raises(RuntimeError, match="task"):
        LibreGTR(str(path), size="s", task="detect", device="cpu")
    detect = wrap_libreyolo_checkpoint(
        LibreGTRModel("s", 2).state_dict(),
        model_family="gtr",
        size="s",
        nc=2,
        task="detect",
        imgsz=640,
    )
    detect_path = tmp_path / "LibreGTRs.pt"
    torch.save(detect, detect_path)
    with pytest.raises(RuntimeError, match="depth"):
        LibreGTR(str(detect_path), size="s", task="depth", device="cpu")


def test_depth_val_and_training_smoke(depth_checkpoint, tmp_path):
    path, _ = depth_checkpoint
    data = _make_depth_yaml(tmp_path)
    model = LibreYOLO(str(path), device="cpu")
    metrics = model.val(data=str(data), imgsz=160, batch=2, workers=0, verbose=False)
    assert np.isfinite(metrics["metrics/abs_rel"])
    frozen = {k: v.clone() for k, v in model.model.state_dict().items()}
    results = model.train(
        data=str(data),
        epochs=1,
        batch=2,
        imgsz=160,
        device="cpu",
        workers=0,
        warmup_iters=0,
        ema=False,
        project=str(tmp_path / "runs"),
        name="depth",
    )
    assert results["best_checkpoint"] or results.get("last_checkpoint")
    changed = any(
        not torch.equal(frozen[k], v) for k, v in model.model.state_dict().items()
    )
    assert changed
    reloaded = LibreYOLO(
        str(results.get("best_checkpoint") or results["last_checkpoint"])
    )
    assert reloaded.task == "depth" and reloaded.names == {0: "depth"}
    with pytest.raises(ValueError, match="LoRA"):
        model.train(data=str(data), epochs=1, lora=True, device="cpu")


def test_depth_onnx_export_matches_pytorch(depth_checkpoint, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    path, _ = depth_checkpoint
    model = LibreYOLO(str(path), device="cpu")
    exported = model.export(
        format="onnx",
        imgsz=160,
        simplify=False,
        output_path=str(tmp_path / "gtr_depth.onnx"),
    )
    session = ort.InferenceSession(str(exported))
    x = torch.rand(1, 3, 160, 160)
    (onnx_out,) = session.run(None, {session.get_inputs()[0].name: x.numpy()})
    with torch.no_grad():
        expected = model.model(x).numpy()
    assert onnx_out.shape == (1, 1, 160, 160)
    np.testing.assert_allclose(onnx_out, expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(NotImplementedError):
        model.export(format="openvino")
