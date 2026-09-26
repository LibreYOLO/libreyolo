"""CPU contracts for GTR semantic segmentation (random weights, small inputs)."""

import random

import numpy as np
import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr import sem
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


def _small(size="s", nc=19, window=64):
    torch.manual_seed(0)
    return sem.LibreGTRSemModel(size, nc, window=window).eval()


def _checkpoint(tmp_path, model, name="LibreGTRs-sem.pt", size="s"):
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size=size,
        nc=model.num_classes,
        names=sem.CITYSCAPES_NAMES if model.num_classes == 19 else None,
        task="semantic",
        imgsz=1024,
    )
    path = tmp_path / name
    torch.save(ckpt, path)
    return path


def test_state_dict_is_recognized_without_colliding_with_detect():
    from libreyolo.models.gtr.nn import LibreGTRModel

    semantic = sem.LibreGTRSemModel("s").state_dict()
    detect = LibreGTRModel("s").state_dict()
    assert LibreGTR.can_load(semantic) and LibreGTR.can_load(detect)
    assert LibreGTR.detect_checkpoint_task(semantic) == "semantic"
    assert LibreGTR.detect_checkpoint_task(detect) is None
    assert LibreGTR.detect_nb_classes(semantic) == 19
    assert sem.is_semantic_state_dict(semantic)
    assert not sem.is_semantic_state_dict(detect)


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_size_detection_from_semantic_weights(size):
    state = {
        k: v
        for k, v in sem.LibreGTRSemModel(size).state_dict().items()
        if k.startswith(("backbone.backbone._model.blocks.0.", "head."))
    }
    assert LibreGTR.detect_size(state) == size


def test_sliding_window_matches_the_upstream_protocol():
    model = _small()
    x = torch.rand(1, 3, 64, 160)
    normalized = (x - model.pixel_mean) / model.pixel_std
    with torch.no_grad():
        out = model(x)
        # Reference: windows at x = 0, 48 and the last one shifted to 96.
        expected = torch.zeros_like(out)
        count = torch.zeros(1, 1, 64, 160)
        for left in (0, 48, 96):
            expected[..., left : left + 64] += model.forward_normalized(
                normalized[..., left : left + 64]
            )
            count[..., left : left + 64] += 1
    torch.testing.assert_close(out, expected / count)
    # A square input with a side divisible by 32 is a single pass.
    square = torch.rand(1, 3, 96, 96)
    with torch.no_grad():
        torch.testing.assert_close(
            model(square),
            model.forward_normalized((square - model.pixel_mean) / model.pixel_std),
        )


def test_inputs_shorter_than_the_window_are_rescaled_back():
    model = _small()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 40, 100))
    assert out.shape == (2, 19, 40, 100)


def test_training_loss_ignores_255_and_backpropagates():
    model = _small().train()
    targets = torch.full((1, 64, 64), sem.IGNORE_INDEX, dtype=torch.long)
    losses = model(torch.rand(1, 3, 64, 64), targets=targets)
    assert losses["total_loss"].item() == 0
    targets[:, :32] = 3
    losses = model(torch.rand(1, 3, 64, 64), targets=targets)
    assert torch.isfinite(losses["total_loss"]) and losses["total_loss"] > 0
    losses["total_loss"].backward()
    assert model.head.classifier.weight.grad.abs().sum() > 0


def test_checkpoint_load_predict_and_task_guard(tmp_path):
    model = sem.LibreGTRSemModel("s", 19)
    path = _checkpoint(tmp_path, model)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR)
    assert loaded.task == "semantic" and loaded.nb_classes == 19
    assert loaded.names[0] == "road" and loaded.names[18] == "bicycle"
    assert loaded.input_size == (1024, 2048)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )

    loaded.model.window = 64
    image = np.random.default_rng(0).integers(0, 255, (48, 80, 3), dtype=np.uint8)
    result = loaded.predict(image, imgsz=(64, 128))[0]
    mask = result.semantic_mask.data
    assert tuple(mask.shape) == (48, 80)
    assert int(mask.max()) < 19

    with pytest.raises(Exception, match="semantic"):
        LibreGTR(str(path), size="s", device="cpu", task="detect")


def test_custom_class_count_rebuilds_the_head(tmp_path):
    model = sem.LibreGTRSemModel("s", 5)
    loaded = LibreYOLO(str(_checkpoint(tmp_path, model)), device="cpu")
    assert loaded.nb_classes == 5
    assert loaded.model.head.classifier.out_channels == 5


def test_download_urls_and_notice():
    url = LibreGTR.get_download_url("LibreGTRm-sem.pt")
    revision = LibreGTR.HF_TASK_REVISIONS[("m", "semantic")] or "main"
    assert url == (
        "https://huggingface.co/LibreYOLO/LibreGTRm-sem/resolve/"
        f"{revision}/LibreGTRm-sem.pt"
    )
    assert LibreGTR.get_download_url("LibreGTRm.pt").endswith("/LibreGTRm.pt")
    assert LibreGTR.detect_task_from_filename("LibreGTRx-sem.pt") == "semantic"
    assert LibreGTR.get_download_notice("LibreGTRs-sem.pt", url) is None


def _write_semantic_dataset(root, count=2):
    import yaml
    from PIL import Image

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        rng = np.random.default_rng(0)
        for i in range(count):
            image = rng.integers(0, 255, (64, 128, 3), dtype=np.uint8)
            mask = np.zeros((64, 128), dtype=np.uint8)
            mask[:, 64:] = 1
            mask[:4] = 255
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{i}.png")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "ignore_index": 255,
                "names": {0: "left", 1: "right"},
            }
        )
    )
    return data


def test_validation_and_short_training_run(tmp_path):
    data = _write_semantic_dataset(tmp_path)
    model = LibreGTR(None, size="s", nb_classes=2, device="cpu", task="semantic")
    model.model.window = 64
    # The canvas the trainer validates on; the default is 1024x2048.
    model.input_size = (64, 128)
    metrics = model.val(data=str(data), batch=2, device="cpu")
    assert 0.0 <= metrics["metrics/mIoU"] <= 1.0

    results = model.train(
        data=str(data),
        epochs=1,
        batch=2,
        imgsz=64,
        device="cpu",
        workers=0,
        warmup_iters=0,
        ema=False,
        project=str(tmp_path / "runs"),
        name="sem",
    )
    assert results["best_checkpoint"] or results["last_checkpoint"]
    assert model.task == "semantic" and model.nb_classes == 2
