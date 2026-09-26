"""CPU contracts for GTR instance segmentation (random weights, fast)."""

import random

import numpy as np
import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.nn import LibreGTRModel
from libreyolo.models.gtr.seg import SEG_MASK_DOWNSAMPLE_RATIO, SEG_HEAD_PREFIX
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


def _seg_model(nc=80, size=160):
    return LibreGTRModel(
        "s",
        nc,
        eval_spatial_size=(size, size),
        mask_downsample_ratio=SEG_MASK_DOWNSAMPLE_RATIO,
    )


def _seg_checkpoint(path, nc=2):
    model = _seg_model(nc, 640)
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=nc,
        names={i: f"c{i}" for i in range(nc)},
        task="segment",
        imgsz=640,
    )
    torch.save(ckpt, path)
    return model


def test_mask_head_emits_quarter_resolution_logits():
    model = _seg_model().eval()
    assert model.has_mask_head
    assert not LibreGTRModel("s", 80).has_mask_head
    with torch.no_grad():
        out = model(torch.randn(1, 3, 160, 160))
    assert set(out) == {"pred_logits", "pred_boxes", "pred_masks"}
    assert out["pred_masks"].shape == (1, 300, 40, 40)


def test_task_and_size_detection_keep_detect_and_segment_apart():
    seg = _seg_model().state_dict()
    det = LibreGTRModel("s", 80).state_dict()
    assert any(k.startswith(SEG_HEAD_PREFIX) for k in seg)
    assert LibreGTR.can_load(seg) and LibreGTR.can_load(det)
    assert LibreGTR.detect_checkpoint_task(seg) == "segment"
    assert LibreGTR.detect_checkpoint_task(det) is None
    assert LibreGTR.detect_size(seg) == "s"

    from libreyolo.models.ec.model import LibreEC

    assert not LibreEC.can_load(seg)


def test_seg_filename_download_url_is_pinned():
    assert LibreGTR.detect_task_from_filename("LibreGTRx-seg.pt") == "segment"
    assert LibreGTR.detect_size_from_filename("LibreGTRx-seg.pt") == "x"
    assert "LibreGTRs/resolve/" in LibreGTR.get_download_url("LibreGTRs.pt")
    revision = LibreGTR.HF_TASK_REVISIONS[("s", "segment")]
    assert revision
    assert LibreGTR.get_download_url("LibreGTRs-seg.pt") == (
        f"https://huggingface.co/LibreYOLO/LibreGTRs-seg/resolve/{revision}/"
        "LibreGTRs-seg.pt"
    )


def test_seg_checkpoint_loads_strictly_and_predicts_masks(tmp_path):
    path = tmp_path / "LibreGTRs-seg.pt"
    model = _seg_checkpoint(path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR) and loaded.task == "segment"
    assert loaded.model.has_mask_head
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )

    image = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    result = loaded.predict(image, conf=0.0, max_det=5)
    assert len(result.boxes) == 5
    assert result.masks.data.shape == (5, 120, 200)

    # A detect-only state dict cannot be loaded into the seg head strictly.
    ckpt = torch.load(path, weights_only=False)
    ckpt["model"] = {
        k: v for k, v in ckpt["model"].items() if not k.startswith(SEG_HEAD_PREFIX)
    }
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError):
        LibreGTR(str(path), size="s", task="segment", device="cpu")


def test_postprocess_thresholds_upsampled_mask_logits_at_zero():
    model = LibreGTR(None, size="s", task="segment", device="cpu")
    logits = torch.full((1, 3, 2), -10.0)
    logits[0, 0, 1] = 5.0
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]] * 3])
    masks = torch.full((1, 3, 4, 4), -1.0)
    masks[0, 0, :, :2] = 1.0
    out = model._postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_masks": masks},
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=(8, 8),
    )
    assert out["num_detections"] == 1 and int(out["classes"][0]) == 1
    mask = np.asarray(out["masks"][0])
    assert mask[:, :3].all() and not mask[:, 5:].any()


def test_seg_training_step_and_lora_recipe():
    from libreyolo.models.gtr.seg_trainer import GTRSegConfig, GTRSegTrainer

    trainer = object.__new__(GTRSegTrainer)
    trainer.config = GTRSegConfig(num_classes=2)
    trainer.device = torch.device("cpu")
    trainer.model = _seg_model(2).train()
    trainer.criterion = trainer.build_criterion()
    targets = torch.zeros(1, 4, 5)
    targets[0, 0] = torch.tensor([1.0, 80.0, 80.0, 60.0, 40.0])
    masks = torch.zeros(1, 1, 160, 160, dtype=torch.bool)
    masks[0, 0, 60:100, 50:110] = True
    out = trainer.on_forward(torch.randn(1, 3, 160, 160), targets, masks)
    assert torch.isfinite(out["total_loss"])
    assert out["loss_mask_ce"] > 0 and out["loss_mask_dice"] > 0
    out["total_loss"].backward()
    head = trainer.model.decoder.decoder.segmentation_head
    assert head.spatial_features_proj.weight.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="Mosaic"):
        GTRSegConfig(mosaic_prob=0.5)

    pytest.importorskip("peft")
    from libreyolo.training.lora import apply_lora_to_gtr

    model = _seg_model(2)
    apply_lora_to_gtr(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith(SEG_HEAD_PREFIX) for n in trainable)


def _write_seg_dataset(root, count=4):
    import cv2
    import yaml

    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        image = rng.integers(0, 255, (96 + 16 * i, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"{i}.jpg"), image)
        (root / "labels" / f"{i}.txt").write_text(
            f"{i % 2} 0.3 0.3 0.7 0.3 0.7 0.6 0.3 0.6\n"
        )
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images", "val": "images", "names": ["a", "b"]}
        )
    )
    return data


def test_seg_training_runs_end_to_end(tmp_path):
    data = _write_seg_dataset(tmp_path)
    model = LibreGTR(None, size="s", task="segment", device="cpu")
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
        name="seg",
    )
    assert results["best_checkpoint"]
    reloaded = LibreYOLO(results["best_checkpoint"], device="cpu")
    assert reloaded.task == "segment" and reloaded.nb_classes == 2


def test_detect_weights_initialize_segment_only_as_explicit_transfer(tmp_path):
    # Upstream initializes GTRSeg from the whole COCO detector.
    detector = LibreGTRModel("s", 2)
    path = tmp_path / "LibreGTRs.pt"
    torch.save(
        wrap_libreyolo_checkpoint(
            detector.state_dict(),
            model_family="gtr",
            size="s",
            nc=2,
            names={0: "a", 1: "b"},
            task="detect",
            imgsz=640,
        ),
        path,
    )
    with pytest.raises(RuntimeError, match="no mask head"):
        LibreGTR(str(path), size="s", task="segment", device="cpu")
    model = LibreGTR(
        str(path),
        size="s",
        task="segment",
        device="cpu",
        allow_detect_to_segment_transfer=True,
    )
    loaded = model.model.state_dict()
    for key, value in detector.state_dict().items():
        torch.testing.assert_close(value, loaded[key], rtol=0, atol=0)
    assert model.model.has_mask_head

    # The transfer stays strict for every non-mask key.
    ckpt = torch.load(path, weights_only=False)
    ckpt["model"].pop("encoder.stages.0.1.weight")
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError):
        LibreGTR(
            str(path),
            size="s",
            task="segment",
            device="cpu",
            allow_detect_to_segment_transfer=True,
        )
