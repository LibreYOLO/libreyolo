"""Trainer smoke tests for native DEIMv2 training."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibreDEIMv2

pytestmark = [pytest.mark.unit, pytest.mark.deimv2]


def test_deimv2_trainer_applies_hgnetv2_size_recipe():
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer

    wrapper = LibreDEIMv2(None, size="atto", device="cpu")
    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="atto",
        num_classes=80,
        data=None,
        device="cpu",
        ema=False,
        eval_interval=-1,
    )

    assert trainer.config.imgsz == 320
    assert trainer.config.epochs == 500
    assert trainer.config.lr0 == 2e-3
    assert trainer.config.backbone_lr_mult == 0.5
    assert trainer.config.losses == ("mal", "boxes")
    assert trainer.config.use_uni_set is False
    assert trainer.config.matcher_change_epoch == 450
    assert trainer.config.warmup_iters == 4000
    assert trainer.config.sanitize_min_size == 12
    assert trainer.effective_lr == pytest.approx(2e-3)

    short_finetune = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="atto",
        num_classes=80,
        data=None,
        device="cpu",
        epochs=20,
        ema=False,
        eval_interval=-1,
    )
    assert short_finetune.config.warmup_iters is None


def test_deimv2_dino_optimizer_groups_dinov3_only_as_backbone_lr():
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer

    wrapper = LibreDEIMv2(None, size="s", device="cpu")
    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data=None,
        device="cpu",
        ema=False,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()

    name_by_id = {id(p): n for n, p in wrapper.model.named_parameters()}
    group_by_name = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            group_by_name[name_by_id[id(param)]] = group

    assert group_by_name["backbone.dinov3._model.patch_embed.proj.weight"][
        "lr_mult"
    ] == pytest.approx(0.05)
    assert group_by_name["backbone.dinov3._model.blocks.0.norm1.weight"][
        "weight_decay"
    ] == 0.0

    sta_weight = next(
        name
        for name in group_by_name
        if name.startswith("backbone.sta.") and name.endswith(".weight")
    )
    assert group_by_name[sta_weight]["lr_mult"] == 1.0


def test_deimv2_reg_max_override_fails_fast():
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer

    wrapper = LibreDEIMv2(None, size="s", device="cpu")
    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="s",
        num_classes=80,
        data=None,
        device="cpu",
        ema=False,
        eval_interval=-1,
        reg_max=16,
    )

    with pytest.raises(ValueError, match="reg_max must match"):
        trainer.on_setup()


def test_deimv2_train_dataset_uses_original_yolo_image_and_boxes(tmp_path):
    from PIL import Image

    from libreyolo.data.dataset import YOLODataset
    from libreyolo.models.deimv2.transforms import DEIMTrainTransform

    img_path = tmp_path / "image.jpg"
    label_path = tmp_path / "image.txt"
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(img_path)
    label_path.write_text("0 0.5 0.5 0.5 0.5\n")

    preproc = DEIMTrainTransform(imgsz=320, strong_augs=False)
    dataset = YOLODataset(
        img_files=[img_path],
        label_files=[label_path],
        img_size=(320, 320),
        preproc=preproc,
    )

    img, label, img_info, _ = dataset.pull_item(0)

    assert img.shape[:2] == (10, 20)
    assert img_info == (10, 20)
    np.testing.assert_allclose(label[0, :4], [5.0, 2.5, 15.0, 7.5])


def test_deimv2_checkpoint_preserves_train_and_ema_state(tmp_path):
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer
    from libreyolo.training.ema import ModelEMA

    wrapper = LibreDEIMv2(None, size="atto", device="cpu")
    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="atto",
        num_classes=80,
        data=None,
        device="cpu",
        ema=True,
        eval_interval=-1,
    )
    trainer.save_dir = tmp_path
    trainer.optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=1e-4)
    trainer.ema_model = ModelEMA(wrapper.model)
    trainer.ema_model.updates = 7

    trainer._save_checkpoint(
        0,
        loss=1.0,
        val_metrics={"mAP50_95": 0.42, "mAP50": 0.5},
    )

    ckpt = torch.load(tmp_path / "weights" / "best.pt", map_location="cpu")
    assert ckpt["best_mAP50_95"] == pytest.approx(0.42)
    assert ckpt["best_mAP50"] == pytest.approx(0.5)
    assert ckpt["best_metric_key"] == "metrics/mAP50-95"
    assert ckpt["best_epoch"] == 1
    assert "train_model" in ckpt
    assert "ema" in ckpt
    assert ckpt["ema_updates"] == 7

    trainer._save_checkpoint(
        1,
        loss=0.9,
        val_metrics={
            "mAP50_95": 0.10,
            "mAP50": 0.6,
            "best_metric": 0.50,
            "best_metric_key": "metrics/mAP50-95(M)",
        },
    )

    ckpt = torch.load(tmp_path / "weights" / "best.pt", map_location="cpu")
    assert ckpt["best_mAP50_95"] == pytest.approx(0.50)
    assert ckpt["best_metric_key"] == "metrics/mAP50-95(M)"
    assert ckpt["best_epoch"] == 2


def test_resume_resets_best_metric_when_metric_key_changes(tmp_path, caplog):
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer

    wrapper = LibreDEIMv2(None, size="atto", device="cpu")
    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="atto",
        num_classes=80,
        data=None,
        device="cpu",
        ema=False,
        eval_interval=-1,
    )
    trainer.best_metric_key = "metrics/mAP50-95(M)"

    checkpoint_path = tmp_path / "detect_best.pt"
    torch.save(
        {
            "epoch": 3,
            "model": wrapper.model.state_dict(),
            "best_mAP50_95": 0.65,
            "best_mAP50": 0.80,
            "best_epoch": 2,
            "best_metric_key": "metrics/mAP50-95(B)",
        },
        checkpoint_path,
    )

    with caplog.at_level("WARNING"):
        trainer.resume(str(checkpoint_path))

    assert "differs from current key" in caplog.text
    assert trainer.start_epoch == 4
    assert trainer.best_mAP50_95 == 0.0
    assert trainer.best_mAP50 == 0.0
    assert trainer.best_epoch == 0


def test_deimv2_trainer_target_translation_smoke():
    from libreyolo.models.deimv2.trainer import DEIMv2Trainer

    wrapper = LibreDEIMv2(None, size="atto", device="cpu")
    wrapper.model.train()

    trainer = DEIMv2Trainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="atto",
        num_classes=80,
        data=None,
        device="cpu",
        ema=False,
        eval_interval=-1,
        batch=1,
    )
    trainer.on_setup()

    imgs = torch.randn(1, 3, 320, 320)
    targets = torch.zeros(1, 120, 5)
    targets[0, 0] = torch.tensor([1.0, 160.0, 160.0, 40.0, 60.0])

    out = trainer.on_forward(imgs, targets)

    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"])
    assert out["total_loss"].item() > 0
    for prefix in ("loss_mal", "loss_bbox", "loss_giou"):
        assert any(k == prefix or k.startswith(prefix + "_") for k in out), (
            f"no key matching {prefix}* in output: {sorted(out)}"
        )

    out["total_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in wrapper.model.encoder.parameters()
    )
