"""CPU contracts for GTR LoRA fine-tuning, Mosaic and batch MixUp."""

import random

import numpy as np
import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
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


class _Images:
    """Minimal ``pull_item`` source: original-size BGR images, xyxy+class labels."""

    def __init__(self, sizes):
        self.items = []
        for i, (h, w) in enumerate(sizes):
            image = np.full((h, w, 3), 40 * (i + 1), dtype=np.uint8)
            label = np.array(
                [[w * 0.25, h * 0.25, w * 0.75, h * 0.75, i % 2]], np.float32
            )
            self.items.append((image, label))

    def __len__(self):
        return len(self.items)

    def pull_item(self, idx):
        image, label = self.items[idx]
        return image.copy(), label.copy(), image.shape[:2], idx


def _mosaic_dataset(prob=1.0, epochs=6, imgsz=160):
    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.trainer import GTRTrainer

    trainer = object.__new__(GTRTrainer)
    trainer.config = GTRConfig(imgsz=imgsz, mosaic_prob=prob)
    transform, dataset_cls = trainer.create_transforms()
    dataset = dataset_cls(
        dataset=_Images([(100, 200), (120, 90), (80, 80), (200, 150)]),
        img_size=(imgsz, imgsz),
        preproc=transform,
        mosaic_prob=prob,
    )
    dataset.set_stop_epoch(28)
    dataset.set_mosaic_epochs(epochs)
    return dataset


def test_recipe_defaults_follow_upstream_finetune_configs():
    from libreyolo.models.gtr.config import GTRConfig

    config = GTRConfig()
    assert (config.mosaic_prob, config.mixup_prob, config.mosaic_epochs) == (
        0.5,
        0.5,
        6,
    )
    assert (config.degrees, config.translate) == (10.0, 0.1)
    assert tuple(config.mosaic_scale) == (0.5, 1.5)
    with pytest.raises(ValueError, match="mixup_prob"):
        GTRConfig(mixup_prob=1.5)
    with pytest.raises(ValueError, match="mosaic_epochs"):
        GTRConfig(mosaic_epochs=-1)


def test_mosaic_pastes_four_half_size_tiles_with_shifted_boxes():
    random.seed(0)
    dataset = _mosaic_dataset()
    image, label = dataset._build_mosaic(*dataset.dataset.pull_item(0)[:2])
    # The 100x200 source is resized to an 80px shorter side: 80x160.
    assert image.shape[0] >= 160 and image.shape[1] >= 320
    assert len(label) == 4
    np.testing.assert_allclose(label[0, :4], [40, 20, 120, 60])
    assert (label[:, 2] <= image.shape[1]).all() and (
        label[:, 3] <= image.shape[0]
    ).all()


def test_mosaic_samples_are_square_and_keep_valid_labels():
    random.seed(1)
    torch.manual_seed(1)
    dataset = _mosaic_dataset()
    for idx in range(len(dataset)):
        image, labels, *_ = dataset[idx]
        assert image.shape == (3, 160, 160)
        valid = labels[(labels[:, 3] > 0) & (labels[:, 4] > 0)]
        assert len(valid) >= 1
        assert (valid[:, 1:] >= 0).all() and (valid[:, 1:] <= 160).all()


def test_mosaic_window_and_cache_bound():
    from libreyolo.models.gtr import transforms

    dataset = _mosaic_dataset(epochs=2)
    dataset.set_epoch(1)
    assert dataset._mosaic_active()
    dataset.set_epoch(2)
    assert not dataset._mosaic_active()

    dataset = _mosaic_dataset()
    dataset._stop_epoch = 3
    dataset.set_epoch(3)
    assert not dataset._mosaic_active()

    dataset = _mosaic_dataset()
    image, label = dataset.dataset.pull_item(0)[:2]
    for _ in range(transforms.MOSAIC_CACHE_SIZE + 10):
        dataset._build_mosaic(image, label)
    assert len(dataset._cache) == transforms.MOSAIC_CACHE_SIZE

    dataset = _mosaic_dataset(prob=0.0)
    assert not dataset._mosaic_active()


def test_mixup_blends_neighbours_and_keeps_both_label_sets():
    from libreyolo.models.gtr.transforms import mixup_batch

    imgs = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
    labels = torch.zeros(2, 3, 5)
    labels[0, 0] = torch.tensor([0, 1, 1, 1, 1])
    labels[1, :2] = torch.tensor([[1, 2, 2, 1, 1], [1, 3, 3, 1, 1]])
    mixed, merged = mixup_batch(imgs, labels, beta=0.5)
    torch.testing.assert_close(mixed, torch.full_like(imgs, 0.5))
    # Image 0 now carries its own box plus image 1's two boxes.
    torch.testing.assert_close(merged[0], torch.cat([labels[0, :1], labels[1, :2]]))
    torch.testing.assert_close(merged[1], torch.cat([labels[1, :2], labels[0, :1]]))


def test_mixup_widens_padding_instead_of_dropping_labels():
    from libreyolo.models.gtr.transforms import mixup_batch

    imgs = torch.zeros(2, 3, 4, 4)
    labels = torch.zeros(2, 3, 5)
    labels[:, :, 3:] = 1.0  # every slot holds a real box
    labels[1, :, 0] = 1.0
    _, merged = mixup_batch(imgs, labels, beta=0.5)
    assert merged.shape == (2, 6, 5)
    assert ((merged[..., 3] > 0) & (merged[..., 4] > 0)).sum() == 12


def test_resume_settings_restore_saved_config_before_overrides(monkeypatch):
    from libreyolo.models.gtr.sem_trainer import GTRSemConfig

    model = object.__new__(LibreGTR)
    model.model_path = "last.pt"
    monkeypatch.setattr(
        LibreGTR,
        "_checkpoint_train_config",
        lambda self, path: {"epochs": 9, "lr0": 0.002, "batch": 4, "size": "x"},
    )
    path, settings = model._resume_settings(
        True, GTRSemConfig, {"batch": 2, "data": "d.yaml", "lr0": None}
    )
    assert path == "last.pt"
    assert settings == {"epochs": 9, "lr0": 0.002, "batch": 2, "data": "d.yaml"}
    assert model._resume_settings(False, GTRSemConfig, {"lr0": None}) == (None, {})


def test_mixup_collate_is_epoch_gated():
    from libreyolo.data.dataset import yolox_collate_fn
    from libreyolo.models.gtr.transforms import GTRMixUpCollate

    batch = [
        (np.full((3, 4, 4), v, np.float32), np.zeros((3, 5), np.float32), (4, 4), i)
        for i, v in enumerate((0.0, 1.0))
    ]
    collate = GTRMixUpCollate(yolox_collate_fn, mixup_prob=1.0, mixup_epochs=1)
    imgs, *_ = collate(batch)
    assert 0 < imgs[0].mean() < 1
    collate.set_epoch(1)
    imgs, *_ = collate(batch)
    assert imgs[0].mean() == 0


def test_lora_adapts_backbone_attention_and_decoder_only():
    pytest.importorskip("peft")
    from libreyolo.training.lora import apply_lora_to_gtr, merge_lora_adapters

    torch.manual_seed(0)
    model = LibreGTRModel("s", 3).eval()
    dense = {k: v.clone() for k, v in model.state_dict().items()}
    image = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        reference = model(image)

    apply_lora_to_gtr(model)
    adapted = {
        name.removesuffix(".lora_A.default")
        for name, _ in model.named_modules()
        if name.endswith(".lora_A.default")
    }
    # 12 backbone blocks x q/k/v, plus 4 decoder layers x 5 Linears.
    assert len(adapted) == 12 * 3 + 4 * 5
    assert all(
        name.endswith(("q_proj", "k_proj", "v_proj"))
        for name in adapted
        if name.startswith("backbone.")
    )
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n for n in trainable if n.startswith("backbone.backbone."))
    assert all(
        "lora_" in n for n in trainable if n.startswith("decoder.decoder.layers.")
    )
    assert any(n.startswith("encoder.") for n in trainable)
    assert any(n.startswith("decoder.dec_score_head.") for n in trainable)

    # LoRA B starts at zero, so the adapted graph matches the dense one.
    with torch.no_grad():
        adapted_out = model(image)
    torch.testing.assert_close(adapted_out["pred_logits"], reference["pred_logits"])

    # Training moves the adapters; merging folds them into dense weights.
    model.train()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    model.eval()
    with torch.no_grad():
        tuned = model(image)["pred_logits"]
    assert merge_lora_adapters(model) == len(adapted)
    assert set(model.state_dict()) == set(dense)
    with torch.no_grad():
        torch.testing.assert_close(
            model(image)["pred_logits"], tuned, rtol=1e-4, atol=1e-4
        )


def test_lora_checkpoint_reloads_with_adapters(tmp_path):
    pytest.importorskip("peft")
    from libreyolo.training.lora import apply_lora_to_gtr, module_has_lora

    model = LibreGTRModel("s", 2)
    apply_lora_to_gtr(model)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=2,
        names={0: "a", 1: "b"},
        task="detect",
        imgsz=160,
    )
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR) and loaded.size == "s"
    assert module_has_lora(loaded.model)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )


def _write_dataset(root, count=4):
    import cv2
    import yaml

    (root / "images").mkdir()
    (root / "labels").mkdir()
    rng = np.random.default_rng(0)
    for i in range(count):
        image = rng.integers(0, 255, (96 + 16 * i, 128, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "images" / f"{i}.jpg"), image)
        (root / "labels" / f"{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.4 0.3\n")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "images", "val": "images", "names": ["a", "b"]}
        )
    )
    return data


def test_lora_training_with_mosaic_and_mixup_runs_end_to_end(tmp_path):
    pytest.importorskip("peft")
    from libreyolo.training.lora import module_has_lora

    data = _write_dataset(tmp_path)
    model = LibreGTR(None, size="s", nb_classes=80, device="cpu")
    frozen = model.model.backbone.backbone._model.blocks[0].attn.q_proj.weight.clone()
    results = model.train(
        data=str(data),
        epochs=2,
        batch=2,
        imgsz=160,
        device="cpu",
        workers=0,
        lora=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
        mosaic_epochs=1,
        warmup_iters=0,
        ema=False,
        project=str(tmp_path / "runs"),
        name="lora",
    )
    assert results["best_checkpoint"]
    assert module_has_lora(model.model)
    base = model.model.backbone.backbone._model.blocks[0].attn.q_proj.base_layer.weight
    torch.testing.assert_close(base, frozen, rtol=0, atol=0)
    prediction = model.predict(np.zeros((96, 128, 3), np.uint8), imgsz=160)
    assert prediction is not None

    exported = model.export(format="onnx", imgsz=160, simplify=False)
    assert exported
