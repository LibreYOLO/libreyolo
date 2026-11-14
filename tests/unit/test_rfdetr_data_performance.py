import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from libreyolo.data.augment.constants import IMAGENET_MEAN, IMAGENET_STD
from libreyolo.data.augment.rfdetr import _normalize_rgb_image
from libreyolo.data.dataset import create_dataloader

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("shape", [(1, 1, 3), (31, 47, 3), (64, 64, 3)])
def test_normalization_is_bitwise_equal_and_does_not_mutate_input(
    normalize, dtype, shape
):
    image = (
        np.random.default_rng(31)
        .integers(0, 256, shape, dtype=np.uint8)
        .astype(dtype)[:, ::-1]
    )
    original = image.copy()
    expected = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    if normalize:
        expected = (expected - IMAGENET_MEAN) / IMAGENET_STD
    actual = _normalize_rgb_image(image, normalize)
    assert np.array_equal(actual, expected) and actual.flags.c_contiguous
    assert np.array_equal(image, original)


@pytest.mark.parametrize("workers", [0, 1])
def test_loader_controls_respect_worker_count(workers):
    dataset = TensorDataset(torch.zeros(8, 3), torch.zeros(8, 1))
    loader = create_dataloader(
        dataset,
        batch_size=2,
        num_workers=workers,
        persistent_workers=True,
        prefetch_factor=3,
    )
    assert loader.persistent_workers is bool(workers)
    assert loader.prefetch_factor == (3 if workers else None)


def test_stable_target_packing_preserves_holes_and_order():
    from libreyolo.models.rfdetr.trainer import _pack_valid_rfdetr_targets

    targets = torch.zeros(4, 8, 6)
    for i in range(4):
        for j in range(i + 1):
            targets[i, 2 * j] = torch.tensor([j, 10.0, 20.0, 3.0, 4.0, 0.1 * j])
    packed, counts = _pack_valid_rfdetr_targets(targets)
    assert counts == [1, 2, 3, 4]
    for i, n in enumerate(counts):
        expected = targets[i][(targets[i, :, 3] > 0) & (targets[i, :, 4] > 0)]
        torch.testing.assert_close(packed[i, :n], expected, rtol=0, atol=0)


def test_combined_loss_logging_preserves_raw_values():
    from types import SimpleNamespace

    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="detect")
    outputs = {
        "loss_ce": torch.tensor(1.25),
        "loss_ce_0": torch.tensor(2.25),
        "loss_bbox": torch.tensor(0.5),
        "loss_giou": 2.0,
    }
    total = torch.tensor(11.5, requires_grad=True)
    actual, components = trainer._training_loss_values(outputs, total)
    assert actual == 11.5
    assert (
        components
        == trainer.get_loss_components(outputs)
        == {"ce": 3.5, "bbox": 0.5, "giou": 2.0}
    )


@pytest.mark.parametrize("task", ["detect", "segment", "obb"])
def test_stock_no_aug_callback_allows_persistent_workers(task):
    from types import SimpleNamespace

    from libreyolo.models.dfine.transforms import DFINEPassThroughDataset
    from libreyolo.models.rfdetr.config import RFDETRConfig
    from libreyolo.models.rfdetr.seg_transforms import (
        RFDETRDetTransform,
        RFDETRSegPassThroughDataset,
        RFDETRSegTransform,
    )
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    transform_cls = RFDETRSegTransform if task == "segment" else RFDETRDetTransform
    wrapper_cls = RFDETRSegPassThroughDataset if task == "segment" else DFINEPassThroughDataset
    transform = transform_cls(imgsz=64, multi_scale=False, patch_size=16, num_windows=2)
    dataset = wrapper_cls(TensorDataset(torch.zeros(2, 3)), (64, 64), preproc=transform)
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task=task)
    trainer.config = RFDETRConfig(workers=2, persistent_workers=True, epochs=4, no_aug_epochs=2)
    trainer.train_loader = SimpleNamespace(dataset=dataset, num_workers=2, persistent_workers=True, collate_fn=None)
    # Both the epoch boundary and a resumed run call this same hook.
    trainer.on_mosaic_disable()
    trainer.on_mosaic_disable()
    assert dataset.preproc is transform


def test_custom_dataset_mutation_still_rejected_for_persistent_workers():
    from types import SimpleNamespace

    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    called = []
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.wrapper_model = SimpleNamespace(task="detect")
    dataset = SimpleNamespace(close_mosaic=lambda: called.append(True))
    trainer.train_loader = SimpleNamespace(dataset=dataset, num_workers=2, persistent_workers=True, collate_fn=None)
    with pytest.raises(RuntimeError, match="persistent_workers"):
        trainer.on_mosaic_disable()
    assert not called
