"""Guard tests: main-process dataset mutations must not silently miss workers.

Mid-training mutations (``close_mosaic`` for the no-aug tail, the DETR-style
``set_epoch`` gating) only reach dataloader workers because the affected
loaders run ``persistent_workers=False`` and respawn workers each epoch. With
``persistent_workers=True`` the forked workers keep their stale dataset copy
and the mutation is a silent no-op. ``ensure_mutation_reaches_workers`` turns
that silent no-op into a loud RuntimeError.

All loaders here use ``num_workers=0`` or plain fakes, so no subprocesses are
spawned and no ``__main__`` guard is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import DataLoader, Dataset

from libreyolo.training.trainer import BaseTrainer, ensure_mutation_reaches_workers

pytestmark = pytest.mark.unit


class _MosaicDataset(Dataset):
    """Minimal dataset implementing the ``close_mosaic`` mutation hook."""

    def __init__(self):
        self.mosaic_closed = False

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return idx

    def close_mosaic(self):
        self.mosaic_closed = True


class _PlainDataset(Dataset):
    """Dataset with no mutation hooks (the pose case)."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return idx


def _fake_loader(dataset, num_workers, persistent_workers):
    """A crafted loader stand-in: real DataLoader refuses persistent_workers
    without workers, and spawning real workers is not what these tests are
    about."""
    return SimpleNamespace(
        dataset=dataset,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


# ---------------------------------------------------------------------------
# ensure_mutation_reaches_workers
# ---------------------------------------------------------------------------


def test_guard_raises_on_persistent_workers_with_hook():
    ds = _MosaicDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=True)
    with pytest.raises(RuntimeError, match="persistent_workers"):
        ensure_mutation_reaches_workers(loader, ds, "close_mosaic")


def test_guard_message_names_target_and_hook():
    ds = _MosaicDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=True)
    with pytest.raises(RuntimeError, match=r"_MosaicDataset\.close_mosaic"):
        ensure_mutation_reaches_workers(loader, ds, "close_mosaic")


def test_guard_silent_without_workers():
    ds = _MosaicDataset()
    loader = _fake_loader(ds, num_workers=0, persistent_workers=True)
    ensure_mutation_reaches_workers(loader, ds, "close_mosaic")


def test_guard_silent_with_non_persistent_workers():
    ds = _MosaicDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=False)
    ensure_mutation_reaches_workers(loader, ds, "close_mosaic")


def test_guard_silent_when_target_lacks_hook():
    # Persistent workers are fine when the dataset has no mutation hook:
    # today's pose loaders (persistent_workers=True) must keep working.
    ds = _PlainDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=True)
    ensure_mutation_reaches_workers(loader, ds, "close_mosaic")
    ensure_mutation_reaches_workers(loader, ds, "set_epoch")


def test_guard_silent_on_none_loader_or_target():
    ds = _MosaicDataset()
    ensure_mutation_reaches_workers(None, ds, "close_mosaic")
    loader = _fake_loader(None, num_workers=2, persistent_workers=True)
    ensure_mutation_reaches_workers(loader, None, "close_mosaic")


def test_guard_checks_arbitrary_target_such_as_collate():
    # deim/dfine also mutate the collate object per epoch; collate runs in
    # the workers too, so the same guard applies.
    class _Collate:
        def set_epoch(self, epoch):
            pass

    cf = _Collate()
    loader = _fake_loader(_PlainDataset(), num_workers=2, persistent_workers=True)
    with pytest.raises(RuntimeError, match=r"_Collate\.set_epoch"):
        ensure_mutation_reaches_workers(loader, cf, "set_epoch")


def test_guard_silent_on_real_inline_dataloader():
    ds = _MosaicDataset()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    ensure_mutation_reaches_workers(loader, ds, "close_mosaic")


# ---------------------------------------------------------------------------
# BaseTrainer.on_mosaic_disable behavior
# ---------------------------------------------------------------------------


def _call_on_mosaic_disable(loader):
    """Invoke the base hook against a bare fake trainer (no full setup)."""
    fake_trainer = SimpleNamespace(train_loader=loader)
    BaseTrainer.on_mosaic_disable(fake_trainer)


def test_on_mosaic_disable_still_closes_mosaic_inline():
    ds = _MosaicDataset()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    _call_on_mosaic_disable(loader)
    assert ds.mosaic_closed


def test_on_mosaic_disable_still_noop_without_hook():
    ds = _PlainDataset()
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    _call_on_mosaic_disable(loader)


def test_on_mosaic_disable_raises_before_mutating_persistent_loader():
    ds = _MosaicDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=True)
    with pytest.raises(RuntimeError, match="persistent_workers"):
        _call_on_mosaic_disable(loader)
    # The failed close must not have half-applied in the main process.
    assert not ds.mosaic_closed


def test_on_mosaic_disable_allows_persistent_loader_without_hook():
    # The live pose configuration: persistent workers, dataset without
    # close_mosaic. Must stay a silent no-op.
    ds = _PlainDataset()
    loader = _fake_loader(ds, num_workers=2, persistent_workers=True)
    _call_on_mosaic_disable(loader)
