"""Tests for the opt-in ``min_samples`` epoch-length floor (issue #768)."""

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from libreyolo.data.dataset import (
    DistributedWithReplacementSampler,
    create_dataloader,
)
from libreyolo.training.config import TrainConfig

pytestmark = pytest.mark.unit


# =============================================================================
# Default behavior: the floor is opt-in and off by default
# =============================================================================


def test_floor_off_by_default_keeps_sampler_types():
    """min_samples=0 (and omitted) builds the exact same loader as today."""
    dataset = [None] * 5

    default_loader = create_dataloader(dataset, batch_size=2, num_workers=0)
    explicit_loader = create_dataloader(
        dataset, batch_size=2, num_workers=0, min_samples=0
    )

    for loader in (default_loader, explicit_loader):
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.sampler.replacement is False
        assert len(loader.sampler) == 5
        assert loader.num_workers == 0
        assert len(loader) == 2  # drop_last kicks in: 5 >= batch_size

    sequential = create_dataloader(
        dataset, batch_size=2, num_workers=0, shuffle=False, min_samples=0
    )
    assert isinstance(sequential.sampler, SequentialSampler)


def test_floor_off_by_default_keeps_passed_sampler():
    dataset = [None] * 5
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True)

    loader = create_dataloader(
        dataset, batch_size=2, num_workers=0, sampler=sampler
    )

    assert loader.sampler is sampler


def test_min_samples_leq_dataset_len_is_a_noop():
    """A floor at or below the dataset length changes nothing."""
    dataset = [None] * 10

    for min_samples in (5, 10):
        loader = create_dataloader(
            dataset,
            batch_size=2,
            num_workers=2,
            min_samples=min_samples,
        )
        assert isinstance(loader.sampler, RandomSampler)
        assert loader.sampler.replacement is False
        assert len(loader.sampler) == 10
        assert loader.num_workers == 2


def test_train_config_default_and_validation():
    assert TrainConfig().min_samples == 0
    with pytest.raises(ValueError, match="min_samples"):
        TrainConfig(min_samples=-1)


# =============================================================================
# Active floor: single-process path
# =============================================================================


def test_active_floor_draws_exactly_min_samples():
    dataset = [None] * 7
    loader = create_dataloader(
        dataset, batch_size=10, num_workers=0, min_samples=50
    )

    sampler = loader.sampler
    assert isinstance(sampler, RandomSampler)
    assert sampler.replacement is True

    indices = list(iter(sampler))
    assert len(indices) == 50
    assert all(0 <= i < 7 for i in indices)
    # drop_last keeps its meaning against the floored epoch length.
    assert len(loader) == 5


def test_active_floor_epoch_draws_differ_and_are_seed_reproducible():
    dataset = [None] * 7
    loader = create_dataloader(
        dataset, batch_size=10, num_workers=0, min_samples=50
    )
    sampler = loader.sampler

    torch.manual_seed(0)
    first_epoch = list(iter(sampler))
    second_epoch = list(iter(sampler))
    assert first_epoch != second_epoch

    torch.manual_seed(0)
    assert list(iter(sampler)) == first_epoch


def test_active_floor_clamps_num_workers_to_dataset_len():
    dataset = [None] * 3
    loader = create_dataloader(
        dataset, batch_size=2, num_workers=8, min_samples=10
    )
    assert loader.num_workers == 3


def test_active_floor_respects_custom_non_distributed_sampler():
    """A caller-provided non-DistributedSampler is never swapped out."""
    dataset = [None] * 5
    sampler = SubsetRandomSampler([0, 1])

    loader = create_dataloader(
        dataset, batch_size=2, num_workers=0, sampler=sampler, min_samples=50
    )

    assert loader.sampler is sampler


# =============================================================================
# Active floor: DDP path
# =============================================================================


def test_active_floor_swaps_distributed_sampler():
    dataset = [None] * 7
    ddp_sampler = DistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True
    )

    loader = create_dataloader(
        dataset, batch_size=4, num_workers=0, sampler=ddp_sampler, min_samples=50
    )

    sampler = loader.sampler
    assert isinstance(sampler, DistributedWithReplacementSampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 1
    assert sampler.seed == ddp_sampler.seed
    assert len(sampler) == 25


def test_ddp_split_sums_to_min_samples_across_two_fake_ranks():
    dataset = [None] * 7
    samplers = [
        DistributedWithReplacementSampler(
            dataset, num_samples=50, num_replicas=2, rank=rank, seed=0
        )
        for rank in (0, 1)
    ]

    per_rank = [list(iter(s)) for s in samplers]
    assert [len(indices) for indices in per_rank] == [25, 25]
    assert sum(len(indices) for indices in per_rank) == 50
    for indices in per_rank:
        assert all(0 <= i < 7 for i in indices)

    # Both ranks shard one shared draw: interleaving the shards reproduces it.
    g = torch.Generator()
    g.manual_seed(0)
    shared = torch.randint(7, (50,), generator=g).tolist()
    assert per_rank[0] == shared[0:50:2]
    assert per_rank[1] == shared[1:50:2]


def test_ddp_odd_min_samples_rounds_up_per_rank():
    dataset = [None] * 7
    sampler = DistributedWithReplacementSampler(
        dataset, num_samples=51, num_replicas=2, rank=0, seed=0
    )
    assert len(sampler) == 26
    assert sampler.total_size == 52


def test_ddp_set_epoch_changes_draws_deterministically():
    dataset = [None] * 7
    sampler = DistributedWithReplacementSampler(
        dataset, num_samples=50, num_replicas=2, rank=0, seed=0
    )

    sampler.set_epoch(0)
    epoch0 = list(iter(sampler))
    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))
    assert epoch0 != epoch1

    sampler.set_epoch(0)
    assert list(iter(sampler)) == epoch0


def test_ddp_sampler_rejects_invalid_args():
    dataset = [None] * 7
    with pytest.raises(ValueError, match="num_samples"):
        DistributedWithReplacementSampler(
            dataset, num_samples=0, num_replicas=2, rank=0
        )
    with pytest.raises(ValueError, match="rank"):
        DistributedWithReplacementSampler(
            dataset, num_samples=10, num_replicas=2, rank=2
        )
