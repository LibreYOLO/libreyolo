"""PyTorch's CPU pool is kept within the cgroup CPU quota."""

import pytest
import torch

from libreyolo.training import cpu_threads

pytestmark = pytest.mark.unit


def _cgroup(tmp_path, v2=None, v1=None):
    if v2 is not None:
        (tmp_path / "cpu.max").write_text(v2)
    if v1 is not None:
        (tmp_path / "cpu").mkdir()
        (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text(str(v1[0]))
        (tmp_path / "cpu" / "cpu.cfs_period_us").write_text(str(v1[1]))
    return tmp_path


@pytest.mark.parametrize(
    "v2,v1,expected",
    [
        ("3071999 100000\n", None, 31),
        ("max 100000\n", None, None),
        (None, (3060000, 100000), 31),
        (None, (-1, 100000), None),
        ("50000 100000", None, 1),
        ("150000 100000", None, 2),
        (None, None, None),
    ],
)
def test_cpu_quota(tmp_path, v2, v1, expected):
    assert cpu_threads.cpu_quota(_cgroup(tmp_path, v2, v1)) == expected


@pytest.fixture
def pool(monkeypatch):
    state = {"n": 128}
    monkeypatch.setattr(torch, "get_num_threads", lambda: state["n"])
    monkeypatch.setattr(torch, "set_num_threads", lambda n: state.update(n=n))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(cpu_threads.os, "sched_getaffinity", lambda pid: set(range(256)), raising=False)
    return state


def test_caps_an_oversized_pool_to_the_quota_and_restores_it(tmp_path, pool):
    previous = cpu_threads.cap_torch_threads(_cgroup(tmp_path, "3000000 100000"))
    assert previous == 128 and pool["n"] == 30
    cpu_threads.restore_torch_threads(previous)
    assert pool["n"] == 128


@pytest.mark.parametrize("env", ["LOCAL_WORLD_SIZE", "WORLD_SIZE"])
def test_quota_is_split_across_local_ranks(tmp_path, pool, monkeypatch, env):
    # torchrun sets LOCAL_WORLD_SIZE; LibreYOLO's own spawn only WORLD_SIZE.
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv(env, "4")
    cpu_threads.cap_torch_threads(_cgroup(tmp_path, "3000000 100000"))
    assert pool["n"] == 7


def test_leaves_a_pool_within_the_quota(tmp_path, pool):
    pool["n"] = 16
    assert cpu_threads.cap_torch_threads(_cgroup(tmp_path, "3071999 100000")) is None
    assert pool["n"] == 16


def test_unlimited_host_is_untouched(tmp_path, pool):
    assert cpu_threads.cap_torch_threads(_cgroup(tmp_path, "max 100000")) is None
    assert pool["n"] == 128


def test_explicit_omp_setting_wins(tmp_path, pool, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    assert cpu_threads.cap_torch_threads(_cgroup(tmp_path, "3071999 100000")) is None
    assert pool["n"] == 128


def test_affinity_mask_also_limits(tmp_path, pool, monkeypatch):
    monkeypatch.setattr(cpu_threads.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
    assert cpu_threads.cap_torch_threads(_cgroup(tmp_path, "max 100000")) == 128
    assert pool["n"] == 8


def test_train_restores_the_pool_even_when_setup_fails(monkeypatch):
    from types import SimpleNamespace

    from libreyolo.training.trainer import BaseTrainer

    calls = []
    monkeypatch.setattr(cpu_threads, "cap_torch_threads", lambda: calls.append("cap") or 64)
    monkeypatch.setattr(cpu_threads, "restore_torch_threads", lambda n: calls.append(("restore", n)))

    def failing_setup():
        raise RuntimeError("bad dataset")

    host = SimpleNamespace(
        setup=failing_setup,
        _build_train_exception_event=lambda exc, elapsed: None,
        _dispatch_artifact_callbacks=lambda *a: None,
        callbacks=SimpleNamespace(on_train_exception=lambda event: None),
    )
    with pytest.raises(RuntimeError, match="bad dataset"):
        BaseTrainer.train(host)
    assert calls == ["cap", ("restore", 64)]
    assert host._threads_before_cap is None
