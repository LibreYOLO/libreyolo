"""Parity and portability tests for the issue #763 training-speed quick wins.

Covers: the broadcast-L1-equals-cdist bitwise identity, every ported matcher
running cdist-free (including empty targets), and the shared fused-optimizer
construction helper's CUDA-only gate, which keeps non-CUDA machines
byte-identical to stock construction.
"""

from __future__ import annotations

import pytest
import torch

import libreyolo.training.optim as optim_mod
from libreyolo.training.optim import build_optimizer

pytestmark = pytest.mark.unit

WEIGHTS = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}


# ---------------------------------------------------------------------------
# Broadcast L1 == cdist, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,m", [(300, 20), (300, 0), (7, 3), (1, 1)])
def test_broadcast_l1_is_bitwise_identical_to_cdist(n, m):
    generator = torch.Generator().manual_seed(763)
    for _ in range(10):
        a = torch.rand(n, 4, generator=generator)
        b = torch.rand(m, 4, generator=generator)
        assert torch.equal(
            torch.cdist(a, b, p=1),
            (a[:, None, :] - b[None, :, :]).abs().sum(-1),
        )


# ---------------------------------------------------------------------------
# Every ported matcher runs cdist-free, on normal and empty targets
# ---------------------------------------------------------------------------


def _outputs(bs, num_queries, num_classes, generator):
    return {
        "pred_logits": torch.randn(bs, num_queries, num_classes, generator=generator) * 3,
        "pred_boxes": torch.rand(bs, num_queries, 4, generator=generator).clamp(1e-3, 1.0),
    }


def _targets(bs, num_classes, generator, counts):
    return [
        {
            "labels": torch.randint(0, num_classes, (n,), generator=generator),
            "boxes": torch.rand(n, 4, generator=generator).clamp(1e-3, 1.0),
        }
        for n in counts
    ]


def _dfine_matcher():
    from libreyolo.models.dfine.matcher import HungarianMatcher

    return HungarianMatcher(WEIGHTS)


def _deim_matcher():
    from libreyolo.models.deim.matcher import HungarianMatcher

    return HungarianMatcher(WEIGHTS)


def _deimv2_matcher():
    from libreyolo.models.deimv2.matcher import HungarianMatcher

    return HungarianMatcher(WEIGHTS)


def _rtdetr_matcher():
    from libreyolo.models.rtdetr.loss import HungarianMatcher

    return HungarianMatcher(WEIGHTS, use_focal_loss=False)


def _ec_seg_matcher():
    from libreyolo.models.ec.seg_loss import ECSegHungarianMatcher

    return ECSegHungarianMatcher(WEIGHTS)


MATCHERS = {
    "dfine": _dfine_matcher,
    "deim": _deim_matcher,
    "deimv2": _deimv2_matcher,
    "rtdetr": _rtdetr_matcher,
    "ec_seg": _ec_seg_matcher,
}


def _normalize_indices(result):
    """Both return conventions -> list of (pred_idx, tgt_idx) tensor pairs."""
    return result["indices"] if isinstance(result, dict) else result


@pytest.mark.parametrize("family", sorted(MATCHERS))
@pytest.mark.parametrize("counts", [(4, 7), (0, 0)], ids=["targets", "empty"])
def test_matcher_is_cdist_free_and_deterministic(family, counts, monkeypatch):
    def _no_cdist(*args, **kwargs):
        raise AssertionError("matcher hot path still calls torch.cdist")

    monkeypatch.setattr(torch, "cdist", _no_cdist)
    matcher = MATCHERS[family]()
    generator = torch.Generator().manual_seed(42)
    outputs = _outputs(2, 30, 5, generator)
    targets = _targets(2, 5, generator, counts)

    first = _normalize_indices(matcher(outputs, targets))
    second = _normalize_indices(matcher(outputs, targets))

    assert len(first) == 2
    for (pi, ti), (pj, tj), n in zip(first, second, counts):
        assert torch.equal(torch.as_tensor(pi), torch.as_tensor(pj))
        assert torch.equal(torch.as_tensor(ti), torch.as_tensor(tj))
        assert len(pi) == len(ti) == n


# ---------------------------------------------------------------------------
# Fused-optimizer construction helper
# ---------------------------------------------------------------------------


def _cpu_params():
    return [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(2)]


def test_build_optimizer_never_fuses_on_cpu():
    opt = build_optimizer(torch.optim.AdamW, _cpu_params(), lr=1e-3)
    assert not any(group.get("fused") for group in opt.param_groups)


def test_build_optimizer_cpu_step_is_bitwise_stock():
    torch.manual_seed(0)
    base = torch.randn(4, 4)
    p_helper = torch.nn.Parameter(base.clone())
    p_stock = torch.nn.Parameter(base.clone())
    opt_helper = build_optimizer(torch.optim.AdamW, [p_helper], lr=1e-2)
    opt_stock = torch.optim.AdamW([p_stock], lr=1e-2)
    for _ in range(3):
        grad = torch.randn(4, 4)
        p_helper.grad = grad.clone()
        p_stock.grad = grad.clone()
        opt_helper.step()
        opt_stock.step()
    assert torch.equal(p_helper, p_stock)


def test_build_optimizer_materializes_generators_and_group_dicts():
    params = (p for p in _cpu_params())
    opt = build_optimizer(torch.optim.SGD, params, lr=0.1)
    assert sum(len(g["params"]) for g in opt.param_groups) == 2

    groups = iter(
        [
            {"params": (p for p in _cpu_params()), "lr": 0.1},
            {"params": _cpu_params(), "lr": 0.2, "weight_decay": 0.0},
        ]
    )
    opt = build_optimizer(torch.optim.SGD, groups, lr=0.1)
    assert [len(g["params"]) for g in opt.param_groups] == [2, 2]


def test_build_optimizer_accepts_bare_tensor_group_params():
    # rfdetr builds per-parameter groups as {"params": tensor} with a BARE
    # tensor; list()-ing it would iterate rows into non-leaf views and torch
    # would raise "can't optimize a non-leaf Tensor" (found by the #765
    # all-family smoke).
    p1, p2 = _cpu_params()
    opt = build_optimizer(
        torch.optim.AdamW,
        [{"params": p1, "lr": 1e-3}, {"params": p2, "lr": 2e-3}],
        lr=1e-3,
    )
    assert [g["params"] for g in opt.param_groups] == [[p1], [p2]]
    assert opt.param_groups[0]["params"][0] is p1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_build_optimizer_bare_tensor_groups_fuse_on_cuda():
    p = torch.nn.Parameter(torch.randn(3, 3, device="cuda"))
    opt = build_optimizer(torch.optim.AdamW, [{"params": p, "lr": 1e-3}], lr=1e-3)
    assert all(group.get("fused") for group in opt.param_groups)


def test_build_optimizer_falls_back_when_fused_unsupported(monkeypatch):
    calls = []

    class _NoFusedAdamW(torch.optim.AdamW):
        def __init__(self, params, **kwargs):
            calls.append(sorted(kwargs))
            if "fused" in kwargs:
                raise TypeError("unexpected keyword argument 'fused'")
            super().__init__(params, **kwargs)

    monkeypatch.setattr(optim_mod, "_all_params_cuda", lambda groups: True)
    opt = build_optimizer(_NoFusedAdamW, _cpu_params(), lr=1e-3)
    assert isinstance(opt, _NoFusedAdamW)
    assert calls == [["fused", "lr"], ["lr"]]


def test_build_optimizer_never_requests_fused_sgd(monkeypatch):
    """Fused SGD corrupts momentum on GradScaler-skipped overflow steps
    (torch 2.11), so SGD must never be constructed with fused=True even when
    every parameter is on CUDA."""
    calls = []

    class _SpySGD(torch.optim.SGD):
        def __init__(self, params, **kwargs):
            calls.append(sorted(kwargs))
            super().__init__(params, **kwargs)

    monkeypatch.setattr(optim_mod, "_all_params_cuda", lambda groups: True)
    opt = build_optimizer(_SpySGD, _cpu_params(), lr=0.1, momentum=0.9)
    assert isinstance(opt, _SpySGD)
    assert calls == [["lr", "momentum"]]
    assert not any(group.get("fused") for group in opt.param_groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_build_optimizer_sgd_stays_stock_on_cuda():
    params = [torch.nn.Parameter(torch.randn(3, 3, device="cuda"))]
    opt = build_optimizer(torch.optim.SGD, params, lr=0.1, momentum=0.9)
    assert not any(group.get("fused") for group in opt.param_groups)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_scaler_overflow_step_leaves_sgd_state_untouched():
    """Regression for the 2026-08-14 nightly nan: an AMP overflow step must
    leave the built SGD's momentum state untouched and the weights unchanged,
    and the next finite step must land exactly where stock SGD lands."""
    p = torch.nn.Parameter(torch.ones(4, device="cuda"))
    opt = build_optimizer(torch.optim.SGD, [p], lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")

    overflow = (p * 1e30).sum() * 1e30
    scaler.scale(overflow).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    assert "momentum_buffer" not in opt.state.get(p, {})
    assert torch.equal(p.detach(), torch.ones(4, device="cuda"))

    finite = p.sum()
    scaler.scale(finite).backward()
    scaler.step(opt)
    scaler.update()
    assert torch.isfinite(p).all()
    assert torch.allclose(p.detach(), torch.full((4,), 0.9, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_build_optimizer_fuses_on_cuda():
    params = [torch.nn.Parameter(torch.randn(3, 3, device="cuda")) for _ in range(2)]
    opt = build_optimizer(torch.optim.AdamW, params, lr=1e-3)
    assert all(group.get("fused") for group in opt.param_groups)
    params[0].grad = torch.zeros_like(params[0])
    params[1].grad = torch.zeros_like(params[1])
    opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_build_optimizer_mixed_devices_stay_unfused():
    params = [
        torch.nn.Parameter(torch.randn(3, 3, device="cuda")),
        torch.nn.Parameter(torch.randn(3, 3)),
    ]
    opt = build_optimizer(torch.optim.AdamW, params, lr=1e-3)
    assert not any(group.get("fused") for group in opt.param_groups)


# ---------------------------------------------------------------------------
# Checkpoint resume must not resurrect the checkpoint's step implementation
# ---------------------------------------------------------------------------


def test_restore_optimizer_state_keeps_live_impl_selection():
    from libreyolo.training.optim import restore_optimizer_state

    donor_param = torch.nn.Parameter(torch.randn(3, 3))
    donor = torch.optim.AdamW([donor_param], lr=5e-4)
    donor_param.grad = torch.randn(3, 3)
    donor.step()
    state = donor.state_dict()
    # Simulate a checkpoint written by a fused CUDA run.
    state["param_groups"][0]["fused"] = True
    state["param_groups"][0]["foreach"] = False

    live_param = torch.nn.Parameter(torch.randn(3, 3))
    live = torch.optim.AdamW([live_param], lr=1e-3)
    live_impl = {
        key: live.param_groups[0][key]
        for key in ("fused", "foreach", "capturable", "differentiable")
    }
    restore_optimizer_state(live, state)

    # Implementation-selection keys stay as constructed for THIS device...
    for key, value in live_impl.items():
        assert live.param_groups[0][key] == value
    assert live.param_groups[0]["fused"] is not True
    # ...while real hyperparameters and state do come from the checkpoint.
    assert live.param_groups[0]["lr"] == 5e-4
    assert len(live.state) == 1
    live_param.grad = torch.randn(3, 3)
    live.step()


# ---------------------------------------------------------------------------
# yolo9 loss-path sync removal (issue #763 item 3)
# ---------------------------------------------------------------------------


def test_yolo9_cls_norm_is_tensor_and_value_preserved():
    from libreyolo.models.yolo9.loss import YOLO9Loss

    stub = type("Stub", (), {"distributed_normalize": False})()
    targets_cls = torch.zeros(2, 8400, 5)
    targets_cls[0, :3, 1] = 1.0
    norm = YOLO9Loss._global_cls_norm(stub, targets_cls)
    assert isinstance(norm, torch.Tensor) and norm.ndim == 0
    assert float(norm) == 3.0
    # The empty batch keeps the clamp floor the old float path had.
    assert float(YOLO9Loss._global_cls_norm(stub, torch.zeros(2, 10, 5))) == 1.0


def test_yolo9_get_loss_components_single_transfer_matches_item(monkeypatch):
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    outputs = {
        "box": torch.tensor(1.25),
        "cls": torch.tensor(0.5),
        "dfl": torch.tensor(2.0),
        "num_fg": torch.tensor(3.0),
    }
    transfers = []
    original_cpu = torch.Tensor.cpu

    def _counting_cpu(self, *args, **kwargs):
        transfers.append(tuple(self.shape))
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", _counting_cpu)
    components = YOLO9Trainer.get_loss_components(None, outputs)
    assert components == {"box": 1.25, "cls": 0.5, "dfl": 2.0}
    assert all(isinstance(v, float) for v in components.values())
    assert transfers == [(3,)]  # one stacked transfer, not one per key


def test_all_reduce_avg_scalar_tensor_matches_float_form():
    from libreyolo.training.distributed import (
        all_reduce_avg_scalar,
        all_reduce_avg_scalar_tensor,
    )

    for value in (torch.tensor(7.0), torch.tensor(0.0), 4.5, 0.2):
        as_tensor = all_reduce_avg_scalar_tensor(value)
        assert isinstance(as_tensor, torch.Tensor) and as_tensor.ndim == 0
        assert float(as_tensor) == all_reduce_avg_scalar(value)
