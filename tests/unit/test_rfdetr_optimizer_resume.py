import copy

import pytest
import torch

from libreyolo.models.rfdetr.optim import merge_adamw_groups
from libreyolo.training.optim import restore_optimizer_state

pytestmark = pytest.mark.unit


def make_optimizer(merged, foreach=False):
    params = [torch.nn.Parameter(torch.full((3,), float(i + 1))) for i in range(6)]
    groups = [
        {"params": p, "lr": 0.01 if i % 2 else 0.02, "lr_mult": 1 if i % 2 else 2}
        for i, p in enumerate(params)
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.03, foreach=foreach)
    if merged:
        merge_adamw_groups(opt)
    return params, opt


def step(params, opt, iteration):
    opt.zero_grad(set_to_none=True)
    sum(
        (p.square() * (i + iteration + 1)).sum() for i, p in enumerate(params)
    ).backward()
    opt.step()


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("saved_merged", [False, True])
def test_group_merge_preserves_updates_and_resume(foreach, saved_merged):
    baseline, old = make_optimizer(saved_merged, foreach)
    for iteration in range(3):
        step(baseline, old, iteration)
    saved = copy.deepcopy(old.state_dict())
    resumed, new = make_optimizer(True, foreach)
    assert len(new.param_groups) == 2
    for p, q in zip(resumed, baseline):
        p.data.copy_(q)
    restore_optimizer_state(new, saved)
    assert len(saved["param_groups"]) == (2 if saved_merged else 6)
    for iteration in range(3, 6):
        step(baseline, old, iteration)
        step(resumed, new, iteration)
    for p, q in zip(baseline, resumed):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                old.state[p][key], new.state[q][key], rtol=0, atol=0
            )


def test_legacy_resume_with_appended_adapter_group():
    baseline, old = make_optimizer(False)
    resumed, new = make_optimizer(True)
    for params, opt in [(baseline, old), (resumed, new)]:
        adapter = torch.nn.Parameter(torch.ones(3))
        params.append(adapter)
        opt.add_param_group({"params": [adapter], "lr": 0.005})
    step(baseline, old, 0)
    for p, q in zip(resumed, baseline):
        p.data.copy_(q)
    restore_optimizer_state(new, copy.deepcopy(old.state_dict()))
    step(baseline, old, 1)
    step(resumed, new, 1)
    for p, q in zip(baseline, resumed):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_distinct_saved_group_settings_are_not_silently_lost():
    _, old = make_optimizer(False)
    _, new = make_optimizer(True)
    saved = old.state_dict()
    saved["param_groups"][2]["betas"] = (0.8, 0.99)
    with pytest.raises(ValueError, match="incompatible saved hyperparameters"):
        restore_optimizer_state(new, saved)


def test_resume_regroups_changed_learning_rates_without_scrambling_moments():
    params, old = make_optimizer(True)
    step(params, old, 0)
    saved = copy.deepcopy(old.state_dict())
    new_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    # All rates now agree: the previous two buckets become a single bucket.
    new = torch.optim.AdamW(
        [{"params": p, "lr": 0.004, "lr_mult": 1.0} for p in new_params],
        weight_decay=0.2,
    )
    merge_adamw_groups(new, override_weight_decay=True)
    restore_optimizer_state(new, saved)
    assert len(new.param_groups) == 1
    assert (
        new.param_groups[0]["lr"] == 0.004
        and new.param_groups[0]["weight_decay"] == 0.2
    )
    for p, q in zip(params, new_params):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                old.state[p][key], new.state[q][key], rtol=0, atol=0
            )


def test_gradient_clipping_order_is_preserved():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    params, opt = make_optimizer(True)
    for p in params:
        p.grad = torch.ones_like(p)
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.optimizer = opt
    assert [id(p) for p in trainer._gradient_clip_parameters()] == [
        id(p) for p in params
    ]


def test_resume_keeps_saved_decay_without_explicit_override():
    _, old = make_optimizer(False)
    _, new = make_optimizer(True)
    saved = old.state_dict()
    for group in saved["param_groups"]:
        group["weight_decay"] = 0.07
    restore_optimizer_state(new, saved)
    assert all(group["weight_decay"] == 0.07 for group in new.param_groups)
