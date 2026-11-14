"""CUDA activation and optimizer checks for RF-DETR's upstream training port.

Run explicitly with ``pytest tests/e2e/test_rfdetr_training_optimizations.py -m e2e``.
Uses random weights and synthetic inputs; no dataset/weight downloads.
"""

from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from libreyolo.models.rfdetr import transformer as tr
from libreyolo.models.rfdetr.config import RFDETRConfig
from libreyolo.models.rfdetr.nn import LibreRFDETRModel
from libreyolo.models.rfdetr.trainer import RFDETRTrainer
from tests.e2e.conftest import cuda_cleanup, requires_cuda

pytestmark = [pytest.mark.e2e, pytest.mark.rfdetr]
try:
    _HAS_ASSIGNMENT = version("torch-hungarian") == "0.1.0rc0"
except PackageNotFoundError:
    _HAS_ASSIGNMENT = False
requires_assignment = pytest.mark.skipif(
    not _HAS_ASSIGNMENT, reason="requires libreyolo[rfdetr-train]"
)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    cuda_cleanup()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_fast_paths_execute_and_update_parameters(dtype, monkeypatch):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unsupported on this GPU")
    torch.manual_seed(37)
    model = LibreRFDETRModel(config="n", nb_classes=4, device="cuda").cuda().train()
    trainer = RFDETRTrainer.__new__(RFDETRTrainer)
    trainer.model, trainer.config = model, RFDETRConfig(num_classes=3)
    trainer.wrapper_model, trainer.world_size = SimpleNamespace(task="detect"), 1
    original_groups = trainer._setup_upstream_optimizer_groups()
    optimizer = trainer._setup_optimizer()
    trainer.optimizer = optimizer
    assert len(optimizer.param_groups) < len(original_groups) // 2
    criterion, _ = model.build_criterion_and_postprocess()
    criterion.cuda().train()
    counts = {"proposal": 0, "attention": 0, "ffn": 0}

    def spy(name, fn):
        def call(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

        return call

    core = model.model.transformer
    monkeypatch.setattr(
        core,
        "_two_stage_group_selection",
        spy("proposal", core._two_stage_group_selection),
    )
    for layer in core.decoder.layers:
        monkeypatch.setattr(
            layer,
            "_grouped_self_attention",
            spy("attention", layer._grouped_self_attention),
        )
    monkeypatch.setattr(tr._LinearReLU, "apply", spy("ffn", tr._LinearReLU.apply))
    images = torch.randn(4, 3, 384, 384, device="cuda")
    targets = [
        {
            "labels": torch.randint(3, (7,), device="cuda"),
            "boxes": torch.rand(7, 4, device="cuda").clamp(0.01, 1),
        }
        for _ in range(4)
    ]
    scaler = torch.amp.GradScaler(
        "cuda", enabled=dtype == torch.float16, init_scale=128
    )
    before = model.model.class_embed.weight.detach().clone()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        losses = criterion(model(images), targets)
        loss = sum(
            value * criterion.weight_dict[key]
            for key, value in losses.items()
            if key in criterion.weight_dict
        )
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    trainer._clip_gradients()
    scaler.step(optimizer)
    scaler.update()
    assert all(counts.values()), counts
    assert not torch.equal(before, model.model.class_embed.weight)


@requires_cuda
@requires_assignment
def test_optional_assignment_objective_on_cuda():
    from libreyolo.models.rfdetr._assignment import assign_many_bucketed

    torch.manual_seed(41)
    sizes, groups, width = [0, 7, 17], 3, 30
    costs = [torch.rand(groups * width, sum(sizes), device="cuda") for _ in range(2)]
    actual = assign_many_bucketed(costs, sizes, groups)
    for cost, layer in zip(costs, actual):
        host, offset = cost.cpu(), 0
        for size, (rows, cols) in zip(sizes, layer):
            assert len(rows) == len(cols) == groups * size
            for group in range(groups):
                r = rows[group * size : (group + 1) * size] - group * width
                c = cols[group * size : (group + 1) * size]
                block = host[
                    group * width : (group + 1) * width, offset : offset + size
                ]
                rr, cc = linear_sum_assignment(block.numpy())
                assert r.unique().numel() == c.unique().numel() == size
                torch.testing.assert_close(
                    block[r, c].sum(), block[rr, cc].sum(), atol=1e-5, rtol=1e-6
                )
            offset += size
