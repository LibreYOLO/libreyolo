"""The default YOLO9 PGI recipe must retain both branches under compilation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.yolo9.nn import LibreYOLO9Model
from libreyolo.models.yolo9.trainer import YOLO9Trainer
from libreyolo.training.compile import TrainCompileManager, build_compile_manager
from libreyolo.training.config import YOLO9Config

pytestmark = pytest.mark.unit


def _host(model, task="detect"):
    host = SimpleNamespace(
        model=model,
        wrapper_model=SimpleNamespace(task=task),
        config=YOLO9Config(compile=True, device="cpu"),
        device=torch.device("cpu"),
        get_model_family=lambda: "yolo9",
        _accum_steps=1,
    )
    host.cuda_graph_train_spec = lambda: YOLO9Trainer.cuda_graph_train_spec(host)
    host.compile_train_spec = lambda: YOLO9Trainer.compile_train_spec(host)
    return host


def _batch():
    generator = torch.Generator().manual_seed(885)
    images = torch.randn(2, 3, 96, 96, generator=generator)
    targets = torch.zeros(2, 4, 5)
    targets[..., 0] = -1
    targets[:, 0] = torch.tensor([1, 0.2, 0.2, 0.8, 0.8])
    return images, targets


def test_default_public_train_builds_pgi_compiler(monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    observed = {}

    class InspectTrainer(YOLO9Trainer):
        def train(self):
            observed["aux_weight"] = self.model.aux_weight
            observed["manager"] = build_compile_manager(self)
            return {}

    wrapper = LibreYOLO9(None, "t", nb_classes=3, device="cpu")
    monkeypatch.setattr(wrapper, "_trainer_class", lambda: InspectTrainer)
    monkeypatch.setattr(
        "libreyolo.data.load_data_config",
        lambda *args, **kwargs: {"nc": 3, "names": {0: "a", 1: "b", 2: "c"}},
    )
    wrapper.train(data="unused.yaml", compile=True, device="cpu")
    assert observed["aux_weight"] == YOLO9Config().aux_weight == 0.25
    assert observed["manager"] is not None
    assert wrapper.model.aux is not None


@pytest.mark.parametrize("weight", [0.25, 0.75])
def test_pgi_split_preserves_losses_gradients_buffers_and_ownership(weight):
    torch.manual_seed(885)
    eager_model = LibreYOLO9Model(config="t", nb_classes=3).enable_aux(weight).train()
    split_model = deepcopy(eager_model)
    host = _host(split_model)
    assert host.cuda_graph_train_spec() is None
    manager = build_compile_manager(host)
    assert manager is not None
    assert {id(p) for p in manager.spec.network.parameters()} == {
        id(p) for p in split_model.parameters()
    }
    state_keys = set(split_model.state_dict())
    images, targets = _batch()
    eager = eager_model(images, targets=targets)
    split = manager.spec.assemble(manager.spec.network(images), images, targets)
    assert split.keys() == eager.keys()
    for key in eager:
        torch.testing.assert_close(split[key], eager[key], rtol=0, atol=0)
    eager["total_loss"].backward()
    split["total_loss"].backward()
    for (name, expected), (_, actual) in zip(
        eager_model.named_parameters(), split_model.named_parameters()
    ):
        assert (expected.grad is None) == (actual.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=1e-5, atol=1e-6, msg=name
            )
    for name, buffer in eager_model.named_buffers():
        torch.testing.assert_close(
            buffer, dict(split_model.named_buffers())[name], rtol=0, atol=0
        )
    assert set(split_model.state_dict()) == state_keys
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in split_model.aux_head.parameters()
    )


def test_default_pgi_compiles_with_aot_autograd(monkeypatch):
    torch.manual_seed(886)
    model = (
        LibreYOLO9Model(config="t", nb_classes=3)
        .enable_aux(YOLO9Config().aux_weight)
        .train()
    )
    original_compile = torch.compile
    requested = []

    def cpu_compile(fn, **kwargs):
        requested.append(kwargs)
        return original_compile(fn, backend="aot_eager", fullgraph=False)

    monkeypatch.setattr(torch, "compile", cpu_compile)
    manager = build_compile_manager(_host(model))
    assert isinstance(manager, TrainCompileManager)
    images, targets = _batch()
    outputs = manager.run(images, targets)
    assert outputs is not None and manager.started and not manager.disabled
    outputs["total_loss"].backward()
    assert requested[0]["backend"] == "inductor"
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_pgi_compiler_rejects_other_tasks():
    model = LibreYOLO9Model(config="t", nb_classes=3).enable_aux()
    assert _host(model, task="segment").compile_train_spec() is None
