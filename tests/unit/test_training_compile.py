"""CPU-safe compiler routing, failure recovery and checkpoint lifecycle."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from libreyolo.training.compile import (
    TrainCompileManager,
    build_compile_manager,
    compiler_options,
)
from libreyolo.training.config import COMPILE_MODES, TrainConfig
from libreyolo.training.cuda_graph import CudaGraphTrainSpec, GraphableNetwork
from libreyolo.training.optim import OptimizerStateMigrationError
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


def _spec(model):
    network = GraphableNetwork(model)

    def assemble(flat, imgs, targets, polygons=None):
        assert not torch.compiler.is_compiling(), "criterion must stay eager"
        return {"total_loss": (network.rebuild(flat) - targets).square().mean()}

    return CudaGraphTrainSpec(network, assemble)


def _manager(model, **kwargs):
    return TrainCompileManager(_spec(model), mode="default", options={}, **kwargs)


@pytest.mark.parametrize(
    "value,expected",
    [
        (False, False),
        (True, "default"),
        ("false", False),
        ("TRUE", "default"),
        *[(v, v) for v in COMPILE_MODES],
    ],
)
def test_compile_config_round_trip(value, expected, tmp_path):
    config = TrainConfig(compile=value)
    assert config.compile == expected
    assert TrainConfig.from_kwargs(**config.to_dict()).compile == expected
    config.to_yaml(tmp_path / "train.yaml")
    from libreyolo.training.config import load_train_cfg

    assert load_train_cfg(tmp_path / "train.yaml")["compile"] == expected


@pytest.mark.parametrize("value", [None, 0, 1, [], {}, "fast", "", "inductor"])
def test_compile_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="compile must be"):
        TrainConfig(compile=value)


@pytest.mark.parametrize("value", ["scipy", "auto", "torch"])
def test_rfdetr_matcher_config_round_trip(value):
    from libreyolo.models.rfdetr.config import RFDETRConfig

    config = RFDETRConfig(matcher_backend=value, compile=True)
    restored = RFDETRConfig.from_kwargs(**config.to_dict())
    assert restored.matcher_backend == value
    assert restored.compile == "default"
    assert RFDETRConfig().matcher_backend == "scipy"
    assert RFDETRConfig().compile is False
    assert RFDETRConfig().nbs == 16


@pytest.mark.parametrize("value", [None, 1, True, "gpu", ""])
def test_rfdetr_matcher_config_rejects_invalid_values(value):
    from libreyolo.models.rfdetr.config import RFDETRConfig

    with pytest.raises(ValueError, match="matcher_backend must be"):
        RFDETRConfig(matcher_backend=value)


@pytest.mark.parametrize("family", ["yolo9", "rfdetr"])
def test_factory_preserves_model_and_parameter_identity(family):
    model = nn.Linear(3, 2)
    host = SimpleNamespace(
        config=TrainConfig(compile=True),
        device=torch.device("cpu"),
        get_model_family=lambda: family,
        _accum_steps=1,
        cuda_graph_train_spec=lambda: _spec(model),
    )
    manager = build_compile_manager(host)
    assert manager.spec.network.module is model
    assert next(iter(manager.spec.network.parameters())) is model.weight
    assert list(model.state_dict()) == ["weight", "bias"]


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"device": torch.device("mps")}, "CPU/CUDA"),
        ({"is_distributed": True}, "distributed"),
        ({"distiller": object()}, "distillation"),
        ({"config": TrainConfig(compile=True, lora=True)}, "LoRA"),
        (
            {"wrapper_model": SimpleNamespace(_quant_manifest={"recipe": "int8"})},
            "quantization",
        ),
        ({"get_model_family": lambda: "unknown"}, "family"),
        ({"cuda_graph_train_spec": lambda: None}, "task or network variant"),
    ],
)
def test_unsupported_runs_warn_and_remain_eager(overrides, reason, caplog):
    host = SimpleNamespace(
        config=TrainConfig(compile=True, cuda_graph=True),
        device=torch.device("cpu"),
        get_model_family=lambda: "yolo9",
        _accum_steps=1,
        cuda_graph_train_spec=lambda: _spec(nn.Linear(3, 2)),
    )
    host.__dict__.update(overrides)
    assert build_compile_manager(host) is None
    assert reason in caplog.text


@pytest.mark.parametrize(
    "mode,explicit",
    [("default", True), ("reduce-overhead", False), ("max-autotune", False)],
)
def test_cuda_graphs_owned_by_compiler(mode, explicit):
    options, graphs = compiler_options(
        mode,
        device=torch.device("cuda"),
        cuda_graph=explicit,
        accumulation_steps=1,
    )
    assert graphs and options["triton.cudagraphs"] is True
    if mode == "max-autotune":
        assert options["max_autotune"] is True


@pytest.mark.parametrize("device,accum", [("cpu", 1), ("cuda", 4)])
def test_graph_fallback_retains_compiler_mode(device, accum, caplog):
    options, graphs = compiler_options(
        "max-autotune",
        device=torch.device(device),
        cuda_graph=True,
        accumulation_steps=accum,
    )
    assert not graphs and options["triton.cudagraphs"] is False
    assert options["max_autotune"] is True
    assert "retaining compile" in caplog.text


def test_mark_step_and_exclusive_dispatch(monkeypatch):
    model = nn.Linear(3, 2)
    manager = _manager(model, cuda_graph=True)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    marker = Mock()
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", marker)
    host = SimpleNamespace(
        _compile_manager=manager,
        _cuda_graph_manager=Mock(),
        on_forward=Mock(),
    )
    for _ in range(2):
        BaseTrainer._forward_train(host, torch.randn(2, 3), torch.randn(2, 2))
    assert marker.call_count == 2
    host._cuda_graph_manager.run.assert_not_called()
    host.on_forward.assert_not_called()
    manager.disabled = True
    BaseTrainer._forward_train(host, torch.randn(2, 3), torch.randn(2, 2))
    host._cuda_graph_manager.run.assert_not_called()
    host.on_forward.assert_called_once()


def test_first_forward_failure_restores_buffers_and_rng(monkeypatch, caplog):
    model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Dropout(0.3))
    reference = deepcopy(model)
    manager = _manager(model)
    imgs, target = torch.randn(4, 3), torch.randn(4, 3)
    rng = torch.get_rng_state()
    reference_spec = _spec(reference)
    expected = reference_spec.assemble(reference_spec.network(imgs), imgs, target)
    torch.set_rng_state(rng)

    def compiler(fn, **kwargs):
        def fail_after_stateful_forward(x):
            fn(x)
            raise RuntimeError("backend unavailable")

        return fail_after_stateful_forward

    monkeypatch.setattr(torch, "compile", compiler)
    assert manager.run(imgs, target) is None
    assert torch.equal(torch.get_rng_state(), rng)
    assert manager.disabled
    assert "using eager training" in caplog.text
    actual = manager.spec.assemble(manager.spec.network(imgs), imgs, target)
    torch.testing.assert_close(
        actual["total_loss"], expected["total_loss"], rtol=0, atol=0
    )
    for name, buffer in model.named_buffers():
        torch.testing.assert_close(
            buffer, dict(reference.named_buffers())[name], rtol=0, atol=0
        )


def test_later_failure_does_not_replay_training(monkeypatch):
    model = nn.Linear(3, 2)
    manager = _manager(model)
    calls = []

    def compiler(fn, **kwargs):
        def fail_later(x):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("late compiler failure")
            return fn(x)

        return fail_later

    monkeypatch.setattr(torch, "compile", compiler)
    manager.run(torch.randn(2, 3), torch.randn(2, 2))
    with pytest.raises(RuntimeError, match="late compiler failure"):
        manager.run(torch.randn(2, 3), torch.randn(2, 2))
    assert len(calls) == 2


class TinyTrainer(BaseTrainer):
    def get_model_family(self):
        return "yolo9"

    def get_model_tag(self):
        return "tiny-compiler-test"

    def create_transforms(self):
        raise AssertionError("synthetic loader owns these tests")

    def _setup_data(self):
        self.train_loader = [(torch.ones(2, 3), torch.ones(2, 2), None, None)]

    def create_scheduler(self, iters_per_epoch):
        return SimpleNamespace(update_lr=lambda step: self.config.lr0)

    def get_loss_components(self, outputs):
        return {}

    def cuda_graph_train_spec(self):
        return _spec(self.model)

    def on_forward(self, imgs, targets, polygons=None):
        return {"total_loss": (self.model(imgs) - targets).square().mean()}


def _trainer(tmp_path, **kwargs):
    return TinyTrainer(
        nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.SiLU(), nn.Linear(4, 2)),
        device="cpu",
        num_classes=2,
        batch=2,
        project=str(tmp_path),
        exist_ok=True,
        optimizer="adamw",
        amp=False,
        **kwargs,
    )


def test_compiled_optimizer_ema_checkpoint_resume_matches_eager(monkeypatch, tmp_path):
    # Exercise actual Dynamo + AOTAutograd on CPU without a platform C++ toolchain.
    compile_api = torch.compile
    requests = []

    def aot_compile(fn, **kwargs):
        requests.append(kwargs)
        return compile_api(fn, backend="aot_eager", fullgraph=False)

    monkeypatch.setattr(torch, "compile", aot_compile)
    eager = _trainer(tmp_path / "eager")
    compiled = _trainer(tmp_path / "compiled", compile=True, cuda_graph=True)
    compiled.model.load_state_dict(eager.model.state_dict())
    original_model = compiled.model
    original_parameters = tuple(compiled.model.parameters())
    eager.setup()
    compiled.setup()
    assert compiled._cuda_graph_manager is None
    assert compiled.model is original_model
    for _ in range(2):
        imgs, targets = torch.randn(4, 3), torch.randn(4, 2)
        losses = []
        for trainer in (eager, compiled):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            out = trainer._forward_train(imgs, targets)
            losses.append(out["total_loss"].detach())
            out["total_loss"].backward()
            trainer.optimizer.step()
            trainer.ema_model.update(trainer.model)
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    assert requests and requests[0]["backend"] == "inductor"
    assert requests[0]["options"]["triton.cudagraphs"] is False
    for expected, actual in zip(eager.model.parameters(), compiled.model.parameters()):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    assert all(a is b for a, b in zip(original_parameters, compiled.model.parameters()))
    compiled._save_checkpoint(epoch=0, loss=1.0, is_best=False)
    path = compiled.save_dir / "weights" / "last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["compile"] == "default"
    for key in ("model", "train_model", "ema"):
        assert set(checkpoint[key]) == set(original_model.state_dict())
        assert all("_orig_mod" not in name for name in checkpoint[key])
    resumed = _trainer(tmp_path / "resumed", compile=True)
    resumed.setup()
    resumed.resume(str(path))
    assert resumed.ema_model.updates == compiled.ema_model.updates
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, compiled.model.state_dict()[name])
    for name, value in eager.ema_model.ema.state_dict().items():
        torch.testing.assert_close(
            value, compiled.ema_model.ema.state_dict()[name], rtol=1e-5, atol=1e-6
        )
    for old, new in zip(
        compiled.optimizer.state.values(), resumed.optimizer.state.values()
    ):
        torch.testing.assert_close(old["exp_avg"], new["exp_avg"])
        torch.testing.assert_close(old["exp_avg_sq"], new["exp_avg_sq"])


@pytest.mark.parametrize("deferred", [False, True])
def test_unsafe_optimizer_migration_is_never_swallowed(monkeypatch, tmp_path, deferred):
    original = _trainer(tmp_path / "original")
    original.setup()
    original._save_checkpoint(epoch=0, loss=1.0, is_best=False)
    path = original.save_dir / "weights" / "last.pt"
    resumed = _trainer(tmp_path / "resumed")
    if deferred:
        resumed.resume(str(path))
    else:
        resumed.setup()

    def reject(*args):
        raise OptimizerStateMigrationError("unsafe parameter mapping")

    monkeypatch.setattr("libreyolo.training.trainer.restore_optimizer_state", reject)
    with pytest.raises(OptimizerStateMigrationError, match="unsafe parameter mapping"):
        if deferred:
            resumed.setup()
        else:
            resumed.resume(str(path))
