"""``train(compile=...)``: option contract, gating, routing and fallback.

CPU tests swap Inductor for the ``aot_eager`` backend where a real trace is
needed; CUDA speed and Inductor numerics are covered by
``tests/e2e/test_training_compile.py``.
"""

import copy
import functools
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from libreyolo.training import compile as compile_mod
from libreyolo.training.config import TrainConfig, normalize_compile
from libreyolo.training.cuda_graph import CudaGraphTrainSpec, GraphableNetwork
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "value,expected",
    [
        (False, False),
        (True, "default"),
        ("true", "default"),
        ("False", False),
        ("default", "default"),
        ("reduce-overhead", "reduce-overhead"),
        ("max-autotune", "max-autotune"),
        (" MAX-AUTOTUNE-NO-CUDAGRAPHS ", "max-autotune-no-cudagraphs"),
    ],
)
def test_compile_values_normalize(value, expected):
    assert normalize_compile(value) == expected
    assert TrainConfig(compile=value).compile == expected


@pytest.mark.parametrize("value", ["fast", 1.0, None, "inductor"])
def test_compile_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="compile must be"):
        TrainConfig(compile=value)


def test_compile_round_trips_through_yaml(tmp_path):
    import yaml

    path = tmp_path / "cfg.yaml"
    TrainConfig(compile="reduce-overhead").to_yaml(path)
    saved = yaml.safe_load(path.read_text())
    assert saved["compile"] == "reduce-overhead"
    assert TrainConfig.from_kwargs(**saved).compile == "reduce-overhead"


def _trainer(device="cuda", **overrides):
    config = SimpleNamespace(compile=True, cuda_graph=False)
    for key, value in overrides.items():
        setattr(config, key, value)
    return SimpleNamespace(
        config=config,
        device=torch.device(device),
        is_distributed=False,
        distiller=None,
        _accum_steps=1,
    )


@pytest.mark.parametrize(
    "trainer,reason",
    [
        (_trainer(device="cpu"), "not CUDA"),
        (_trainer(device="mps"), "not CUDA"),
        (SimpleNamespace(**{**vars(_trainer()), "is_distributed": True}), "distributed"),
        (SimpleNamespace(**{**vars(_trainer()), "distiller": object()}), "distillation"),
    ],
)
def test_unsupported_runs_warn_and_stay_eager(trainer, reason, caplog):
    with caplog.at_level(logging.WARNING):
        assert compile_mod.build_train_compiler(trainer) is None
    assert reason in caplog.text


def test_compile_off_builds_nothing():
    assert compile_mod.build_train_compiler(_trainer(compile=False)) is None


def test_mode_options_and_graph_policy(caplog):
    options, graphs = compile_mod._inductor_options(
        "default", cuda_graph=False, accum_steps=1, dynamic=None
    )
    assert (options, graphs) == ({"triton.cudagraphs": False}, False)

    options, graphs = compile_mod._inductor_options(
        "max-autotune", cuda_graph=False, accum_steps=1, dynamic=None
    )
    assert graphs and options["max_autotune"] and options["triton.cudagraphs"]

    _, graphs = compile_mod._inductor_options(
        "default", cuda_graph=True, accum_steps=1, dynamic=None
    )
    assert graphs

    # Graph trees keep gradients in the graph pool: never with accumulation.
    with caplog.at_level(logging.WARNING):
        options, graphs = compile_mod._inductor_options(
            "reduce-overhead", cuda_graph=True, accum_steps=4, dynamic=True
        )
    assert not graphs and options["triton.cudagraphs"] is False
    assert "gradient accumulation" in caplog.text
    import torch._inductor.config as inductor_config

    if hasattr(inductor_config.triton, "coalesce_tiling_analysis"):
        assert options["triton.coalesce_tiling_analysis"] is False


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x):
        return {"out": self.lin(x)}


def _toy_host(model, spec_fn=None):
    network = GraphableNetwork(model)

    def assemble(flat, imgs, targets, polygons=None):
        out = network.rebuild(flat)["out"]
        return {"total_loss": (out - targets).pow(2).mean()}

    spec = CudaGraphTrainSpec(network=network, assemble=assemble)
    host = SimpleNamespace(
        compile_train_spec=spec_fn or (lambda: spec),
        compile_dynamic=lambda: None,
        on_forward=lambda imgs, targets, polygons=None: {
            "total_loss": (model(imgs)["out"] - targets).pow(2).mean()
        },
    )
    return host, spec


@pytest.fixture
def aot_eager(monkeypatch):
    real = torch.compile
    calls = []

    def fake(network, **kwargs):
        calls.append(kwargs)
        return real(network, backend="aot_eager", dynamic=kwargs.get("dynamic"))

    monkeypatch.setattr(compile_mod.torch, "compile", fake)
    return calls


def test_compiled_step_matches_eager_and_keeps_the_model_untouched(aot_eager):
    torch.manual_seed(0)
    model = _Toy()
    reference = copy.deepcopy(model)
    host, spec = _toy_host(model)
    host._train_compiler = compile_mod.TrainCompiler(
        "default", cuda_graph=False, accum_steps=1
    )
    x, y = torch.randn(5, 4), torch.randn(5, 3)

    out = BaseTrainer._forward_train(host, x, y)
    out["total_loss"].backward()
    ref = (reference(x)["out"] - y).pow(2).mean()
    ref.backward()

    assert torch.equal(out["total_loss"], ref)
    assert torch.equal(model.lin.weight.grad, reference.lin.weight.grad)
    assert host._train_compiler.spec is spec and len(aot_eager) == 1
    # The model itself is not wrapped: no compiled call, no prefixed keys.
    assert "_compiled_call_impl" not in vars(model) or model._compiled_call_impl is None
    assert set(model.state_dict()) == {"lin.weight", "lin.bias"}


def test_missing_boundary_warns_once_and_runs_eager(aot_eager, caplog):
    model = _Toy()
    host, _ = _toy_host(model, spec_fn=lambda: None)
    compiler = compile_mod.TrainCompiler("default", cuda_graph=False, accum_steps=1)
    host._train_compiler = compiler
    x, y = torch.randn(2, 4), torch.randn(2, 3)
    with caplog.at_level(logging.WARNING):
        BaseTrainer._forward_train(host, x, y)
        BaseTrainer._forward_train(host, x, y)
    assert caplog.text.count("no compiled training boundary") == 1
    assert compiler.disabled and host._train_compiler is None and aot_eager == []


def test_compiler_failure_falls_back_to_eager(monkeypatch, caplog):
    model = _Toy()
    host, _ = _toy_host(model)
    failing = MagicMock(side_effect=torch._dynamo.exc.TorchDynamoException("codegen"))
    monkeypatch.setattr(compile_mod.torch, "compile", lambda network, **kw: failing)
    compiler = compile_mod.TrainCompiler("default", cuda_graph=False, accum_steps=1)
    host._train_compiler = compiler
    x, y = torch.randn(2, 4), torch.randn(2, 3)
    with caplog.at_level(logging.WARNING):
        out = BaseTrainer._forward_train(host, x, y)
    assert torch.isfinite(out["total_loss"])
    assert compiler.disabled and "compilation failed" in caplog.text
    BaseTrainer._forward_train(host, x, y)
    failing.assert_called_once()


def test_runtime_errors_propagate(monkeypatch):
    model = _Toy()
    host, _ = _toy_host(model)
    monkeypatch.setattr(
        compile_mod.torch,
        "compile",
        lambda network, **kw: MagicMock(side_effect=torch.OutOfMemoryError("oom")),
    )
    host._train_compiler = compile_mod.TrainCompiler(
        "default", cuda_graph=False, accum_steps=1
    )
    with pytest.raises(torch.OutOfMemoryError):
        BaseTrainer._forward_train(host, torch.randn(2, 4), torch.randn(2, 3))


def test_graph_replay_marks_each_step(monkeypatch):
    model = _Toy()
    host, _ = _toy_host(model)
    monkeypatch.setattr(compile_mod.torch, "compile", lambda network, **kw: network)
    marks = MagicMock()
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", marks)
    compiler = compile_mod.TrainCompiler("reduce-overhead", cuda_graph=False, accum_steps=1)
    host._train_compiler = compiler
    for _ in range(3):
        BaseTrainer._forward_train(host, torch.randn(2, 4), torch.randn(2, 3))
    assert compiler.cudagraphs and marks.call_count == 3


def test_rfdetr_compiles_dynamic_only_with_multi_scale():
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    host = SimpleNamespace(_multi_scale_scales=lambda: [320, 352, 384])
    assert RFDETRTrainer.compile_dynamic(host) is True
    host = SimpleNamespace(_multi_scale_scales=list)
    assert RFDETRTrainer.compile_dynamic(host) is None


def test_yolo9_pgi_boundary_matches_the_model_forward():
    """The default YOLO9 recipe trains the PGI auxiliary branch; capture skips
    it, the compile boundary must reproduce the model's own loss exactly."""
    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    torch.manual_seed(0)
    model = LibreYOLO9Model(config="t", nb_classes=3).enable_aux(0.25).train()
    reference = copy.deepcopy(model)
    host = SimpleNamespace(model=model, wrapper_model=SimpleNamespace(task="detect"))
    host.cuda_graph_train_spec = functools.partial(
        YOLO9Trainer.cuda_graph_train_spec, host
    )
    assert host.cuda_graph_train_spec() is None
    spec = YOLO9Trainer.compile_train_spec(host)
    assert spec is not None

    x = torch.randn(2, 3, 64, 64)
    targets = torch.zeros(2, 3, 5)
    targets[:, 0] = torch.tensor([1.0, 0.2, 0.2, 0.6, 0.6])
    expected = reference(x, targets=targets)
    got = spec.assemble(spec.network(x), x, targets)
    assert expected.keys() == got.keys()
    for key in expected:
        assert torch.equal(expected[key], got[key]), key
    expected["total_loss"].backward()
    got["total_loss"].backward()
    for (name, p_ref), (_, p_got) in zip(
        reference.named_parameters(), model.named_parameters()
    ):
        assert (p_ref.grad is None) == (p_got.grad is None), name
        if p_ref.grad is not None:
            assert torch.equal(p_ref.grad, p_got.grad), name

    model.aux_weight = 0.0
    assert YOLO9Trainer.compile_train_spec(host) is None


def test_compiler_fallback_hands_cuda_graph_to_the_capture_manager(monkeypatch):
    """compile=True with cuda_graph=True: if compilation fails, the eager
    capture manager takes over instead of losing graphs for the run."""
    model = _Toy()
    host, _ = _toy_host(model)
    monkeypatch.setattr(
        compile_mod.torch,
        "compile",
        lambda network, **kw: MagicMock(side_effect=torch._dynamo.exc.TorchDynamoException("x")),
    )
    host._train_compiler = compile_mod.TrainCompiler("default", cuda_graph=True, accum_steps=1)
    host.config = SimpleNamespace(cuda_graph=True)
    host.device = torch.device("cuda")
    host.is_distributed = False
    host.distiller = None
    host._accum_steps = 1
    host._cuda_graph_manager = None
    host._start_cuda_graph_manager = lambda: BaseTrainer._start_cuda_graph_manager(host)
    BaseTrainer._forward_train(host, torch.randn(2, 4), torch.randn(2, 3))
    assert host._train_compiler is None
    assert host._cuda_graph_manager is not None
