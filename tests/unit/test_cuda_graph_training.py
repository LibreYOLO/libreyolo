"""Tests for CUDA graph capture of the training step.

CPU tests cover the dispatch machinery (tree flattening, shape counting,
fallback guarantees, trainer routing, family gating). CUDA tests gate the
core promise: enabling ``cuda_graph`` must not change training numerics,
so eager and graphed runs are compared step by step, loss and parameters
both, for YOLO9 and RF-DETR.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from libreyolo.training.cuda_graph import (
    CudaGraphTrainSpec,
    GraphableNetwork,
    TrainGraphManager,
    _thread_local_capture_errors,
    flatten_tree,
    unflatten_tree,
)
from libreyolo.training.trainer import BaseTrainer

pytestmark = pytest.mark.unit

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


# =============================================================================
# Tree flattening
# =============================================================================


class TestTreeFlatten:
    def test_roundtrip_nested(self):
        a, b, c = torch.zeros(1), torch.ones(2), torch.full((3,), 2.0)
        tree = {
            "pred": a,
            "aux": [{"x": b}, {"x": c}],
            "meta": (None, 7, "tag"),
        }
        flat, skeleton = flatten_tree(tree)
        assert len(flat) == 3
        rebuilt = unflatten_tree(skeleton, flat)
        # Identity, not equality: autograd connectivity depends on the
        # rebuilt tree containing the same tensor objects.
        assert rebuilt["pred"] is a
        assert rebuilt["aux"][0]["x"] is b
        assert rebuilt["aux"][1]["x"] is c
        assert rebuilt["meta"] == (None, 7, "tag")

    def test_plain_list(self):
        t = [torch.zeros(1), torch.zeros(2)]
        flat, skeleton = flatten_tree(t)
        rebuilt = unflatten_tree(skeleton, flat)
        assert isinstance(rebuilt, list)
        assert rebuilt[0] is t[0] and rebuilt[1] is t[1]

    def test_graphable_network_adapter(self):
        class Toy(nn.Module):
            def forward(self, x):
                return {"a": x * 2, "b": [x + 1, None]}

        net = GraphableNetwork(Toy())
        x = torch.arange(4.0)
        flat = net(x)
        assert isinstance(flat, tuple) and len(flat) == 2
        rebuilt = net.rebuild(flat)
        assert torch.equal(rebuilt["a"], x * 2)
        assert rebuilt["b"][1] is None
        # The wrapped module's parameters are visible through the adapter,
        # which is what lets capture pass them as graph inputs.
        assert isinstance(net.module, Toy)


# =============================================================================
# Manager dispatch
# =============================================================================


def _fake_cuda_batch(shape=(2, 3, 64, 64)):
    """A stand-in for a CUDA batch tensor, usable on CPU-only machines."""
    imgs = MagicMock(spec=torch.Tensor)
    imgs.is_cuda = True
    imgs.shape = torch.Size(shape)
    imgs.dtype = torch.float32
    imgs.device = "cuda:0"
    return imgs


def _spec():
    return CudaGraphTrainSpec(network=MagicMock(), assemble=MagicMock())


class TestTrainGraphManager:
    def test_non_cuda_input_disables(self):
        manager = TrainGraphManager()
        out = manager.run(_spec(), torch.zeros(1, 3, 8, 8))
        assert out is None
        assert manager.disabled

    def test_captures_after_threshold_and_replays(self):
        manager = TrainGraphManager(warmup_threshold=3)
        spec = _spec()
        imgs = _fake_cuda_batch()
        graphed = MagicMock(return_value=("flat",))
        with patch(
            "torch.cuda.make_graphed_callables", return_value=graphed
        ) as make:
            assert manager.run(spec, imgs) is None
            assert manager.run(spec, imgs) is None
            assert not manager.captured
            out = manager.run(spec, imgs)
        assert out == ("flat",)
        assert manager.captured
        make.assert_called_once()
        # Subsequent same-shape batches replay without recapturing.
        with patch("torch.cuda.make_graphed_callables") as make_again:
            assert manager.run(spec, imgs) == ("flat",)
        make_again.assert_not_called()

    def test_shape_mismatch_falls_back_eager(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        with patch(
            "torch.cuda.make_graphed_callables",
            return_value=MagicMock(return_value=("flat",)),
        ):
            assert manager.run(spec, _fake_cuda_batch((2, 3, 64, 64))) == ("flat",)
        # A different shape (multi-scale batch, last partial batch) must run
        # eager without disabling the captured graph.
        assert manager.run(spec, _fake_cuda_batch((1, 3, 64, 64))) is None
        assert not manager.disabled
        assert manager.run(spec, _fake_cuda_batch((2, 3, 64, 64))) == ("flat",)

    def test_capture_failure_disables_permanently(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        imgs = _fake_cuda_batch()
        with patch(
            "torch.cuda.make_graphed_callables", side_effect=RuntimeError("boom")
        ):
            assert manager.run(spec, imgs) is None
        assert manager.disabled
        # Disabled means no further capture attempts at all.
        with patch("torch.cuda.make_graphed_callables") as make:
            assert manager.run(spec, imgs) is None
        make.assert_not_called()

    def test_buffer_snapshot_failure_disables_instead_of_raising(self):
        """An OOM cloning the BatchNorm snapshot must not kill the run.

        The snapshot allocates a copy of every module buffer, on device, at
        the moment memory is tightest: immediately before capture reserves
        static input, output and workspace buffers for a whole forward and
        backward. It used to run outside the capture guard, so that OOM
        propagated out of the training step and ended the run. An opt-in
        speed flag must degrade to eager, never to a crash.
        """
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        with patch.object(
            TrainGraphManager,
            "_snapshot_buffers",
            side_effect=torch.cuda.OutOfMemoryError("snapshot OOM"),
        ):
            assert manager.run(spec, _fake_cuda_batch()) is None
        assert manager.disabled

    def test_buffer_snapshot_failure_skips_restore(self):
        """With no snapshot taken there is nothing to restore.

        The handler restores unconditionally, so the failure path must leave
        an empty snapshot rather than a half-built one it would then try to
        copy back into live buffers.
        """
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        with patch.object(
            TrainGraphManager,
            "_snapshot_buffers",
            side_effect=torch.cuda.OutOfMemoryError("snapshot OOM"),
        ):
            with patch.object(TrainGraphManager, "_restore_buffers") as restore:
                assert manager.run(spec, _fake_cuda_batch()) is None
        restore.assert_called_once_with([])

    def test_replay_failure_disables(self):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = _spec()
        imgs = _fake_cuda_batch()
        graphed = MagicMock(side_effect=[("flat",), RuntimeError("stale")])
        with patch("torch.cuda.make_graphed_callables", return_value=graphed):
            assert manager.run(spec, imgs) == ("flat",)
        assert manager.run(spec, imgs) is None
        assert manager.disabled


# =============================================================================
# Invalidation (mid-run changes to what the captured region computes)
# =============================================================================
#
# YOLOX turns on its L1 branch when mosaic closes, which adds tensors to the
# network's output. Replaying the pre-switch graph past that point would keep
# training the pre-switch network, so the family trainer invalidates and the
# manager re-captures.


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return (self.lin(x),)


class TestCaptureInvalidation:
    def _captured(self, network):
        manager = TrainGraphManager(warmup_threshold=1)
        spec = CudaGraphTrainSpec(network=network, assemble=MagicMock())
        imgs = _fake_cuda_batch()

        def fake_make_graphed(module, sample, **kwargs):
            # Mirror the real call's in-place rebind of module.forward.
            module.forward = MagicMock(return_value=("flat",))
            return module

        with patch(
            "torch.cuda.make_graphed_callables", side_effect=fake_make_graphed
        ):
            assert manager.run(spec, imgs) == ("flat",)
        return manager, spec, imgs

    def test_invalidate_restores_eager_forward_and_recaptures(self):
        network = _ToyNet()
        eager_forward = network.forward
        manager, spec, imgs = self._captured(network)
        assert manager.captured
        assert "forward" in network.__dict__  # capture rebound it

        manager.invalidate("mosaic close")
        assert not manager.captured
        assert not manager.disabled
        # The eager forward is back: a re-capture's warm-up must not replay
        # the dead graph (that fails with "Cannot prepare for replay during
        # capturing stage" and permanently disables capture).
        assert "forward" not in network.__dict__
        assert network.forward.__func__ is eager_forward.__func__

        with patch(
            "torch.cuda.make_graphed_callables",
            return_value=MagicMock(return_value=("flat2",)),
        ) as make:
            assert manager.run(spec, imgs) == ("flat2",)
        make.assert_called_once()

    def test_invalidate_resets_the_warmup_count(self):
        manager = TrainGraphManager(warmup_threshold=3)
        spec = _spec()
        imgs = _fake_cuda_batch()
        assert manager.run(spec, imgs) is None
        assert manager.run(spec, imgs) is None
        manager.invalidate("nothing captured yet")
        # Counting starts over, so a switch landing near the end of a run
        # never pays for a capture it cannot amortise.
        with patch("torch.cuda.make_graphed_callables") as make:
            assert manager.run(spec, imgs) is None
            assert manager.run(spec, imgs) is None
        make.assert_not_called()

    def test_invalidate_after_disable_is_a_noop(self):
        manager = TrainGraphManager(warmup_threshold=1)
        manager.disabled = True
        manager.invalidate("late switch")
        assert manager.disabled
        assert not manager.captured

    def test_capture_failure_restores_eager_forward(self):
        network = _ToyNet()
        manager = TrainGraphManager(warmup_threshold=1)
        spec = CudaGraphTrainSpec(network=network, assemble=MagicMock())
        with patch(
            "torch.cuda.make_graphed_callables", side_effect=RuntimeError("boom")
        ):
            assert manager.run(spec, _fake_cuda_batch()) is None
        assert manager.disabled
        # The eager fallback must call the real forward, not a half-built graph.
        assert "forward" not in network.__dict__

    def test_trainer_hook_routes_to_manager(self):
        manager = TrainGraphManager()
        host = SimpleNamespace(_cuda_graph_manager=manager)
        with patch.object(TrainGraphManager, "invalidate") as invalidate:
            BaseTrainer.invalidate_cuda_graph(host, "because")
        invalidate.assert_called_once_with("because")

    def test_trainer_hook_without_manager_is_a_noop(self):
        BaseTrainer.invalidate_cuda_graph(SimpleNamespace(), "because")


# =============================================================================
# Gradient ownership across replays (gradient accumulation)
# =============================================================================
#
# ``make_graphed_callables``' backward returns ``static_grad.detach()`` for
# every parameter. With ``.grad`` None, autograd adopts that tensor as the
# parameter's ``.grad``, so the next replay overwrites it and then adds its
# own gradient: an accumulation window of graphed micro-batches produced
# ``2 * g_last`` instead of ``sum(g)``. The fake below keeps exactly that
# backward contract so the bug reproduces on CPU.


class _CudaLikeBatch(torch.Tensor):
    """A real CPU tensor that passes the manager's CUDA gate."""

    @property
    def is_cuda(self):
        return True


class _StaticGradLinear(torch.autograd.Function):
    """Linear whose backward writes and returns the same buffers every call,
    as ``make_graphed_callables``' ``Graphed.backward`` does."""

    @staticmethod
    def forward(ctx, static_grads, x, weight, bias):
        ctx.static_grads = static_grads
        ctx.save_for_backward(x)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        weight_grad, bias_grad = ctx.static_grads
        weight_grad.copy_(grad_out.t() @ x)
        bias_grad.copy_(grad_out.sum(0))
        return None, None, weight_grad.detach(), bias_grad.detach()


def _fake_make_graphed_static_grads(module, sample, **kwargs):
    lin = module.module.lin
    static_grads = [torch.zeros_like(lin.weight), torch.zeros_like(lin.bias)]

    def replay(imgs):
        x = imgs.as_subclass(torch.Tensor)
        return (_StaticGradLinear.apply(static_grads, x, lin.weight, lin.bias),)

    module.forward = replay
    return module


class TestAccumulatedGradOwnership:
    def _window_grads(self, preserve, windows=2, accum=4):
        torch.manual_seed(0)
        network = _ToyNet()
        reference = _ToyNet()
        reference.load_state_dict(network.state_dict())
        manager = TrainGraphManager(
            warmup_threshold=1, preserve_accumulated_grads=preserve
        )
        spec = CudaGraphTrainSpec(
            network=GraphableNetwork(network), assemble=MagicMock()
        )
        batches = [torch.randn(2, 4) for _ in range(windows * accum)]
        results = []
        with patch(
            "torch.cuda.make_graphed_callables",
            side_effect=_fake_make_graphed_static_grads,
        ):
            for window in range(windows):
                network.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                for step in range(accum):
                    x = batches[window * accum + step]
                    flat = manager.run(spec, x.as_subclass(_CudaLikeBatch))
                    assert flat is not None  # every micro-batch is graphed
                    (flat[0].pow(2).sum() / accum).backward()
                    (reference(x)[0].pow(2).sum() / accum).backward()
                results.append(
                    (network.lin.weight.grad.clone(), reference.lin.weight.grad)
                )
        return results

    def test_replayed_microbatches_accumulate_like_eager(self):
        for graphed, eager in self._window_grads(preserve=True):
            torch.testing.assert_close(graphed, eager)

    def test_without_ownership_the_window_is_corrupted(self):
        """Documents the failure the ownership step prevents."""
        graphed, eager = self._window_grads(preserve=False)[0]
        assert not torch.allclose(graphed, eager)

    def test_no_copy_when_accumulation_is_off(self):
        network = _ToyNet()
        network.lin.weight.grad = torch.ones_like(network.lin.weight)
        grad = network.lin.weight.grad
        TrainGraphManager(preserve_accumulated_grads=False)._own_accumulated_grads(
            network
        )
        assert network.lin.weight.grad is grad
        TrainGraphManager()._own_accumulated_grads(network)
        assert network.lin.weight.grad is not grad
        torch.testing.assert_close(network.lin.weight.grad, grad)


# =============================================================================
# Stochastic-layer detection
# =============================================================================
#
# Capture does not disable dropout or stochastic depth, but a replayed graph
# consumes the generator on its own schedule, so it does not reproduce the
# sequence an eager step would draw. Families that have such layers inside the
# captured region are statistically equivalent, not bit-identical, and the
# manager says so once at capture time rather than leaving it to be found in
# a diff.


class TestStochasticLayerDetection:
    def test_clean_network_reports_none(self):
        assert TrainGraphManager._stochastic_layers(_ToyNet()) == []

    def test_active_dropout_is_reported(self):
        net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        assert TrainGraphManager._stochastic_layers(net) == ["1"]

    def test_disabled_dropout_is_not_reported(self):
        net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.0))
        assert TrainGraphManager._stochastic_layers(net) == []

    def test_stochastic_depth_is_reported_by_name(self):
        class DropPath(nn.Module):
            def __init__(self, drop_prob):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return x

        net = nn.Sequential(DropPath(0.1), DropPath(0.0))
        assert TrainGraphManager._stochastic_layers(net) == ["0"]


# =============================================================================
# Capture error mode (pin-memory thread race)
# =============================================================================
#
# Capture defaults to capture_error_mode="global", under which a cudaHostAlloc
# from a DataLoader pin-memory thread invalidates the capture AND poisons the
# pin thread, killing the run on the next batch fetch (seen twice on the first
# RF100-VL Vast campaign as "AcceleratorError ... in pin memory thread").
# Capture must therefore run in "thread_local" mode, where other threads'
# CUDA calls proceed normally.


class TestThreadLocalCaptureErrors:
    def test_swaps_and_restores_cudagraph_symbol(self):
        original = torch.cuda.CUDAGraph
        with _thread_local_capture_errors():
            assert torch.cuda.CUDAGraph is not original
            assert issubclass(torch.cuda.CUDAGraph, original)
        assert torch.cuda.CUDAGraph is original

    def test_restores_on_exception(self):
        original = torch.cuda.CUDAGraph
        with pytest.raises(RuntimeError, match="boom"):
            with _thread_local_capture_errors():
                raise RuntimeError("boom")
        assert torch.cuda.CUDAGraph is original

    def test_concurrent_contexts_serialize_and_restore_the_original(self):
        """Unsynchronized, two overlapping contexts would each save a
        different "original" (the second saves the first's subclass) and
        the interleaved restores could leave a temporary subclass installed
        for the rest of the process. The patch lock must serialize them:
        every context sees the true original as its base class, and the
        true original is what remains at the end."""
        import threading
        import time

        original = torch.cuda.CUDAGraph
        bases = []

        def use_context():
            with _thread_local_capture_errors():
                bases.append(torch.cuda.CUDAGraph.__mro__[1])
                time.sleep(0.02)

        threads = [threading.Thread(target=use_context) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert torch.cuda.CUDAGraph is original
        assert bases == [original] * 4

    def test_forces_thread_local_at_capture_begin(self):
        class FakeGraph:
            def capture_begin(self, *args, **kwargs):
                self.begin_kwargs = kwargs

        with patch("torch.cuda.CUDAGraph", FakeGraph):
            with _thread_local_capture_errors():
                graph = torch.cuda.CUDAGraph()
                # torch.cuda.graph.__enter__ passes the default explicitly;
                # the subclass must override it.
                graph.capture_begin(capture_error_mode="global")
        assert graph.begin_kwargs["capture_error_mode"] == "thread_local"

    def test_manager_captures_under_the_patch(self):
        """The capture path must instantiate the patched graph class."""
        manager = TrainGraphManager(warmup_threshold=1)
        original = torch.cuda.CUDAGraph
        seen = {}

        def record_class(*args, **kwargs):
            seen["cls"] = torch.cuda.CUDAGraph
            return MagicMock(return_value=("flat",))

        with patch("torch.cuda.make_graphed_callables", side_effect=record_class):
            assert manager.run(_spec(), _fake_cuda_batch()) == ("flat",)
        assert seen["cls"] is not original
        assert issubclass(seen["cls"], original)
        assert torch.cuda.CUDAGraph is original

    @requires_cuda
    def test_capture_survives_concurrent_pinned_allocations(self):
        """Regression for the campaign killer: capture while a stand-in for
        the DataLoader pin-memory thread keeps calling cudaHostAlloc."""
        import threading

        net = GraphableNetwork(nn.Conv2d(3, 8, 3, padding=1).cuda())
        spec = CudaGraphTrainSpec(
            network=net,
            assemble=lambda flat, *args: {"total_loss": flat[0].sum()},
        )
        manager = TrainGraphManager(warmup_threshold=1)
        imgs = torch.randn(2, 3, 32, 32, device="cuda")

        stop = threading.Event()
        errors: list[BaseException] = []

        def hammer():
            held = []
            size = 0
            while not stop.is_set():
                try:
                    # Strictly growing sizes defeat the caching host
                    # allocator, so every iteration is a real cudaHostAlloc,
                    # exactly what the pin-memory thread issues per batch.
                    held.append(
                        torch.empty(4096 + size * 640, dtype=torch.uint8).pin_memory()
                    )
                    size += 1
                except BaseException as exc:  # must never happen
                    errors.append(exc)
                    return

        thread = threading.Thread(target=hammer, daemon=True)
        thread.start()
        try:
            out = manager.run(spec, imgs)
        finally:
            stop.set()
            thread.join(timeout=30)

        assert not errors, f"pin-memory stand-in thread was poisoned: {errors[0]!r}"
        assert out is not None
        assert manager.captured and not manager.disabled
        # And the graph still replays.
        assert manager.run(spec, imgs) is not None


# =============================================================================
# Trainer routing
# =============================================================================


class _RoutingHost:
    """Minimal stand-in exercising BaseTrainer._forward_train unbound."""

    def __init__(self, manager, spec):
        self._cuda_graph_manager = manager
        self._cuda_graph_spec = None
        self._cuda_graph_spec_resolved = False
        self._spec_to_return = spec
        self.on_forward_calls = 0

    def cuda_graph_train_spec(self):
        return self._spec_to_return

    def on_forward(self, imgs, targets, polygons=None):
        self.on_forward_calls += 1
        return {"total_loss": torch.zeros(())}


class TestForwardTrainRouting:
    def test_no_manager_goes_eager(self):
        host = _RoutingHost(manager=None, spec=None)
        out = BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        assert "total_loss" in out

    def test_family_without_spec_disables_and_goes_eager(self):
        manager = TrainGraphManager()
        host = _RoutingHost(manager=manager, spec=None)
        BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        assert manager.disabled

    def test_spec_used_when_graph_runs(self):
        manager = TrainGraphManager()
        flat = (torch.ones(2),)
        assembled = {"total_loss": torch.ones(())}
        spec = CudaGraphTrainSpec(
            network=MagicMock(), assemble=MagicMock(return_value=assembled)
        )
        host = _RoutingHost(manager=manager, spec=spec)
        with patch.object(TrainGraphManager, "run", return_value=flat):
            out = BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert out is assembled
        assert host.on_forward_calls == 0
        spec.assemble.assert_called_once()

    def test_graph_miss_falls_back_to_on_forward(self):
        manager = TrainGraphManager()
        spec = CudaGraphTrainSpec(network=MagicMock(), assemble=MagicMock())
        host = _RoutingHost(manager=manager, spec=spec)
        with patch.object(TrainGraphManager, "run", return_value=None):
            BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert host.on_forward_calls == 1
        spec.assemble.assert_not_called()

    def test_spec_resolution_exception_disables(self):
        manager = TrainGraphManager()
        host = _RoutingHost(manager=manager, spec=None)
        host.cuda_graph_train_spec = MagicMock(side_effect=RuntimeError("nope"))
        BaseTrainer._forward_train(host, torch.zeros(1), torch.zeros(1))
        assert manager.disabled
        assert host.on_forward_calls == 1


# =============================================================================
# Family gating
# =============================================================================


class TestYolo9SpecGating:
    def _host(self, task="detect"):
        from libreyolo.models.yolo9.nn import LibreYOLO9Model
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=LibreYOLO9Model(config="t", nb_classes=3),
        )
        return YOLO9Trainer.cuda_graph_train_spec, host

    def test_detect_supported(self):
        fn, host = self._host()
        spec = fn(host)
        assert spec is not None
        assert spec.network.module is host.model

    def test_non_detect_task_unsupported(self):
        fn, host = self._host(task="pose")
        assert fn(host) is None

    def test_derived_head_unsupported(self):
        fn, host = self._host()
        # Subclassed heads (e2e dual assignment) compute loss at a
        # different boundary; the exact-type gate must reject them.
        class DerivedHead(type(host.model.head)):
            pass

        derived = DerivedHead.__new__(DerivedHead)
        derived.__dict__.update(host.model.head.__dict__)
        host.model.head = derived
        assert fn(host) is None

    def test_pgi_aux_unsupported(self):
        fn, host = self._host()
        host.model.enable_aux(0.25)
        assert fn(host) is None


class TestRFDETRSpecGating:
    def _host(self, task="detect", **model_flags):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        # __new__ dodges the heavy DINOv2 build; gating only reads flags.
        model = object.__new__(LibreRFDETRModel)
        flags = {
            "segmentation": False,
            "pose": False,
            "obb": False,
            "classification": False,
            "semantic": False,
        }
        flags.update(model_flags)
        for key, value in flags.items():
            setattr(model, key, value)
        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=model,
            criterion=MagicMock(weight_dict={}),
            _targets_to_rfdetr_list=MagicMock(),
        )
        return RFDETRTrainer.cuda_graph_train_spec, host

    def test_detect_supported(self):
        fn, host = self._host()
        spec = fn(host)
        assert spec is not None
        assert spec.network.module is host.model

    def test_task_variants_unsupported(self):
        for flag in ("segmentation", "pose", "obb", "classification"):
            fn, host = self._host(**{flag: True})
            assert fn(host) is None, flag
        fn, host = self._host(task="segment")
        assert fn(host) is None

    def test_missing_criterion_unsupported(self):
        fn, host = self._host()
        host.criterion = None
        assert fn(host) is None


class TestClassifyMixinGating:
    """One mixin serves ResNet, ConvNeXt, MobileNetV4 and EfficientNetV2.

    All four train on ``F.cross_entropy(model(imgs), targets)``, so the
    network never reads the labels and the whole of it is capturable.
    """

    def _host(self, task="classify", model=None):
        from libreyolo.models.base.classify_cuda_graph import ClassifyCudaGraphMixin

        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=nn.Linear(4, 3) if model is None else model,
        )
        return ClassifyCudaGraphMixin.cuda_graph_train_spec, host

    def test_classify_supported(self):
        fn, host = self._host()
        spec = fn(host)
        assert spec is not None
        assert spec.network.module is host.model

    def test_other_tasks_unsupported(self):
        for task in ("detect", "segment", "semantic"):
            fn, host = self._host(task=task)
            assert fn(host) is None, task

    def test_non_module_unsupported(self):
        fn, host = self._host(model="not a module")
        assert fn(host) is None

    def test_assemble_matches_the_eager_loss(self):
        import torch.nn.functional as F

        fn, host = self._host()
        spec = fn(host)
        imgs = torch.randn(2, 4)
        targets = torch.tensor([0, 2])
        logits = host.model(imgs)
        spec.network(imgs)  # records the output skeleton
        out = spec.assemble((logits,), imgs, targets)
        assert torch.equal(out["total_loss"], F.cross_entropy(logits, targets))
        assert "loss_ce" in out

    def test_every_classify_trainer_inherits_it(self):
        from libreyolo.models.base.classify_cuda_graph import ClassifyCudaGraphMixin
        from libreyolo.models.convnext.trainer import ConvNeXtTrainer
        from libreyolo.models.efficientnetv2.trainer import EfficientNetV2Trainer
        from libreyolo.models.mobilenetv4.trainer import MobileNetV4Trainer
        from libreyolo.models.resnet.trainer import ResNetTrainer

        for trainer in (
            ResNetTrainer,
            ConvNeXtTrainer,
            MobileNetV4Trainer,
            EfficientNetV2Trainer,
        ):
            assert issubclass(trainer, ClassifyCudaGraphMixin), trainer.__name__


class TestSemanticMixinGating:
    def _host(self, task="semantic", model=None):
        from libreyolo.models.base.semantic_cuda_graph import (
            SemanticLogitsCudaGraphMixin,
        )

        class Net(nn.Module):
            def forward_logits(self, imgs):
                return imgs * 2

            def loss_from_logits(self, logits, targets):
                return {"total_loss": logits.sum(), "sem": logits.sum()}

        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=Net() if model is None else model,
        )
        return SemanticLogitsCudaGraphMixin.cuda_graph_train_spec, host

    def test_semantic_supported(self):
        fn, host = self._host()
        assert fn(host) is not None

    def test_other_tasks_unsupported(self):
        fn, host = self._host(task="detect")
        assert fn(host) is None

    def test_network_without_the_split_unsupported(self):
        fn, host = self._host(model=nn.Linear(4, 4))
        assert fn(host) is None

    def test_both_families_expose_the_split(self):
        from libreyolo.models.lingbotvision.nn import LingBotVisionSemanticSegmenter
        from libreyolo.models.segformer.nn import LibreSegformerNet

        for cls in (LibreSegformerNet, LingBotVisionSemanticSegmenter):
            assert hasattr(cls, "forward_logits"), cls.__name__
            assert hasattr(cls, "loss_from_logits"), cls.__name__


class TestDETRMixinGating:
    """One mixin serves D-FINE, DEIM, DEIMv2, RT-DETR v1/v2/v4 and EC.

    The decoder reads the ground truth to size its denoising queries, so the
    capture stops at the encoder.
    """

    def _host(self, task="detect", model=None, criterion=MagicMock()):
        from libreyolo.models.base.detr_cuda_graph import DETREncoderCudaGraphMixin

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Identity()
                self.encoder = nn.Identity()
                self.decoder = nn.Identity()

        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task=task),
            model=Net() if model is None else model,
            criterion=criterion,
        )
        return DETREncoderCudaGraphMixin.cuda_graph_train_spec, host

    def test_detect_supported(self):
        fn, host = self._host()
        assert fn(host) is not None

    def test_other_tasks_unsupported(self):
        for task in ("segment", "pose", "semantic"):
            fn, host = self._host(task=task)
            assert fn(host) is None, task

    def test_model_without_the_trio_unsupported(self):
        fn, host = self._host(model=nn.Linear(4, 4))
        assert fn(host) is None

    def test_missing_criterion_unsupported(self):
        fn, host = self._host(criterion=None)
        assert fn(host) is None

    def test_every_detr_trainer_inherits_it(self):
        from libreyolo.models.base.detr_cuda_graph import DETREncoderCudaGraphMixin
        from libreyolo.models.deim.trainer import DEIMTrainer
        from libreyolo.models.deimv2.trainer import DEIMv2Trainer
        from libreyolo.models.dfine.trainer import DFINETrainer
        from libreyolo.models.ec.trainer import ECTrainer
        from libreyolo.models.rtdetr.trainer import RTDETRTrainer
        from libreyolo.models.rtdetrv2.trainer import RTDETRv2Trainer
        from libreyolo.models.rtdetrv4.trainer import RTDETRv4Trainer

        for trainer in (
            DFINETrainer,
            DEIMTrainer,
            DEIMv2Trainer,
            ECTrainer,
            RTDETRTrainer,
            RTDETRv2Trainer,
            RTDETRv4Trainer,
        ):
            assert issubclass(trainer, DETREncoderCudaGraphMixin), trainer.__name__

    def test_capture_half_runs_the_families_own_forward(self):
        """The capturable half must not re-derive the family's prefix.

        DEIMv2's backbone emits more maps than its encoder consumes without
        splitting a low-level feature off, which a hand-rolled prefix got
        wrong (it passed ``low_level_feat=`` to a decoder that has no such
        argument). Running the real forward with the decoder stubbed keeps
        every family's own branching.
        """
        from libreyolo.models.base.detr_cuda_graph import _BackboneEncoder

        seen = {}

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Identity()
                self.encoder = nn.Identity()
                self.decoder = nn.Identity()

            def forward(self, x, targets=None):
                a, b = x, x * 2
                return self.decoder([a, b], targets=targets, extra=x * 3)

        net = Net()
        original_decoder_forward = net.decoder.forward
        adapter = _BackboneEncoder(net)
        flat = adapter(torch.ones(2))
        assert len(flat) == 3  # two feats plus the one tensor kwarg
        feats, kwargs = adapter.split(list(flat))
        assert len(feats) == 2
        assert set(kwargs) == {"extra"}
        assert torch.equal(kwargs["extra"], torch.full((2,), 3.0))
        # The stub is removed again: the decoder must be callable afterwards.
        assert net.decoder.forward.__func__ is original_decoder_forward.__func__
        seen.clear()

    def test_shared_tensor_kwarg_is_not_duplicated(self):
        """EC hands ``feats[0]`` to its mask head; it must not be emitted twice.

        ``make_graphed_callables`` returns one static output buffer per
        output tensor, and the same tensor appearing twice would give the
        eager half two aliases of one buffer instead of the two distinct
        values the family expects.
        """
        from libreyolo.models.base.detr_cuda_graph import _BackboneEncoder

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Identity()
                self.encoder = nn.Identity()
                self.decoder = nn.Identity()

            def forward(self, x, targets=None):
                feats = [x, x * 2]
                return self.decoder(feats, targets=targets, spatial_feat=feats[0])

        adapter = _BackboneEncoder(Net())
        flat = adapter(torch.ones(2))
        assert len(flat) == 2
        feats, kwargs = adapter.split(list(flat))
        assert kwargs["spatial_feat"] is feats[0]


# =============================================================================
# CUDA parity: enabling cuda_graph must not change training numerics
# =============================================================================


def _yolo9_targets(bsz, device, generator):
    t = torch.zeros(bsz, 50, 5)
    n = 10
    cls = torch.randint(0, 3, (bsz, n), generator=generator).float()
    cx = torch.rand(bsz, n, generator=generator) * 0.8 + 0.1
    cy = torch.rand(bsz, n, generator=generator) * 0.8 + 0.1
    w = torch.rand(bsz, n, generator=generator) * 0.2 + 0.05
    h = torch.rand(bsz, n, generator=generator) * 0.2 + 0.05
    t[:, :n, 0] = cls
    t[:, :n, 1] = (cx - w / 2).clamp(0, 1)
    t[:, :n, 2] = (cy - h / 2).clamp(0, 1)
    t[:, :n, 3] = (cx + w / 2).clamp(0, 1)
    t[:, :n, 4] = (cy + h / 2).clamp(0, 1)
    return t.to(device)


def _run_steps(model, forward_fn, imgs, targets, steps):
    """Shared harness: SGD + AMP GradScaler loop, mirrors the trainer."""
    from torch.amp import GradScaler, autocast

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = GradScaler("cuda")
    losses = []
    for _ in range(steps):
        with autocast("cuda", cache_enabled=False):
            outputs = forward_fn(imgs, targets)
            loss = outputs["total_loss"]
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))
    return losses


@requires_cuda
class TestCudaParityYolo9:
    def test_loss_trajectory_identical(self):
        from libreyolo.models.yolo9.nn import LibreYOLO9Model
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        bsz, size, steps = 2, 128, 6
        gen = torch.Generator().manual_seed(3)
        imgs = torch.randn(bsz, 3, size, size, generator=gen).cuda()
        targets = _yolo9_targets(bsz, "cuda", gen)

        def build():
            torch.manual_seed(11)
            model = LibreYOLO9Model(config="t", nb_classes=3).cuda().train()
            return model

        # Eager arm mirrors on_forward exactly.
        model_e = build()
        eager = _run_steps(
            model_e,
            lambda i, t: model_e(i, targets=t),
            imgs,
            targets,
            steps,
        )

        # Graphed arm goes through the real spec + manager.
        model_g = build()
        host = SimpleNamespace(
            wrapper_model=SimpleNamespace(task="detect"), model=model_g
        )
        spec = YOLO9Trainer.cuda_graph_train_spec(host)
        assert spec is not None
        manager = TrainGraphManager(warmup_threshold=1)

        def graphed_forward(i, t):
            flat = manager.run(spec, i)
            assert flat is not None
            return spec.assemble(flat, i, t)

        graphed = _run_steps(model_g, graphed_forward, imgs, targets, steps)

        assert manager.captured
        assert eager == pytest.approx(graphed, rel=0, abs=0), (
            f"eager {eager} != graphed {graphed}"
        )
        # Parameters must match exactly after the same number of steps.
        for (name, pe), (_, pg) in zip(
            model_e.named_parameters(), model_g.named_parameters()
        ):
            assert torch.equal(pe, pg), name
        # Buffers too: capture warm-up must not leave extra BatchNorm
        # running-stat updates behind (the manager snapshots and restores
        # them), or validation, EMA and checkpoints would drift from eager.
        buffer_names = 0
        for (name, be), (_, bg) in zip(
            model_e.named_buffers(), model_g.named_buffers()
        ):
            assert torch.equal(be, bg), name
            buffer_names += 1
        assert buffer_names > 0, "expected BatchNorm buffers to compare"


@requires_cuda
class TestCudaParityRFDETR:
    """RF-DETR parity, with a tolerance matched to eager's own noise floor.

    Unlike YOLO9, RF-DETR training is not bitwise reproducible even eager
    to eager: the deformable-attention backward accumulates with atomics,
    so two identical seeded eager runs diverge from step 1 (measured max
    relative difference about 4e-4 over 4 steps on an RTX 5070 Ti). The
    graph contract therefore is: the first step, whose forward and loss
    run on identical weights, must match bit for bit, and the trajectory
    must stay within the eager run-to-run noise band. A real gradient bug
    (wrong boundary, stale buffers) diverges orders of magnitude faster.
    """

    # The DINOv2 backbone build fetches pretrained weights.
    @pytest.mark.external_data
    def test_loss_trajectory_identical(self):
        from libreyolo.models.rfdetr.nn import LibreRFDETRModel
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        steps = 4

        def build():
            torch.manual_seed(23)
            model = LibreRFDETRModel(config="n", nb_classes=3, device="cuda")
            model = model.cuda().train()
            criterion, _ = model.build_criterion_and_postprocess()
            criterion.to("cuda")
            return model, criterion

        model_e, criterion_e = build()
        size = model_e.resolution
        bsz = 2
        gen = torch.Generator().manual_seed(5)
        imgs = torch.randn(bsz, 3, size, size, generator=gen).cuda()
        # The RF-DETR target converter expects pixel coordinates.
        targets = _yolo9_targets(bsz, "cuda", gen)
        targets[..., 1:5] *= size

        def make_host(model, criterion):
            host = SimpleNamespace(
                wrapper_model=SimpleNamespace(task="detect"),
                model=model,
                criterion=criterion,
                device=torch.device("cuda"),
            )
            host._targets_to_rfdetr_list = (
                lambda *args, **kwargs: RFDETRTrainer._targets_to_rfdetr_list(
                    host, *args, **kwargs
                )
            )
            return host

        host_e = make_host(model_e, criterion_e)

        def eager_forward(i, t):
            return RFDETRTrainer.on_forward(host_e, i, t)

        eager = _run_steps(model_e, eager_forward, imgs, targets, steps)

        model_g, criterion_g = build()
        host_g = make_host(model_g, criterion_g)
        spec = RFDETRTrainer.cuda_graph_train_spec(host_g)
        assert spec is not None
        manager = TrainGraphManager(warmup_threshold=1)

        def graphed_forward(i, t):
            flat = manager.run(spec, i)
            assert flat is not None
            return spec.assemble(flat, i, t)

        graphed = _run_steps(model_g, graphed_forward, imgs, targets, steps)

        assert manager.captured
        assert eager[0] == graphed[0], (
            f"step-0 loss must be bit-identical: {eager[0]} != {graphed[0]}"
        )
        assert eager == pytest.approx(graphed, rel=5e-3), (
            f"trajectory outside eager noise band: eager {eager} != "
            f"graphed {graphed}"
        )
