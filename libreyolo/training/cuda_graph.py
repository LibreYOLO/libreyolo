"""CUDA graph capture for the training-step network forward and backward.

Small and mid-size detectors are launch-bound during training: the host
spends more time submitting kernel launches for the forward and backward
passes than the GPU spends executing them, so the GPU idles between
launches. Capturing the network's forward and backward into CUDA graphs
(via ``torch.cuda.make_graphed_callables``) collapses thousands of per-step
launches into two replays.

Only the *network* is captured. The loss stays eager by design: detection
losses are data-dependent (boolean-mask selects, Hungarian matching,
host-side branching on assignment results), which is both illegal during
graph capture and pointless to capture. The optimizer step, gradient
clipping, EMA update and LR schedule also stay eager; they are a small
fraction of step time and interact with GradScaler host-side logic.

Enablement is opt-in per run (``train(..., cuda_graph=True)``). Each model
family that supports capture provides a :class:`CudaGraphTrainSpec` through
``BaseTrainer.cuda_graph_train_spec``; families without a spec silently run
eager, so the flag is always safe to pass.

Shape handling is dispatch-based, not constraint-based: a graph is valid
for exactly the input shape it was captured with, so the manager counts
shapes and captures one graph once a shape has repeated
``warmup_threshold`` times. Batches at any other shape (multi-scale
batches, the last partial batch of an epoch) run eager, unchanged, so a
shape the graph does not cover costs speed and nothing else.

Capture-time contracts (enforced by the trainer, documented here):

- Capture happens lazily on a real training batch, after ``setup()`` has
  finished every model mutation (weight loading, freezing, device moves).
  Replacing a parameter tensor *after* capture would leave the graph
  reading freed storage; in-place updates (optimizer steps, EMA source
  reads, ``load_state_dict``) are safe because replay reads the same
  addresses.
- ``make_graphed_callables`` warm-up runs a few forward/backward passes on
  the capture batch. Its backward goes through ``torch.autograd.grad`` with
  ``only_inputs=True``, so it never writes ``.grad`` and cannot pollute an
  accumulation window (which zeroes only at the window's start, not per
  micro-batch). Stateful buffers (BatchNorm running stats) *are* advanced by
  those extra forwards, so they are snapshotted before capture and restored
  in place afterwards and the buffer trajectory matches eager exactly.
- Replay, unlike warm-up, *does* hand gradients to autograd: the graphed
  backward returns its static gradient buffers. When a parameter's ``.grad``
  is None, autograd adopts that buffer as ``.grad`` instead of copying it,
  and the next replay overwrites it before accumulating, so ``g0 + g1``
  silently becomes ``2 * g1``. Under gradient accumulation the manager
  therefore gives every live ``.grad`` its own storage before each replay
  (one gradient-sized copy per micro-batch after the first of a window).
  Without accumulation the loop drops ``.grad`` between forward and
  backward, so nothing can be overwritten and no copy is made.
- Under AMP, capture and replay must run with autocast caching disabled;
  the trainer's autocast context handles this when a manager is active.
- Distributed training and distillation are not captured in this version;
  the trainer falls back to eager for those runs.
- A family whose captured region stops matching what the loss needs partway
  through a run (YOLOX adds its L1 branch at mosaic close) calls
  ``BaseTrainer.invalidate_cuda_graph``; the manager drops the graph,
  restores the eager forward and re-captures once a shape settles again.

Any failure to capture, at any point, permanently falls back to eager
training for the rest of the run with a single warning.

Capture is bit-identical for most families. Three documented exceptions,
all inherent rather than defects: families whose own eager training is not
reproducible (deformable-attention atomics, TF32 reduction order) stay
inside that spread; RTMDet's cross-level shared head convolutions sum their
three gradient contributions in a different order; and a network with
dropout or stochastic depth inside the captured region draws its own random
stream on replay (the manager logs this at capture time). See
``docs/training_cuda_graphs.md`` for the per-family table, and
``tests/e2e/test_cuda_graph_training_families.py`` for the gates.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Serializes the torch.cuda.CUDAGraph swap below. Without it, two host
# threads capturing concurrently would each save a different "original"
# (the second thread saves the first thread's subclass) and the interleaved
# restores could leave a temporary subclass installed for the rest of the
# process. Capture is rare and takes seconds, so holding the lock across
# the whole capture is fine.
_capture_patch_lock = threading.Lock()


@contextmanager
def _thread_local_capture_errors() -> Iterator[None]:
    """Make ``make_graphed_callables`` capture with ``thread_local`` errors.

    Capture defaults to ``capture_error_mode="global"``, under which a CUDA
    call from *any* host thread invalidates the capture. A DataLoader with
    ``pin_memory=True`` keeps a pin-memory thread alive that calls
    ``cudaHostAlloc`` whenever it stages the next prefetched batch, so a
    lazy capture taken on a live training batch races with it: the capture
    dies with ``cudaErrorStreamCaptureUnsupported`` and, worse, the error
    poisons the pin-memory thread, which re-raises on the next batch fetch
    and kills the whole run (observed twice on the first RF100-VL Vast
    campaign, as "AcceleratorError ... in pin memory thread").

    ``thread_local`` is PyTorch's documented mode for exactly this
    situation: only the capturing thread is restricted during capture, and
    other threads' CUDA calls proceed normally without being captured. The
    pin-memory thread touches only its own pinned host buffers, never the
    capturing stream, so nothing it does belongs in the graph.

    ``make_graphed_callables`` constructs its own ``CUDAGraph`` objects and
    exposes no way to choose the mode, so for the duration of the call the
    ``torch.cuda.CUDAGraph`` symbol it instantiates is swapped for a
    subclass that forces the mode at ``capture_begin``.
    """
    with _capture_patch_lock:
        original = torch.cuda.CUDAGraph

        class _ThreadLocalCaptureGraph(original):  # type: ignore[misc, valid-type]
            def capture_begin(self, *args: Any, **kwargs: Any) -> None:
                kwargs["capture_error_mode"] = "thread_local"
                super().capture_begin(*args, **kwargs)

        torch.cuda.CUDAGraph = _ThreadLocalCaptureGraph
        try:
            yield
        finally:
            torch.cuda.CUDAGraph = original

# A shape must repeat this many times before capture pays off. One-shot
# shapes (a lone odd-size batch) then never capture, while any steady-state
# loop converges within the first few steps of epoch 0.
DEFAULT_WARMUP_THRESHOLD = 3
# Warm-up iterations make_graphed_callables runs before capture, so lazy
# allocations, cuDNN benchmark picks and autotuning settle first.
DEFAULT_NUM_WARMUP_ITERS = 3

GraphKey = Tuple[Tuple[int, ...], torch.dtype, str]


# =============================================================================
# Output-tree flattening
# =============================================================================
#
# ``make_graphed_callables`` requires the captured callable to return a
# tensor or a tuple of tensors, while family networks return lists (YOLO
# heads) or nested dicts (DETR-style ``{"pred_logits": ..., "aux_outputs":
# [{...}, ...]}``). These helpers split any nesting of dict/list/tuple into
# a flat tensor tuple plus a rebuildable skeleton. Autograd connectivity is
# preserved because unflattening reinserts the *same* tensor objects the
# graphed callable returned.


@dataclass(frozen=True)
class _Leaf:
    """Placeholder for tensor index ``index`` in a flattened tree."""

    index: int


def flatten_tree(node: Any) -> Tuple[List[torch.Tensor], Any]:
    """Split a nested container of tensors into (tensors, skeleton)."""
    tensors: List[torch.Tensor] = []

    def walk(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
            return _Leaf(len(tensors) - 1)
        if isinstance(item, tuple):
            return tuple(walk(v) for v in item)
        if isinstance(item, list):
            return [walk(v) for v in item]
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        # Non-tensor leaves (None, numbers) are baked into the skeleton.
        return item

    return tensors, walk(node)


def unflatten_tree(skeleton: Any, tensors: List[torch.Tensor]) -> Any:
    """Rebuild the original nesting from a skeleton and a flat tensor list."""
    if isinstance(skeleton, _Leaf):
        return tensors[skeleton.index]
    if isinstance(skeleton, tuple):
        return tuple(unflatten_tree(v, tensors) for v in skeleton)
    if isinstance(skeleton, list):
        return [unflatten_tree(v, tensors) for v in skeleton]
    if isinstance(skeleton, dict):
        return {k: unflatten_tree(v, tensors) for k, v in skeleton.items()}
    return skeleton


# =============================================================================
# Family spec
# =============================================================================


class GraphableNetwork(nn.Module):
    """Adapter giving any family network the captured-callable contract.

    ``make_graphed_callables`` rebinds the forward of whatever callable it
    is handed, and requires that forward to return a tensor tuple. Family
    models return lists (YOLO heads) or nested dicts (DETR outputs) and
    their forwards must stay eager for validation, EMA evaluation and
    checkpointing, so capture always goes through this adapter instead:
    it flattens the wrapped module's output to a tuple and remembers the
    nesting, and ``rebuild`` restores it for the eager loss.

    The output *structure* must be static across calls (tensor shapes are
    already static by the capture contract); the skeleton recorded at
    capture time is reused for every replay.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.skeleton: Any = None

    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        flat, self.skeleton = flatten_tree(self.module(imgs))
        return tuple(flat)

    def rebuild(self, flat: Tuple[torch.Tensor, ...]) -> Any:
        return unflatten_tree(self.skeleton, list(flat))


@dataclass
class CudaGraphTrainSpec:
    """What a model family hands the trainer to enable training capture.

    Attributes:
        network: Module whose ``forward(imgs)`` runs the full trainable
            network and returns a flat tuple of tensors. Use
            :class:`GraphableNetwork` (or another dedicated adapter),
            never the family model itself: ``make_graphed_callables``
            rebinds the callable's forward, and the family model's forward
            must stay eager for validation, EMA evaluation and
            checkpointing.
        assemble: Called eagerly as ``assemble(flat, imgs, targets,
            polygons)`` with the network's output tuple; runs the loss and
            returns the same outputs dict ``on_forward`` would produce
            (must include ``"total_loss"``).
    """

    network: nn.Module
    assemble: Callable[
        [Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, Any], Dict
    ]


# =============================================================================
# Manager
# =============================================================================


class TrainGraphManager:
    """Owns shape counting, capture and replay for one training run.

    The manager captures at most one graph (the first shape to repeat
    ``warmup_threshold`` times). Every graph pins its own static input,
    output and workspace buffers for the forward *and* backward pass, so a
    per-shape cache like inference uses would multiply peak training
    memory; one graph covers the static-shape case completely and still
    wins on the dominant shape of a multi-scale run.
    """

    def __init__(
        self,
        warmup_threshold: int = DEFAULT_WARMUP_THRESHOLD,
        num_warmup_iters: int = DEFAULT_NUM_WARMUP_ITERS,
        preserve_accumulated_grads: bool = True,
    ):
        self.warmup_threshold = max(1, int(warmup_threshold))
        self.num_warmup_iters = max(1, int(num_warmup_iters))
        # True whenever ``.grad`` can survive from one replay to the next
        # (gradient accumulation). The trainer passes False only for its
        # one-step-per-batch loops, which drop ``.grad`` before every
        # backward and so never need the copy.
        self.preserve_accumulated_grads = bool(preserve_accumulated_grads)
        self.disabled = False
        self._graphed: Optional[Callable] = None
        self._graph_key: Optional[GraphKey] = None
        self._shape_counts: Dict[GraphKey, int] = {}
        self._eager_after_capture = 0
        # ``make_graphed_callables`` rebinds the module's forward in place.
        # Kept so a later invalidate() can put the eager forward back before
        # re-capturing; without it the next capture's warm-up would call the
        # previous graph and die replaying inside an active capture.
        self._captured_network: Optional[nn.Module] = None
        self._forward_before_capture: Optional[Tuple[bool, Any]] = None

    @property
    def captured(self) -> bool:
        return self._graphed is not None

    @staticmethod
    def _key_for(imgs: torch.Tensor) -> GraphKey:
        return (tuple(imgs.shape), imgs.dtype, str(imgs.device))

    def _disable(self, reason: str) -> None:
        self.disabled = True
        self._graphed = None
        self._restore_eager_forward()
        logger.warning(
            "cuda_graph training capture disabled, falling back to eager: %s",
            reason,
        )

    def _restore_eager_forward(self) -> None:
        """Undo ``make_graphed_callables``' in-place forward rebind.

        It replaces ``module.forward`` with a closure that replays the graph.
        Leaving that in place after dropping a graph is unsafe twice over: a
        re-capture's warm-up would replay the dead graph inside the new
        capture (``Cannot prepare for replay during capturing stage``), and
        the eager fallback would silently keep replaying it.
        """
        network = self._captured_network
        if network is None:
            return
        had_own, original = self._forward_before_capture or (False, None)
        if had_own and original is not None:
            network.forward = original
        else:
            network.__dict__.pop("forward", None)
        self._captured_network = None
        self._forward_before_capture = None

    def _own_accumulated_grads(self, network: nn.Module) -> None:
        """Give every live ``.grad`` its own storage before a replay.

        The graphed backward returns its static gradient buffers. A parameter
        whose ``.grad`` was None adopts that buffer as its ``.grad`` (autograd
        steals rather than copies a gradient nothing else references), so
        the next replay would overwrite the accumulated value and then add
        its own gradient on top, turning ``g0 + g1`` into ``2 * g1``. Cloning
        first makes the replay accumulate into memory the graph never writes.
        """
        if not self.preserve_accumulated_grads:
            return
        for param in network.parameters():
            if param.grad is not None:
                param.grad = param.grad.clone()

    def invalidate(self, reason: str) -> None:
        """Drop the captured graph and re-capture on a later batch.

        A graph is only valid while the captured region's op sequence is
        fixed. Some families change that sequence *during* a run: YOLOX
        turns on its L1 branch when mosaic closes, which adds tensors to
        the network's output. Replaying the old graph past such a switch
        would silently keep training the pre-switch network, so the family
        trainer calls this at the switch and the manager re-captures once
        the new shape has settled.

        Shape counts are reset too, so the warm-up threshold applies again
        and a switch that lands on the last few batches of a run never pays
        for a capture it cannot amortise.
        """
        if self.disabled or self._graphed is None:
            self._shape_counts.clear()
            return
        self._graphed = None
        self._graph_key = None
        self._shape_counts.clear()
        self._restore_eager_forward()
        logger.info("cuda_graph: capture invalidated (%s); will re-capture", reason)

    def run(
        self, spec: CudaGraphTrainSpec, imgs: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Run the network under a graph if possible.

        Returns the network's flat output tuple, or ``None`` when the
        caller must run this batch eagerly (not captured yet, shape
        mismatch, or capture permanently disabled).
        """
        if self.disabled:
            return None
        if not imgs.is_cuda:
            self._disable("training batches are not on a CUDA device")
            return None

        key = self._key_for(imgs)

        if self._graphed is not None:
            if key != self._graph_key:
                self._eager_after_capture += 1
                return None
            try:
                self._own_accumulated_grads(spec.network)
                return self._graphed(imgs)
            except Exception as exc:  # replay must never kill a run
                self._disable(f"graph replay failed: {exc!r}")
                return None

        seen = self._shape_counts.get(key, 0) + 1
        self._shape_counts[key] = seen
        if seen < self.warmup_threshold:
            return None
        return self._capture_and_run(spec, imgs)

    @staticmethod
    def _stochastic_layers(network: nn.Module) -> List[str]:
        """Names of active randomness-drawing layers inside the capture.

        Capture does not break these: PyTorch registers the generator with
        the graph and advances its offset per replay, so dropout keeps
        dropping and stochastic depth keeps dropping paths. What changes is
        *which* elements: a replayed graph consumes the generator on its own
        schedule, so it does not reproduce the sequence an eager run of the
        same step would draw.

        The consequence is worth stating out loud rather than discovering
        from a diff: for a network with these layers, a graphed run is
        statistically equivalent to the eager run, and equivalent to
        changing the seed, but its loss trajectory is not the eager one.
        Every family without them stays bit-identical.
        """
        stochastic = []
        for name, module in network.named_modules():
            if isinstance(module, nn.modules.dropout._DropoutNd):
                if float(getattr(module, "p", 0.0)) > 0.0:
                    stochastic.append(name)
                continue
            # Stochastic depth ships under several names across families
            # (DropPath, StochasticDepth, TimmDropPath); they share the
            # drop-probability attribute rather than a base class.
            for attr in ("drop_prob", "p"):
                value = getattr(module, attr, None)
                if (
                    isinstance(value, float)
                    and value > 0.0
                    and "drop" in type(module).__name__.lower()
                ):
                    stochastic.append(name)
                    break
        return stochastic

    @staticmethod
    def _snapshot_buffers(
        network: nn.Module,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Clone every module buffer so warm-up side effects can be undone.

        ``make_graphed_callables`` warm-up runs a few extra forward passes
        on the live model, advancing stateful buffers (BatchNorm running
        mean/var, ``num_batches_tracked``) beyond what one eager step
        performs. Restoring the snapshot afterwards keeps the buffer
        trajectory exactly eager-equivalent, so validation, EMA and
        checkpoints see the same statistics either way. Restoration is
        ``copy_`` in place: the captured kernels record buffer addresses,
        and those must not change.
        """
        return [(buf, buf.detach().clone()) for buf in network.buffers()]

    @staticmethod
    def _restore_buffers(
        snapshot: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        with torch.no_grad():
            for buf, saved in snapshot:
                buf.copy_(saved)

    def _capture_and_run(
        self, spec: CudaGraphTrainSpec, imgs: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        forward_before = (
            "forward" in spec.network.__dict__,
            spec.network.__dict__.get("forward"),
        )
        # Empty until the snapshot succeeds, so the handler below can restore
        # unconditionally: restoring nothing is a no-op.
        buffer_snapshot: List[Tuple[torch.Tensor, torch.Tensor]] = []
        try:
            # Inside the guard on purpose. Cloning every buffer is a real
            # allocation on the device, and it happens at the moment memory
            # is tightest: right before capture reserves static input, output
            # and workspace buffers for a whole forward and backward. An OOM
            # here used to escape and kill the run, which is the one outcome
            # an opt-in speed flag must never produce.
            buffer_snapshot = self._snapshot_buffers(spec.network)
            # The sample clone becomes the graph's static input buffer; the
            # live batch tensor must stay caller-owned. allow_unused_input
            # mirrors eager autograd: parameters a family's training forward
            # never touches (e.g. DETR heads exercised only at inference)
            # get no gradient eagerly either, and the parity tests hold both
            # modes to the same trajectory. The TypeError retry keeps older
            # torch versions without the keyword working.
            sample = (imgs.detach().clone(),)
            with _thread_local_capture_errors():
                try:
                    graphed = torch.cuda.make_graphed_callables(
                        spec.network,
                        sample,
                        num_warmup_iters=self.num_warmup_iters,
                        allow_unused_input=True,
                    )
                except TypeError:
                    graphed = torch.cuda.make_graphed_callables(
                        spec.network,
                        sample,
                        num_warmup_iters=self.num_warmup_iters,
                    )
        except Exception as exc:
            # A partially-applied capture can leave the rebound forward
            # behind; put the eager one back before giving up.
            self._captured_network = spec.network
            self._forward_before_capture = forward_before
            self._disable(f"graph capture failed: {exc!r}")
            # A failed capture can leave asynchronous stream-capture errors
            # queued; drain them here so they surface inside this handler
            # instead of crashing an unrelated later launch, giving the
            # eager fallback a clean stream to run on.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            self._restore_buffers(buffer_snapshot)
            return None

        # Undo warm-up buffer drift before the first replay: the replay
        # below then performs this batch's single BatchNorm update, exactly
        # as the eager step would have.
        self._restore_buffers(buffer_snapshot)

        self._captured_network = spec.network
        self._forward_before_capture = forward_before
        self._graphed = graphed
        self._graph_key = self._key_for(imgs)
        shape = tuple(imgs.shape)
        logger.info(
            "cuda_graph: captured training forward/backward at input shape %s",
            (shape,),
        )
        stochastic = self._stochastic_layers(spec.network)
        if stochastic:
            logger.info(
                "cuda_graph: the captured network has %d randomness-drawing "
                "layers (e.g. %s); replay draws its own random stream, so "
                "this run is statistically equivalent to the eager one but "
                "will not reproduce its exact loss trajectory",
                len(stochastic),
                ", ".join(stochastic[:3]),
            )
        try:
            self._own_accumulated_grads(spec.network)
            return self._graphed(imgs)
        except Exception as exc:
            self._disable(f"first graph replay failed: {exc!r}")
            return None
