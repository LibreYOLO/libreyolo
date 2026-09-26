"""Opt-in ``torch.compile`` of the training network (``train(compile=...)``).

The compiled region is the family's network/loss boundary (the same split
CUDA graph capture uses, via ``BaseTrainer.compile_train_spec``): the network
forward and its backward are compiled by Inductor, while target assignment,
the loss, the optimizer step, EMA and the LR schedule stay eager. The live
model object is never wrapped or rebound, so optimizer parameters, EMA,
validation, checkpoints (no ``_orig_mod.`` prefixes) and export are the eager
ones.

Unsupported runs (non-CUDA devices, distributed training, distillation,
families or tasks without a boundary) train eagerly after one warning. A
compiler failure also degrades to eager for the rest of the run; errors
raised while running compiled kernels (out of memory, CUDA faults) propagate.
"""

from __future__ import annotations

import contextlib
import logging

import torch
import torch._dynamo

from .config import normalize_compile

logger = logging.getLogger(__name__)


def _unsupported_reason(trainer) -> str | None:
    if trainer.device.type != "cuda":
        # CPU Inductor training measured 15-60x slower than eager on RF-DETR.
        return f"device {trainer.device.type!r} is not CUDA"
    if getattr(trainer, "is_distributed", False):
        return "distributed training is not supported yet"
    if getattr(trainer, "distiller", None) is not None:
        return "distillation runs are not supported"
    return None


def _inductor_options(
    mode: str, *, cuda_graph: bool, accum_steps: int, dynamic: bool | None
) -> tuple[dict, bool]:
    """Expand ``mode`` into Inductor options and decide on CUDA graph trees.

    ``torch.compile`` accepts ``mode`` or ``options``, not both, so the mode
    is expanded here and graph replay is set explicitly. Graph trees keep the
    backward's gradients in the graph's memory pool, so gradient accumulation
    across replays is unsafe and turns them off.
    """
    from torch._inductor import list_mode_options

    options = dict(list_mode_options(mode))
    wants_graphs = cuda_graph or bool(options.get("triton.cudagraphs", False))
    reason = None
    if accum_steps > 1:
        reason = f"gradient accumulation is on ({accum_steps} micro-batches per step)"
    elif not callable(getattr(torch.compiler, "cudagraph_mark_step_begin", None)):
        reason = "this PyTorch has no torch.compiler.cudagraph_mark_step_begin"
    use_graphs = wants_graphs and reason is None
    if wants_graphs and reason is not None:
        logger.warning(
            "compile: CUDA graph replay disabled because %s; compiling without it.",
            reason,
        )
    options["triton.cudagraphs"] = use_graphs
    if dynamic:
        # Unsupported with dynamic shapes, yet it still runs and can assert,
        # sending whole frames back to eager. Absent on older torch.
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config.triton, "coalesce_tiling_analysis"):
            options["triton.coalesce_tiling_analysis"] = False
    return options, use_graphs


def _eager_backward_lowering():
    """Compile the backward with the forward, so its failures are catchable.

    AOTAutograd otherwise lowers a static-shape backward lazily, inside
    ``loss.backward()``, where a compiler failure would end the run instead
    of falling back to eager.
    """
    import torch._functorch.config as functorch_config

    if hasattr(functorch_config, "force_non_lazy_backward_lowering"):
        return functorch_config.patch(force_non_lazy_backward_lowering=True)
    return contextlib.nullcontext()


class TrainCompiler:
    """Owns the compiled network callable for one training run."""

    def __init__(self, mode: str, *, cuda_graph: bool, accum_steps: int):
        self.mode = mode
        self.cuda_graph = cuda_graph
        self.accum_steps = accum_steps
        self.spec = None
        self.disabled = False
        self.cudagraphs = False
        self._compiled = None

    def _disable(self, reason: str) -> None:
        self.disabled = True
        self._compiled = None
        logger.warning("compile=%r: training runs eager (%s).", self.mode, reason)

    def _build(self, trainer) -> None:
        spec = trainer.compile_train_spec()
        if spec is None:
            self._disable(
                f"{type(trainer).__name__} has no compiled training boundary "
                "for this task or network variant"
            )
            return
        dynamic = trainer.compile_dynamic()
        options, self.cudagraphs = _inductor_options(
            self.mode,
            cuda_graph=self.cuda_graph,
            accum_steps=self.accum_steps,
            dynamic=dynamic,
        )
        self.spec = spec
        # Compiling the adapter, not the model: the model keeps its eager
        # forward for validation, EMA and export.
        self._compiled = torch.compile(spec.network, dynamic=dynamic, options=options)
        logger.info(
            "compile=%r: compiling the training network (dynamic=%s, CUDA graph "
            "replay=%s); the first steps include compilation time.",
            self.mode,
            dynamic,
            self.cudagraphs,
        )

    def run(self, trainer, imgs: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
        """Run the compiled network, or return None to run this batch eagerly."""
        if self.disabled:
            return None
        if self._compiled is None:
            try:
                self._build(trainer)
            except Exception as exc:  # noqa: BLE001 - optional speed feature
                self._disable(f"setup failed: {exc!r}")
            if self.disabled:
                return None
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        try:
            with _eager_backward_lowering():
                return self._compiled(imgs)
        except torch._dynamo.exc.TorchDynamoException as exc:
            # Raised while tracing or compiling a frame, before its kernels
            # run. Frames compiled earlier in this call may already have
            # run: their autograd graph is dropped, so the eager retry only
            # repeats BatchNorm running-stat updates for this batch.
            self._disable(f"compilation failed: {type(exc).__name__}: {exc}")
            return None


def build_train_compiler(trainer) -> TrainCompiler | None:
    """Return the run's compiler, or None (with a warning) when unsupported."""
    mode = normalize_compile(getattr(trainer.config, "compile", False))
    if mode is False:
        return None
    reason = _unsupported_reason(trainer)
    if reason is not None:
        logger.warning("compile=%r ignored (%s); training runs eager.", mode, reason)
        return None
    return TrainCompiler(
        mode,
        cuda_graph=bool(getattr(trainer.config, "cuda_graph", False)),
        accum_steps=trainer._accum_steps,
    )
