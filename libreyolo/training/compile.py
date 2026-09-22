"""Opt-in Inductor compilation of the training network, with an eager loss.

The callable belongs to the trainer, not the model. Optimizer parameter
identity, EMA, validation and state-dict names therefore stay unchanged.
CUDA graphs, when requested, belong exclusively to Inductor.
"""

from __future__ import annotations

import logging

import torch

from .config import normalize_compile
from .cuda_graph import CudaGraphTrainSpec

logger = logging.getLogger(__name__)


def compiler_options(
    mode: str,
    *,
    device: torch.device,
    cuda_graph: bool,
    accumulation_steps: int,
) -> tuple[dict, bool]:
    """Resolve a mode without nesting Inductor and eager CUDA graphs."""
    from torch._inductor import list_mode_options

    # torch.compile accepts mode OR options. Expand the documented mode
    # options before setting capture explicitly, preserving autotuning knobs.
    options = dict(list_mode_options(mode))
    wants_graphs = cuda_graph or bool(options.get("triton.cudagraphs", False))
    reason = None
    if device.type != "cuda":
        reason = "the device is not CUDA"
    elif accumulation_steps > 1:
        reason = "gradient accumulation is enabled"
    elif not callable(getattr(torch.compiler, "cudagraph_mark_step_begin", None)):
        reason = "this PyTorch version has no CUDA graph step marker"
    use_graphs = wants_graphs and reason is None
    options["triton.cudagraphs"] = use_graphs
    if wants_graphs and reason is not None:
        logger.warning(
            "Compiler CUDA graphs disabled because %s; retaining compile=%r.",
            reason,
            mode,
        )
    return options, use_graphs


class TrainCompileManager:
    """Compile one existing family network/loss split on the first batch.

    Only the first forward can fall back after a compiler failure. Buffer
    and RNG snapshots make that eager retry safe for BatchNorm and dropout.
    After a successful compiled forward, failures propagate: replaying a
    partially executed forward/backward could alter the training trajectory.
    No warm-up optimizer step, extra backward, or global compiler setting is
    used. Inductor may itself keep unsupported regions eager.
    """

    def __init__(
        self,
        spec: CudaGraphTrainSpec,
        *,
        mode: str,
        options: dict,
        cuda_graph: bool = False,
    ):
        self.spec = spec
        self.mode = mode
        self.options = options
        self.cuda_graph = cuda_graph
        self.compiled = None
        self.disabled = False
        self.started = False

    def run(self, imgs, targets, polygons=None):
        if self.disabled:
            return None
        if self.compiled is None:
            try:
                # A bound method avoids installing an OptimizedModule on the
                # live model or rebinding its forward for validation/EMA.
                self.compiled = torch.compile(
                    self.spec.network.forward,
                    backend="inductor",
                    fullgraph=False,
                    dynamic=None,
                    options=self.options,
                )
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 - optional compiler startup
                self._disable(exc)
                return None
        if self.cuda_graph:
            torch.compiler.cudagraph_mark_step_begin()
        if self.started:
            flat = self.compiled(imgs)
        else:
            buffers = [(b, b.detach().clone()) for b in self.spec.network.buffers()]
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(imgs.device) if imgs.is_cuda else None
            try:
                flat = self.compiled(imgs)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 - restore before the eager retry
                with torch.no_grad():
                    for buffer, saved in buffers:
                        buffer.copy_(saved)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng, imgs.device)
                self._disable(exc)
                return None
            self.started = True
            logger.info(
                "Training network compile=%r started (Inductor CUDA graphs requested=%s); "
                "criterion and optimizer remain eager.",
                self.mode,
                self.cuda_graph,
            )
        # Keep matching, loss-side host control flow and logging out of the
        # compiled network. Criterion failures are never compiler fallbacks.
        return self.spec.assemble(flat, imgs, targets, polygons)

    def _disable(self, exc: Exception) -> None:
        self.compiled = None
        self.disabled = True
        logger.warning(
            "Training compile=%r could not start (%s: %s); using eager training.",
            self.mode,
            type(exc).__name__,
            exc,
        )


def build_compile_manager(trainer) -> TrainCompileManager | None:
    """Resolve the supported training boundary after model/EMA setup.

    The initial supported boundary is the existing YOLO9/RF-DETR detection
    network split. DDP cannot bypass its wrapper, and distillation/LoRA/QAT
    introduce extra hooks or mutable state; these run eager with a warning.
    """
    mode = normalize_compile(getattr(trainer.config, "compile", False))
    if mode is False:
        return None
    family = trainer.get_model_family()
    wrapper = getattr(trainer, "wrapper_model", None)
    reason = None
    if trainer.device.type not in ("cpu", "cuda"):
        reason = (
            f"device {trainer.device.type!r} is outside the CPU/CUDA training boundary"
        )
    elif getattr(trainer, "is_distributed", False):
        reason = "distributed training has no compiled network/loss boundary yet"
    elif getattr(trainer, "distiller", None) is not None:
        reason = "distillation hooks require the eager training path"
    elif getattr(trainer.config, "lora", False):
        reason = "LoRA training requires the eager training path"
    elif getattr(wrapper, "_quant_manifest", None):
        reason = "quantization-aware training requires the eager training path"
    elif family not in ("yolo9", "rfdetr"):
        reason = f"family {family!r} has no compiled training boundary yet"
    elif not callable(getattr(torch, "compile", None)):
        reason = "torch.compile is unavailable"
    if reason is not None:
        logger.warning("compile=%r ignored (%s); using eager training.", mode, reason)
        return None
    try:
        spec = trainer.cuda_graph_train_spec()
        if spec is None:
            logger.warning(
                "compile=%r ignored (%s task or network variant has no supported "
                "training boundary); using eager training.",
                mode,
                family,
            )
            return None
        options, use_graphs = compiler_options(
            mode,
            device=trainer.device,
            cuda_graph=bool(getattr(trainer.config, "cuda_graph", False)),
            accumulation_steps=trainer._accum_steps,
        )
    except Exception as exc:  # noqa: BLE001 - unavailable compiler/spec uses eager
        logger.warning(
            "compile=%r setup failed (%s: %s); using eager training.",
            mode,
            type(exc).__name__,
            exc,
        )
        return None
    if family == "rfdetr":
        logger.info(
            "RF-DETR compiled regions use portable deformable attention; "
            "optional native attention providers decline compiler tracing."
        )
    return TrainCompileManager(spec, mode=mode, options=options, cuda_graph=use_graphs)
