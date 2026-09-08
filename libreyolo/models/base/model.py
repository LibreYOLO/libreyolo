"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import math
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import batched_nms

from ...tasks import (
    detect_task_suffix,
    normalize_task,
    resolve_task,
    task_suffix_pattern,
    task_to_suffix,
)
from ...training.config import TrainConfig, load_train_cfg
from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput
from ...utils.logging import ensure_default_logging
from ...utils.model_info import build_model_info, format_model_info
from ...utils.results import Results
from ...utils.serialization import (
    REQUIRED_CHECKPOINT_METADATA_KEYS,
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
)
from ...validation.preprocessors import StandardValPreprocessor
from .cuda_graph import (
    GraphRunner,
    forward_maybe_graphed,
    normalize_cuda_graph_mode,
)

logger = logging.getLogger(__name__)


# Keys that come from the model wrapper instance (``self.size``,
# ``self.nb_classes``) and are passed explicitly to the family trainer. If a
# cfg yaml carries them too, they would collide with the explicit kwargs and
# raise ``TypeError: got multiple values``. ``TrainConfig.to_yaml()`` writes
# both, so a user-generated starter yaml hits this naturally.
_WRAPPER_OWNED_CFG_KEYS = frozenset({"size", "num_classes"})


def _wrap_train_with_cfg(train_fn: Callable) -> Callable:
    """Add shared config-file and scratch-initialization handling to ``train()``.

    Loads the yaml as a dict and merges it into kwargs with user-provided
    kwargs winning. Keys consumed by positional args (and a small set of
    wrapper-owned keys like ``size``/``num_classes``) are dropped from the
    cfg dict to avoid ``TypeError: got multiple values``.
    """
    sig = inspect.signature(train_fn)
    pos_names = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if pos_names and pos_names[0] == "self":
        pos_names = pos_names[1:]

    @functools.wraps(train_fn)
    def wrapper(self, *args, cfg=None, **user_kwargs):
        merged = dict(user_kwargs)
        if cfg is not None:
            cfg_kwargs = load_train_cfg(cfg)
            consumed = set(pos_names[: len(args)]) | _WRAPPER_OWNED_CFG_KEYS
            merged = {k: v for k, v in cfg_kwargs.items() if k not in consumed}
            merged.update(user_kwargs)

        resume = merged.get("resume", False)
        if resume and not merged.get("single_cls", False):
            resume_source = (
                resume
                if isinstance(resume, (str, Path)) and not isinstance(resume, bool)
                else None
            )
            checkpoint_config = self._checkpoint_train_config(resume_source)
            if bool(checkpoint_config.get("single_cls", False)):
                merged["single_cls"] = True

        if merged.get("single_cls"):
            from ..registry import group_of

            group = group_of(self.FAMILY)
            task = getattr(self, "task", "detect")
            if group not in {"g0", "g1"} or task != "detect":
                raise ValueError(
                    "single_cls=True is supported only for G0/G1 detection "
                    f"models; got family={self.FAMILY!r} ({group}), task={task!r}."
                )

        if merged.get("pretrained") is False:
            from ..registry import group_of

            if group_of(self.FAMILY) in {"g0", "g1", "g2"}:
                bound = sig.bind_partial(self, *args, **merged)
                bound.apply_defaults()
                extras = next(
                    (
                        bound.arguments.get(param.name, {})
                        for param in sig.parameters.values()
                        if param.kind == inspect.Parameter.VAR_KEYWORD
                    ),
                    {},
                )
                if bound.arguments.get("resume", extras.get("resume", False)):
                    raise ValueError("pretrained=False cannot be combined with resume.")
                self._reset_for_scratch(
                    seed=bound.arguments.get("seed", extras.get("seed", 0))
                )
                if "pretrained" not in sig.parameters:
                    merged.pop("pretrained")

        return train_fn(self, *args, **merged)

    wrapper._libreyolo_cfg_wrapped = True  # type: ignore[attr-defined]
    return wrapper


class BaseModel(ABC):
    """Abstract base class for LibreYOLO model wrappers.

    Subclasses must implement the abstract methods to provide model-specific
    behavior for initialization, forward pass, and postprocessing.

    Class constants subclasses should set:
        FAMILY: Model family identifier (e.g. "yolox").
        FILENAME_PREFIX: Prefix for weight filenames (e.g. "LibreYOLOX").
        INPUT_SIZES: Mapping of size code to input resolution.
        TRAIN_CONFIG: TrainConfig subclass with family-specific defaults.
        val_preprocessor_class: Preprocessor class for validation.
        validator_class: Override the validator used by val(); defaults to task-based dispatch.
    """

    # Class-level model metadata — subclasses override these
    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    WEIGHT_EXT: ClassVar[str] = ".pt"
    INPUT_SIZES: ClassVar[dict[str, int]] = {}
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("detect",)
    DEFAULT_TASK: ClassVar[str] = "detect"
    # Override when multiple runtime tasks intentionally share one checkpoint
    # artifact. Filename parsing then advertises only the tasks with distinct
    # published weight files while SUPPORTED_TASKS remains the runtime surface.
    WEIGHT_TASKS: ClassVar[Optional[tuple[str, ...]]] = None
    # When True, the task suffix is mandatory in weight filenames (e.g. a
    # classify-only family requires ``-cls``); detect families leave it optional.
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = False
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {}
    TRAIN_CONFIG: ClassVar[Optional[type[TrainConfig]]] = None
    val_preprocessor_class = StandardValPreprocessor
    validator_class: ClassVar[Optional[type]] = None
    # Dataset-variant weight suffixes (e.g. "visdrone" accepts
    # ``LibreYOLO9P2s-visdrone.pt``). Families that publish checkpoints
    # trained on a non-default dataset opt in; the variant stays part of the
    # Hugging Face repo name in ``get_download_url``.
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = ()

    # Batched-predict policy: True when ``_preprocess`` yields stackable
    # (1, C, H, W) tensors and every tensor in the ``_forward`` output keeps
    # a leading batch dim (the contract batched validation already relies
    # on). Set False where that does not hold (e.g. generative VLMs).
    SUPPORTS_BATCHED_PREDICT: ClassVar[bool] = True

    # Family-specific inputs consumed while preparing one prediction image.
    # Guided families opt in explicitly (for example with ``("mask",)`` or
    # ``("trimap",)``); ordinary families accept no auxiliary image inputs.
    # Required keys must also appear in PREDICT_INPUT_KWARGS.
    PREDICT_INPUT_KWARGS: ClassVar[tuple[str, ...]] = ()
    REQUIRED_PREDICT_INPUT_KWARGS: ClassVar[tuple[str, ...]] = ()

    # CUDA-graph policy: True once a family's ``_forward`` is verified to
    # capture and replay bit-identically. Capture forbids host-visible work
    # mid-forward (``.item()``, ``.cpu()``, writing a Python int into a CUDA
    # tensor), so families opt in only after a parity test covers them.
    SUPPORTS_CUDA_GRAPH: ClassVar[bool] = False

    # How a family embeds a finite video under task="embed".
    #   "frames" (default) — one Results per decoded frame, the historical
    #       behavior every existing family keeps.
    #   "clip"            — one Results for the whole clip: frames are sampled
    #       uniformly, encoded, and pooled by the family into a single row.
    # Families opt in explicitly; the generic runner takes the clip path only
    # when the resolved task is "embed", the source is a finite video, and the
    # family declares clip support.
    VIDEO_EMBED_MODE: ClassVar[str] = "frames"

    # TTA policy — subclasses may override
    TTA_ENABLED: ClassVar[bool] = True
    # True for families that resize to a fixed square regardless of input size
    # (DETR-style). Multi-scale TTA is a no-op for them; only flip adds value.
    TTA_FIXED_SIZE: ClassVar[bool] = False
    # Scale factors applied to the PIL image before each TTA pass.
    # Each scale × 2 flips = N passes. Default (1.0,) is flip-only.
    # Override with e.g. (0.83, 1.0, 1.33) for 6-pass multi-scale TTA.
    # Ignored when TTA_FIXED_SIZE is True.
    TTA_SCALES: ClassVar[Tuple[float, ...]] = (1.0,)

    # Model registry — auto-populated by __init_subclass__
    _registry: ClassVar[List[Type["BaseModel"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "can_load")
            and not getattr(cls.can_load, "__isabstractmethod__", False)
            and cls not in BaseModel._registry
        ):
            BaseModel._registry.append(cls)

        if "train" in cls.__dict__ and not getattr(
            cls.train, "_libreyolo_cfg_wrapped", False
        ):
            cls.train = _wrap_train_with_cfg(cls.train)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: Union[str, dict, None],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        ensure_default_logging()
        scratch_init = bool(kwargs.pop("_scratch_init", False))
        self.family = self.FAMILY
        self.task = self._resolve_task(task)
        valid_sizes = self._get_valid_sizes()
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: {', '.join(valid_sizes)}"
            )

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            if isinstance(device, int):
                device = f"cuda:{device}"
            if isinstance(device, str) and device.isdigit():
                device = f"cuda:{device}"
            self.device = torch.device(device)

        self.size = size
        self.nb_classes = nb_classes
        self.input_size = self._get_task_input_sizes()[size]
        self._loaded_checkpoint_train_config: dict[str, Any] | None = None
        # Built lazily on the first cuda_graph=True call so models that never
        # ask for capture pay nothing.
        self._graph_runner: Optional["GraphRunner"] = None
        self._cuda_graph_mode: Optional[str] = None

        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Resolve bare filenames (e.g. "LibreYOLOXn.pt") to weights/ directory
        # so direct instantiation works the same as the factory.
        if isinstance(model_path, str):
            model_path = self._resolve_weights_path(model_path)

        # Signal _init_model that weights will be loaded immediately after, so
        # subclasses can skip pretrained backbone downloads that would be wasted.
        self._loading_from_weights = isinstance(model_path, (str, Path, dict))
        self._initializing_from_scratch = scratch_init
        try:
            self.model = self._init_model()
        finally:
            self._loading_from_weights = False
            self._initializing_from_scratch = False
        self._training_from_scratch = scratch_init

        if model_path is None:
            self.model_path = None
        elif isinstance(model_path, dict):
            self.model_path = None
            self._cache_checkpoint_train_config(model_path)
            state_dict = self._prepare_state_dict(self._strip_ddp_prefix(model_path))
            self._validate_loaded_state_dict_for_task(state_dict, model_path)
            self._prepare_model_for_state_dict(state_dict)
            self._load_state_dict_logged(state_dict, source="state dict")
        else:
            self.model_path = model_path

        if model_path is None:
            self.model.train()
        else:
            self.model.eval()
        self.model.to(self.device)

    @classmethod
    def _from_scratch(
        cls,
        *,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        seed: int = 0,
        **kwargs,
    ) -> "BaseModel":
        """Construct an architecture without loading model or backbone weights."""
        cls._seed_scratch_initialization(seed)
        return cls(
            None,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            _scratch_init=True,
            **kwargs,
        )

    @staticmethod
    def _seed_scratch_initialization(seed: int) -> None:
        if seed is None or int(seed) < 0:
            return
        import random

        import numpy as np

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_weights_path(model_path: str) -> str:
        """Resolve bare filenames (e.g. ``LibreYOLOXn.pt``) to ``weights/`` dir."""
        path = Path(model_path)
        if path.parent == Path(".") and not model_path.startswith(("./", "../")):
            weights_path = Path("weights") / path.name
            if weights_path.exists():
                return str(weights_path)
            if path.exists():
                return str(path)
            return str(weights_path)
        return model_path

    # =========================================================================
    # Abstract interface — subclasses must implement
    # =========================================================================

    @abstractmethod
    def _init_model(self) -> nn.Module:
        """Initialize and return the neural network model."""
        pass

    @abstractmethod
    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        pass

    @staticmethod
    @abstractmethod
    def _get_preprocess_numpy():
        """Return the ``preprocess_numpy(img_rgb_hwc, input_size)`` callable for this model family."""
        pass

    @abstractmethod
    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess image for inference.

        Returns:
            Tuple of (input_tensor, original_image, original_size, ratio).
        """
        pass

    def _preprocess_predict(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
        **predict_input_kwargs,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess one public prediction input.

        The default keeps every existing family on its established
        ``_preprocess`` path. Families that declare auxiliary prediction
        inputs may override this hook to prepare the image and its guide
        together.
        """
        return self._preprocess(
            image,
            color_format,
            input_size=input_size,
            **predict_input_kwargs,
        )

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        pass

    # =========================================================================
    # CUDA graph capture
    # =========================================================================

    def _require_cuda_graph_support(self) -> None:
        if not self.SUPPORTS_CUDA_GRAPH:
            raise NotImplementedError(
                f"cuda_graph is not supported for the '{self.family}' family yet. "
                "Capture requires a forward with no host-visible work, which is "
                "verified per family. Run without cuda_graph=True, or export to "
                "ONNX/TensorRT for a fused deployment path."
            )

    def _get_graph_runner(self) -> "GraphRunner":
        if self._graph_runner is None:
            self._graph_runner = GraphRunner(
                forward_fn=self._forward, family=self.family
            )
        return self._graph_runner

    def _forward_graphed(self, input_tensor: torch.Tensor) -> Any:
        """Run ``_forward``, replaying a captured CUDA graph when in scope."""
        return forward_maybe_graphed(self, input_tensor)

    @contextlib.contextmanager
    def cuda_graph_scope(self, mode: Any = True):
        """Route forwards in this block through captured graphs.

        Predict sets this once per call rather than threading a flag through
        every internal predict path. Validates support up front so an
        unsupported family fails loudly instead of silently running eager.

        Args:
            mode: ``True``/``"on"`` captures on first use, ``"auto"`` waits for
                a shape to repeat, ``False`` is a no-op.
        """
        normalized = normalize_cuda_graph_mode(mode)
        if normalized is None:
            yield
            return
        self._require_cuda_graph_support()
        previous = self._cuda_graph_mode
        self._cuda_graph_mode = normalized
        try:
            yield
        finally:
            self._cuda_graph_mode = previous

    def capture_graph(
        self, imgsz: Optional[int] = None, batch: int = 1, dtype: Any = None
    ) -> None:
        """Capture a CUDA graph now for the given input shape.

        Warmup and capture cost far more than a replay, so call this up front
        when a first-request latency spike matters. Later
        ``predict(..., cuda_graph=True)`` calls at the same shape replay the
        captured graph.

        Args:
            imgsz: Input resolution. Defaults to the model's input size.
            batch: Batch size the graph is captured for. A graph is valid only
                for the exact shape it captured, so this must match how you
                call predict.
            dtype: Input dtype. Defaults to the model's parameter dtype.

        Raises:
            NotImplementedError: If the family has not opted in.
            CudaGraphUnavailable: If capture is impossible or fails.
        """
        self._require_cuda_graph_support()
        size = imgsz or self.input_size
        if dtype is None:
            dtype = next(self.model.parameters()).dtype
        dummy = torch.zeros((batch, 3, size, size), dtype=dtype, device=self.device)
        with torch.no_grad():
            self._get_graph_runner().capture(dummy)

    def graph_info(self) -> Dict[str, Any]:
        """Report captured graphs, replay counts and any eager-fallback reason."""
        if self._graph_runner is None:
            return {
                "family": self.family,
                "supported": self.SUPPORTS_CUDA_GRAPH,
                "captured": [],
                "graph_count": 0,
                "eager_fallbacks": 0,
                "fallback_reason": None,
            }
        info = self._graph_runner.info()
        info["supported"] = self.SUPPORTS_CUDA_GRAPH
        return info

    def release_graphs(self) -> None:
        """Free every captured graph and its static buffers."""
        if self._graph_runner is not None:
            self._graph_runner.release()
            self._graph_runner = None

    def _invalidate_cuda_graphs(self, reason: str) -> None:
        """Drop captured graphs after a change that relocates parameters.

        A graph records memory addresses, not values, so anything that
        *replaces* modules or tensors (quantize, dequantize, a device move, a
        rebuilt head) leaves captured kernels pointing at storage that is stale
        or already freed. In-place weight updates are safe and do not need this;
        replacement does. Every such call site must invalidate, because the
        cache key of shape/dtype/device cannot observe the change on its own.
        """
        if self._graph_runner is None:
            return
        logger.debug("cuda_graph: invalidating captured graphs (%s)", reason)
        self.release_graphs()

    @abstractmethod
    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        """Postprocess model output to detections."""
        pass

    # =========================================================================
    # Concrete defaults — subclasses may override
    # =========================================================================

    def get_distill_config(self) -> Dict:
        """Return distillation config for this model instance.

        Returns:
            Dict with keys:
                - tap_points: List[str] — module paths for forward hooks
                - channels: List[int] — channel dimensions per tap point
                - strides: List[int] — spatial strides per tap point

        Subclasses that support distillation must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_distill_config(). "
            f"Distillation is not yet supported for the '{self.FAMILY}' family."
        )

    def _get_valid_sizes(self) -> List[str]:
        return list(self._get_task_input_sizes().keys())

    @classmethod
    def _supported_tasks(cls) -> tuple[str, ...]:
        return tuple(normalize_task(task) for task in cls.SUPPORTED_TASKS)

    def _resolve_task(self, task: str | None) -> str:
        return resolve_task(
            explicit_task=task,
            default_task=self.DEFAULT_TASK,
            supported_tasks=self.SUPPORTED_TASKS,
        )

    def _get_task_input_sizes(self) -> dict[str, int]:
        if self.TASK_INPUT_SIZES:
            return self.TASK_INPUT_SIZES.get(self.task, self.INPUT_SIZES)
        return self.INPUT_SIZES

    def _get_model_name(self) -> str:
        return self.FAMILY

    def _get_input_size(self) -> int:
        return self.input_size

    def _strict_loading(self) -> bool:
        """Return whether to use strict mode when loading weights."""
        return True

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Transform state dict keys before loading.

        Override in subclasses that need to remap legacy key names.
        """
        return state_dict

    def _checkpoint_train_config(
        self, source: str | Path | dict | None = None
    ) -> dict[str, Any]:
        """Return the saved training config for a checkpoint, if available."""
        if source is None:
            cached = getattr(self, "_loaded_checkpoint_train_config", None)
            if cached is not None:
                return dict(cached)
            source = getattr(self, "model_path", None) or getattr(
                self, "_weight_source", None
            )
        if source is None:
            return {}
        if isinstance(source, dict):
            checkpoint = source
        else:
            path = Path(source)
            if not path.exists():
                return {}
            try:
                checkpoint = load_untrusted_torch_file(
                    path,
                    map_location="cpu",
                    context="checkpoint training-config probe",
                )
            except Exception:
                return {}
        return self._cache_checkpoint_train_config(checkpoint)

    def _cache_checkpoint_train_config(self, checkpoint: Any) -> dict[str, Any]:
        """Cache and return checkpoint training config metadata."""
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        cached = dict(config) if isinstance(config, dict) else {}
        self._loaded_checkpoint_train_config = cached
        return dict(cached)

    def _adapt_checkpoint_num_classes(
        self,
        ckpt_nc: int | None,
        checkpoint_task: str | None = None,
    ) -> int | None:
        """Return the class count to use when adapting checkpoint weights."""
        return ckpt_nc

    def _filter_incoming_state_dict(
        self,
        state_dict: dict,
        *,
        loaded: dict | None = None,
        checkpoint_task: str | None = None,
    ) -> dict:
        """Filter checkpoint tensors before loading.

        Families can override this when a permitted cross-task load keeps the
        reusable backbone/neck tensors but drops incompatible task-specific
        heads.
        """
        return state_dict

    def _prepare_scratch_init(self) -> None:
        """Let wrappers reset checkpoint-derived architecture state."""

    def _is_scratch_build(self) -> bool:
        """Return whether this build belongs to a scratch-training lifecycle."""
        return bool(
            getattr(self, "_initializing_from_scratch", False)
            or getattr(self, "_training_from_scratch", False)
        )

    def _reset_for_scratch(self, *, seed: int = 0) -> None:
        """Replace the loaded network with a freshly initialized architecture."""
        self._seed_scratch_initialization(seed)
        self._prepare_scratch_init()
        self._graph_runner = None
        self._cuda_graph_mode = None
        self._initializing_from_scratch = True
        try:
            model = self._init_model()
        finally:
            self._initializing_from_scratch = False

        self.model = model
        self.model_path = None
        self._training_from_scratch = True
        self.names = (
            {i: name for i, name in enumerate(COCO_CLASSES)}
            if self.nb_classes == 80
            else {i: f"class_{i}" for i in range(self.nb_classes)}
        )
        self.model.train().to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        """Rebuild model with a new class count, preserving weights where shapes match."""
        old_state = self.model.state_dict()
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        # Signal _init_model to skip pretrained backbone downloads — old_state
        # already holds all backbone weights which are restored below, so
        # downloading pretrained weights here is pure waste.
        self._in_rebuild = True
        try:
            self.model = self._init_model()
        finally:
            self._in_rebuild = False

        new_state = self.model.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        self.model.load_state_dict(new_state)
        self.model.to(self.device)

    def _rebuild_for_checkpoint_classes(self, new_nb_classes: int, state_dict: dict):
        """Rebuild for checkpoint class count before loading its state dict."""
        self._rebuild_for_new_classes(new_nb_classes)

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        """Validate task-specific state-dict shape before non-strict loading."""
        return None

    @classmethod
    def _filename_regex(cls) -> Optional[re.Pattern]:
        """Compile regex for matching weight filenames with optional task suffix."""
        if not cls.INPUT_SIZES or not cls.FILENAME_PREFIX:
            return None
        all_sizes = set(cls.INPUT_SIZES)
        for task_sizes in cls.TASK_INPUT_SIZES.values():
            all_sizes.update(task_sizes)
        sizes = sorted(all_sizes, key=len, reverse=True)
        sizes_pattern = "|".join(re.escape(size) for size in sizes)
        prefix = cls.FILENAME_PREFIX.lower()
        ext = re.escape(cls.WEIGHT_EXT)
        suffixes = task_suffix_pattern(cls.WEIGHT_TASKS or cls.SUPPORTED_TASKS)
        if suffixes:
            # Families with no suffixless (detect) task can require the task
            # suffix so that e.g. ``LibreResNet50.pt`` is not accepted as a
            # classify checkpoint -- only ``LibreResNet50-cls.pt`` is canonical.
            optional = "" if getattr(cls, "REQUIRE_TASK_SUFFIX", False) else "?"
            suffix_group = rf"(?P<task>{suffixes}){optional}"
        else:
            suffix_group = ""
        variant_group = ""
        if cls.WEIGHT_VARIANTS:
            variants = "|".join(
                re.escape(variant.lower()) for variant in cls.WEIGHT_VARIANTS
            )
            variant_group = rf"(?P<variant>-(?:{variants}))?"
        return re.compile(
            rf"{prefix}(?P<size>{sizes_pattern}){suffix_group}{variant_group}{ext}"
        )

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Extract model size from a weight filename."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        return m.group("size") if m else None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> Optional[str]:
        """Extract canonical task from a weight filename (e.g. '-seg' -> 'segment')."""
        pattern = cls._filename_regex()
        if pattern is None:
            return detect_task_suffix(filename)
        m = pattern.search(filename.lower())
        task_suffix = m.groupdict().get("task") if m else None
        if task_suffix:
            return normalize_task(task_suffix.lstrip("-"))
        return None

    @classmethod
    def detect_variant_from_filename(cls, filename: str) -> Optional[str]:
        """Extract the dataset-variant suffix from a weight filename, if any."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        variant = m.groupdict().get("variant") if m else None
        return variant.lstrip("-") if variant else None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Return this family's native tensor dict for a recognized upstream layout.

        Called by :mod:`libreyolo.models.autoconvert` on metadata-less
        checkpoints. The default claims layouts whose keys already match the
        native port (``can_load``). Families whose upstream key naming differs
        from the native port override this with a remap, and return ``None``
        for layouts they do not recognize.
        """
        return dict(state_dict) if cls.can_load(state_dict) else None

    @classmethod
    def default_checkpoint_names(cls, nc: int) -> Optional[Dict[int, str]]:
        """Return known upstream labels when a raw checkpoint has no metadata."""
        return None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        """Infer the task from task-specific head keys, or ``None`` if unknown."""
        return None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Return the Hugging Face download URL for the given weight filename."""
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename)
        task_suffix = task_to_suffix(task)
        suffix = f"-{task_suffix}" if task_suffix else ""
        variant = cls.detect_variant_from_filename(filename)
        variant_suffix = f"-{variant}" if variant else ""
        name = f"{cls.FILENAME_PREFIX}{size}{suffix}{variant_suffix}"
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}{cls.WEIGHT_EXT}"

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        """Return an optional warning shown before auto-downloading weights."""
        return None

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        """Verify a freshly auto-downloaded weight file before it is loaded.

        Hook called by ``download_weights`` after a successful download. The
        default trusts LibreYOLO's own Hugging Face mirror and does nothing;
        families that fetch third-party objects (e.g. YOLO-NAS from Deci's CDN)
        override this to checksum-pin the download and fail closed on mismatch.
        """
        return None

    def _get_val_preprocessor(self, img_size: int | None = None):
        """Return the validation preprocessor for this model."""
        if img_size is None:
            img_size = self._get_input_size()
        return self.val_preprocessor_class(img_size=(img_size, img_size))

    # =========================================================================
    # Weight loading internals
    # =========================================================================

    @staticmethod
    def _strip_ddp_prefix(state_dict: dict) -> dict:
        """Strip 'module.' prefix from DDP-wrapped state_dict keys."""
        if any(k.startswith("module.") for k in state_dict):
            return {k.removeprefix("module."): v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def _sanitize_names(names: dict, nc: int) -> Dict[int, str]:
        """Sanitize a class names dict: ensure int keys, fill gaps, trim to nc."""
        sanitized = {}
        for k, v in names.items():
            try:
                sanitized[int(k)] = str(v)
            except (ValueError, TypeError):
                continue

        result = {}
        for i in range(nc):
            result[i] = sanitized.get(i, f"class_{i}")
        return result

    def _load_weights(self, model_path: str):
        """Load model weights from file.

        Handles raw state_dicts and training checkpoint dicts.
        Auto-rebuilds model architecture if checkpoint has different nc.
        Also handles DDP prefix stripping and cross-family rejection.
        """
        path = Path(model_path)
        if not path.exists() and path.parent == Path("."):
            weights_path = Path("weights") / path.name
            if weights_path.exists():
                model_path = str(weights_path)
                path = weights_path

        if not path.exists():
            from ...utils.download import download_weights

            download_weights(model_path, self.size)
            path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        try:
            loaded = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="model weights",
            )
            self._cache_checkpoint_train_config(loaded)

            if isinstance(loaded, dict):
                metadata_keys = set(REQUIRED_CHECKPOINT_METADATA_KEYS) - {"model"}
                if metadata_keys & set(loaded):
                    metadata_errors = validate_checkpoint_metadata(
                        loaded,
                        strict=False,
                    )
                    if metadata_errors:
                        logger.warning(
                            "LibreYOLO checkpoint metadata is missing or incomplete "
                            "for %s: %s. Loading through the legacy compatibility path.",
                            model_path,
                            "; ".join(metadata_errors),
                        )
                if "model" in loaded:
                    state_dict = loaded["model"]
                elif "state_dict" in loaded:
                    state_dict = loaded["state_dict"]
                else:
                    state_dict = loaded

                state_dict = self._prepare_state_dict(
                    self._strip_ddp_prefix(state_dict)
                )

                # Reject cross-family loading
                own_family = self._get_model_name()
                ckpt_family = loaded.get("model_family", "")
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'. "
                        f"Use the correct model class for this checkpoint."
                    )

                normalized_ckpt_task = None
                ckpt_task = loaded.get("task")
                if ckpt_task is not None:
                    normalized_ckpt_task = normalize_task(ckpt_task)
                    if (
                        normalized_ckpt_task != self.task
                        and not self._allow_checkpoint_task_mismatch(
                            normalized_ckpt_task
                        )
                    ):
                        raise RuntimeError(
                            f"Checkpoint was trained for task='{normalized_ckpt_task}' "
                            f"but this model was initialized for task='{self.task}'. "
                            "Pass the matching task or use the correct checkpoint."
                        )

                ckpt_nc = loaded.get("nc")
                ckpt_names = loaded.get("names")

                # Infer nc from names if missing from checkpoint
                if ckpt_nc is None and ckpt_names is not None:
                    ckpt_nc = len(ckpt_names)

                # Infer nc from existing tensor detect_nb_classes
                if ckpt_nc is None and hasattr(self, "detect_nb_classes"):
                    ckpt_nc = self.detect_nb_classes(state_dict)

                ckpt_nc = self._adapt_checkpoint_num_classes(
                    ckpt_nc,
                    normalized_ckpt_task,
                )
                state_dict = self._filter_incoming_state_dict(
                    state_dict,
                    loaded=loaded,
                    checkpoint_task=normalized_ckpt_task,
                )

                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_checkpoint_classes(ckpt_nc, state_dict)

                effective_nc = ckpt_nc if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)
                self._validate_loaded_state_dict_for_task(state_dict, loaded)
            else:
                state_dict = self._prepare_state_dict(loaded)

            quant_manifest = loaded.get("quant") if isinstance(loaded, dict) else None
            if quant_manifest:
                from ...quant import apply_quant_structure

                apply_quant_structure(self, quant_manifest)

            self._prepare_model_for_state_dict(state_dict)
            self._load_state_dict_logged(state_dict, source=str(model_path))
            self.model.to(self.device).eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _load_state_dict_logged(self, state_dict: dict, source: str) -> None:
        """Load with the family's strictness, making silent key drops visible.

        Families that load with strict=False (e.g. YOLOX) previously discarded
        missing/unexpected keys without a trace: a partially matching
        checkpoint would "load" and then quietly predict with fresh-initialized
        tensors wherever keys were absent. Shape mismatches raise regardless of
        strictness, but name mismatches do not, so this logs them. A healthy
        load stays silent.

        Families with custom ``_load_weights`` overrides must either route
        their final ``load_state_dict`` through this helper (YOLO-NAS) or
        police the returned missing/unexpected keys themselves with
        family-specific rules (D-FINE/DEIM/EC tolerate regenerated anchor
        buffers but raise on other unexpected keys; DINOv2/RF-DETR/PIDNet
        validate the key sets against the expected architecture).
        """
        result = self.model.load_state_dict(
            state_dict, strict=self._strict_loading()
        )
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if missing or unexpected:
            logger.warning(
                "Non-strict load from %s: %d missing key(s) (model keeps "
                "initialized weights for these) and %d unexpected key(s) "
                "(ignored). First missing: %s. First unexpected: %s. "
                "This usually indicates a size/task/architecture mismatch.",
                source,
                len(missing),
                len(unexpected),
                missing[:5],
                unexpected[:5],
            )
            logger.debug("All missing keys: %s", missing)
            logger.debug("All unexpected keys: %s", unexpected)

    def _allow_checkpoint_task_mismatch(self, checkpoint_task: str) -> bool:
        """Return whether a family permits loading a checkpoint from another task."""
        return False

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        """Family hook: adapt the live module graph to an incoming state dict.

        Runs after any class-count rebuild and right before
        ``load_state_dict``. Families that support ``lora=True`` and rely on
        the base loader override this to replay adapter injection when the
        checkpoint carries LoRA keys; the default is a no-op.
        """
        return None

    # =========================================================================
    # Public API
    # =========================================================================

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return sorted(self._get_available_layers().keys())

    def info(self, detailed: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """Return model metadata and lightweight architecture counts.

        Args:
            detailed: Include per-parameter rows.
            verbose: Log a human-readable summary.

        Returns:
            JSON-friendly model information dictionary.
        """
        data = build_model_info(self, detailed=detailed)
        if verbose:
            logger.info(format_model_info(data))
        return data

    @property
    def _runner(self):
        if not hasattr(self, "_runner_instance") or self._runner_instance is None:
            from .inference import InferenceRunner

            self._runner_instance = InferenceRunner(self)
        return self._runner_instance

    def __call__(
        self, source=None, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        return self._runner(source, **kwargs)

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def embed(self, source=None, **kwargs) -> torch.Tensor:
        """Return all embedding rows for ``source`` as ``(N_total, D)``.

        This is a convenience wrapper over :meth:`predict`. Models must be
        constructed with ``task="embed"`` so their results populate
        ``Results.embeddings``.
        """
        if "embed" not in self._supported_tasks():
            raise NotImplementedError(
                f"The '{self.family}' family does not support task='embed'."
            )
        from ...utils.results import stack_result_embeddings

        return stack_result_embeddings(self.predict(source, **kwargs))

    def _postprocess_embeddings(
        self,
        output: Any,
        *,
        gallery=None,
        threshold: Optional[float] = 0.4,
    ) -> Dict[str, Any]:
        """Normalize whole-image features and build the shared result payload."""
        threshold = 0.4 if threshold is None else float(threshold)
        features = output[0] if isinstance(output, (list, tuple)) else output
        features = torch.as_tensor(features).float()
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError(
                "Embedding models must emit features with shape (N, D); "
                f"got {tuple(features.shape)}."
            )
        norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if bool((norms <= 1e-12).any()):
            raise ValueError("Embedding model emitted an all-zero feature row.")
        normalized = (features / norms).cpu()
        payload: Dict[str, Any] = {"embeddings": normalized}
        if gallery is not None:
            payload["identities"] = gallery.identify(
                normalized,
                threshold=threshold,
                model=self,
            )
        return payload

    def _predict_augment(
        self,
        image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        **kwargs,
    ) -> Results:
        """Run TTA inference and merge via per-class NMS.

        Scales are read from TTA_SCALES (class variable); each scale x 2 flips
        = one batch of passes. TTA_FIXED_SIZE models always use flip-only.
        """
        if getattr(self, "task", "detect") == "obb":
            raise ValueError(
                "Test-time augmentation does not support oriented boxes yet. "
                "Use augment=False for OBB models."
            )
        if getattr(self, "task", "detect") == "pose":
            raise ValueError(
                "Test-time augmentation does not support pose keypoints yet. "
                "Use augment=False for pose models."
            )
        if getattr(self, "task", "detect") == "point":
            raise ValueError(
                "Test-time augmentation does not support point-task models yet. "
                "Use augment=False for point models."
            )
        if getattr(self, "task", "detect") == "depth":
            raise ValueError(
                "Test-time augmentation does not support depth estimation yet. "
                "Use augment=False for depth models."
            )
        if getattr(self, "task", "detect") == "normal":
            raise ValueError(
                "Test-time augmentation does not support surface normals yet. "
                "Use augment=False for normal models."
            )
        if getattr(self, "task", "detect") == "edge":
            raise ValueError(
                "Test-time augmentation does not support edge detection yet. "
                "Use augment=False for edge models."
            )
        if getattr(self, "task", "detect") == "restore":
            raise ValueError(
                "Test-time augmentation does not support restoration models yet. "
                "Use augment=False for restore models."
            )
        if getattr(self, "task", "detect") == "ocr":
            raise ValueError(
                "Test-time augmentation does not support OCR models yet. "
                "Use augment=False for OCR models."
            )

        from PIL import Image as PILImage
        from ...utils.image_loader import ImageLoader

        effective_imgsz = imgsz if imgsz is not None else self._get_input_size()
        img_pil = ImageLoader.load(image, color_format=color_format)
        image_path = image if isinstance(image, (str, Path)) else None
        orig_w, orig_h = img_pil.size

        if getattr(self, "task", "detect") == "semantic":
            return self._predict_augment_semantic(
                img_pil,
                image_path,
                (orig_w, orig_h),
                effective_imgsz,
                color_format,
                **kwargs,
            )

        if getattr(self, "task", "detect") == "panoptic":
            return self._predict_augment_panoptic(
                img_pil,
                image_path,
                (orig_w, orig_h),
                effective_imgsz,
                color_format,
                **kwargs,
            )

        scales = (1.0,) if self.TTA_FIXED_SIZE else self.TTA_SCALES

        aug_dets = []
        for scale in scales:
            if scale == 1.0:
                scaled = img_pil
            else:
                scaled = img_pil.resize(
                    (int(orig_w * scale), int(orig_h * scale)),
                    PILImage.Resampling.BILINEAR,
                )
            for is_flipped in (False, True):
                src = (
                    scaled.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                    if is_flipped
                    else scaled
                )
                tensor, _, orig_size, ratio = self._preprocess(
                    src, color_format, input_size=effective_imgsz
                )
                with torch.no_grad():
                    raw = self._forward(tensor.to(self.device))
                det = self._postprocess(
                    raw, conf, iou, orig_size, max_det=max_det, ratio=ratio, **kwargs
                )
                aug_dets.append((det, orig_size, is_flipped, scale))

        if getattr(self, "task", "detect") == "classify":
            return self._merge_classify_tta(aug_dets, image_path, (orig_w, orig_h))

        return self._merge_tta(aug_dets, iou, image_path, (orig_w, orig_h), classes)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        """Return raw ``[1, C, H, W]`` semantic logits at ``original_size``.

        Semantic families must implement this: flip TTA merges views before
        the argmax, so it needs the pre-argmax logits that ``_postprocess``
        throws away.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _postprocess_semantic_logits(), "
            "which the semantic task requires (it backs both _postprocess and "
            "flip TTA)."
        )

    def _predict_augment_semantic(
        self,
        img_pil,
        image_path,
        original_size: Tuple[int, int],
        effective_imgsz,
        color_format: str,
        **kwargs,
    ) -> Results:
        """Flip-only TTA for semantic segmentation.

        Runs the image and its horizontal flip, flips the flipped view's
        logits back into alignment, averages softmax probabilities across
        the two views (not raw logits — see ``_postprocess_semantic_logits``
        callers), then argmaxes once. Scale variation (``TTA_SCALES``) does
        not apply to dense per-pixel prediction, so this always runs exactly
        two forward passes regardless of the family's TTA policy flags.
        """
        from PIL import Image as PILImage

        from ...utils.results import Results, SemanticMask
        from ...utils.tta import average_flip_softmax

        orig_w, orig_h = original_size
        logits_views = []
        for is_flipped in (False, True):
            src = (
                img_pil.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
                if is_flipped
                else img_pil
            )
            tensor, _, orig_size, ratio = self._preprocess(
                src, color_format, input_size=effective_imgsz
            )
            with torch.no_grad():
                raw = self._forward(tensor.to(self.device))
            logits = self._postprocess_semantic_logits(
                raw, orig_size, ratio=ratio, input_size=effective_imgsz, **kwargs
            )
            if is_flipped:
                logits = logits.flip(-1)
            logits_views.append(logits)

        avg_probs = average_flip_softmax(logits_views[0], logits_views[1])
        mask = avg_probs.argmax(dim=1)[0].cpu()
        return Results(
            boxes=None,
            orig_shape=(orig_h, orig_w),
            path=str(image_path) if image_path else None,
            names=self.names,
            semantic_mask=SemanticMask(mask.long(), (orig_h, orig_w)),
        )

    def _predict_augment_panoptic(
        self,
        img_pil,
        image_path,
        original_size: Tuple[int, int],
        effective_imgsz,
        color_format: str,
        **kwargs,
    ) -> Results:
        """Flip-only TTA for panoptic segmentation. Override per family.

        No family-generic implementation exists (unlike semantic's
        ``_postprocess_semantic_logits`` hook): panoptic decode is
        query-based (Mask2Former/MaskFormer-style), and the flip-merge
        strategy for query outputs is architecture-specific.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement panoptic flip-TTA."
        )

    def _merge_classify_tta(
        self,
        aug_dets: list,
        image_path,
        original_size: Tuple[int, int],
    ) -> Results:
        """Merge classification TTA by averaging probability vectors."""
        from ...utils.results import Probs, Results

        probs = [
            torch.as_tensor(det["probs"], dtype=torch.float32)
            for det, _, _, _ in aug_dets
            if "probs" in det
        ]
        avg_probs = (
            torch.stack(probs, dim=0).mean(dim=0)
            if probs
            else torch.zeros(0, dtype=torch.float32)
        )
        orig_w, orig_h = original_size
        return Results(
            boxes=None,
            orig_shape=(orig_h, orig_w),
            path=str(image_path) if image_path else None,
            names=self.names,
            probs=Probs(avg_probs),
        )

    def _merge_tta(
        self,
        aug_dets: list,
        iou_thres: float,
        image_path,
        original_size: Tuple[int, int],
        classes: Optional[List[int]] = None,
    ) -> Results:
        """Merge TTA detections from multiple augmented views via per-class NMS."""
        from ...utils.results import Boxes, Masks, Results

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_classes: List[torch.Tensor] = []
        all_masks: List[Optional[torch.Tensor]] = []
        has_masks = False

        for det, orig_size, is_flipped, scale in aug_dets:
            if det["num_detections"] == 0:
                continue

            w = orig_size[0]  # width of the (possibly scaled) augmented image
            boxes = torch.as_tensor(det["boxes"], dtype=torch.float32)
            scores = torch.as_tensor(det["scores"], dtype=torch.float32)
            cls = torch.as_tensor(det["classes"], dtype=torch.float32)

            if is_flipped:
                boxes = torch.stack(
                    [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]],
                    dim=1,
                )

            if scale != 1.0:
                boxes = boxes / scale
                orig_w_val, orig_h_val = original_size
                boxes[:, 0::2].clamp_(0, orig_w_val)
                boxes[:, 1::2].clamp_(0, orig_h_val)

            raw_m = det.get("masks")
            m = None
            # Masks in scaled views are in the wrong pixel space; skip them
            if raw_m is not None and scale == 1.0:
                has_masks = True
                m = raw_m if isinstance(raw_m, torch.Tensor) else torch.as_tensor(raw_m)
                if is_flipped:
                    m = m.flip(-1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(cls)
            all_masks.append(m)

        def _empty_results():
            return Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                    torch.zeros(0, dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if not all_boxes:
            return _empty_results()

        masks_cat: Optional[torch.Tensor] = None
        if has_masks:
            # Drop aug views that returned boxes but no masks to keep rows aligned
            paired = [
                (b, s, c, m)
                for b, s, c, m in zip(all_boxes, all_scores, all_classes, all_masks)
                if m is not None
            ]
            if paired:
                all_boxes, all_scores, all_classes, mask_list = map(list, zip(*paired))
                masks_cat = torch.cat(mask_list, dim=0)

        boxes_cat = torch.cat(all_boxes, dim=0)
        scores_cat = torch.cat(all_scores, dim=0)
        classes_cat = torch.cat(all_classes, dim=0)

        # Drop non-finite rows — batched_nms is undefined on NaN/Inf inputs.
        finite_mask = torch.isfinite(boxes_cat).all(dim=1) & torch.isfinite(scores_cat)
        if not finite_mask.all():
            boxes_cat = boxes_cat[finite_mask]
            scores_cat = scores_cat[finite_mask]
            classes_cat = classes_cat[finite_mask]
            if masks_cat is not None:
                masks_cat = masks_cat[finite_mask]
            if boxes_cat.numel() == 0:
                return _empty_results()

        # Shift to non-negative coords — batched_nms's class-offset trick
        # uses (boxes.max() + 1), which only separates classes when all
        # coords are non-negative. Translation-invariant for IoU.
        nms_boxes = boxes_cat - boxes_cat.min().clamp(max=0)
        # Per-class NMS in a single batched dispatch (class-offset trick).
        keep = batched_nms(nms_boxes, scores_cat, classes_cat.long(), iou_thres)
        if len(keep) == 0:
            return _empty_results()
        final_boxes = boxes_cat[keep]
        final_scores = scores_cat[keep]
        final_classes = classes_cat[keep]

        if classes is not None:
            cls_mask = torch.zeros(len(final_classes), dtype=torch.bool)
            for cid in classes:
                cls_mask |= final_classes == cid
            final_boxes = final_boxes[cls_mask]
            final_scores = final_scores[cls_mask]
            final_classes = final_classes[cls_mask]
            keep = keep[cls_mask]

        masks_obj = None
        if masks_cat is not None:
            masks_obj = Masks(masks_cat[keep], orig_shape)

        return Results(
            boxes=Boxes(final_boxes, final_scores, final_classes),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
            masks=masks_obj,
        )

    def track(
        self,
        source: Union[
            ImageInput,
            List[ImageInput],
            Tuple[ImageInput, ...],
            Iterator[ImageInput],
        ],
        *,
        track_conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        fps: float = 30.0,
        color_format: str = "auto",
        output_path: Optional[str] = None,
        tracker: str = "bytetrack",
        tracker_config=None,
        augment: bool = False,
        **tracker_kwargs,
    ) -> Generator[Results, None, None]:
        """Track objects across video frames or an image sequence.

        Runs detection on each frame and associates detections across time.
        Four trackers are available via ``tracker``: ByteTrack (default) and
        OC-SORT are motion-only; BoT-SORT adds an improved width/height motion
        model and camera-motion compensation; Deep OC-SORT adds appearance
        (ReID) embeddings so identities survive long occlusions and crossing
        targets, at the cost of a small embedding network run per frame (its
        weights are downloaded on first use). Yields one Results per frame
        with ``track_id`` set.

        Args:
            source: A video file path, a directory of images (sorted by
                name), a finite list/tuple of already-loaded images (paths,
                PIL Images, NumPy arrays, or tensors), or an iterator/
                generator yielding images one at a time for incremental,
                unbounded tracking. In every case, items are treated as
                consecutive frames of one logical video and tracked in
                order.
            track_conf: Confidence threshold for the tracker's first
                association stage — ``track_high_thresh`` for ByteTrack and
                BoT-SORT, ``det_thresh`` for OC-SORT and Deep OC-SORT. For the
                motion trackers the detector runs at a lower threshold
                internally so low-confidence detections remain available for
                recovery. For ByteTrack and BoT-SORT it must be >=
                ``track_low_thresh`` (default 0.1). Ignored when
                *tracker_config* is given, or when the matching key is passed
                explicitly in ``tracker_kwargs``.
            iou: IoU threshold for NMS during detection.
            imgsz: Override input image size.
            classes: Filter to specific class IDs.
            max_det: Maximum detections per frame.
            save: If True, save annotated video to *output_path*.
            show: Display tracked frames in a window.
            vid_stride: Process every N-th frame.
            fps: Frame rate to assume when *source* is an image sequence
                (directory, list, or iterator) rather than a video file,
                since those have no real capture rate of their own. Affects
                the fps written into a saved output video, and — for
                ByteTrack/BoT-SORT only — the lost-track buffer timing
                (OC-SORT/Deep OC-SORT count lost-track age in frames, not
                time, so *fps* doesn't affect them). With *vid_stride* > 1,
                the buffer is timed against ``fps / vid_stride`` — the rate
                at which frames are actually retained and reach the tracker.
                Ignored for video files, which use their own detected fps.
                A typed *tracker_config* keeps its explicit ``frame_rate`` for
                tracker timing, while *fps* and *vid_stride* still control
                image-sequence sampling and saved-video playback. A mismatch
                emits a warning rather than overwriting the explicit config.
            color_format: Color format for NumPy image-sequence items:
                ``"rgb"``, ``"bgr"``, or ``"auto"``. Other image input
                types are normalized to RGB by ``ImageLoader``. Ignored for
                video files.
            output_path: Path for saved video. Defaults to
                ``runs/track/<video_stem>.mp4`` for a video file, or
                ``runs/track/<name>.mp4`` for an image sequence.
            tracker: Which tracker to use: ``"bytetrack"``, ``"botsort"``,
                ``"ocsort"`` or ``"deepocsort"``. Ignored when
                *tracker_config* is given (the config type selects the tracker).
            tracker_config: A ``TrackConfig`` (ByteTrack), ``BoTSortConfig``
                (BoT-SORT), ``OCSortConfig`` (OC-SORT), or
                ``DeepOCSortConfig`` (Deep OC-SORT) instance, or None to build
                one from **tracker_kwargs.
            **tracker_kwargs: Forwarded to the selected tracker's
                ``from_kwargs`` (``TrackConfig``, ``BoTSortConfig``,
                ``OCSortConfig`` or ``DeepOCSortConfig``).

        Yields:
            Results with ``track_id`` attribute set as an (N,) int tensor.
        """
        task = getattr(self, "task", "detect")
        if task == "classify":
            raise NotImplementedError(
                "Tracking does not support classification models. Use predict()."
            )
        if task == "obb":
            raise NotImplementedError(
                "Tracking does not support oriented boxes yet. "
                "Use predict() for OBB models."
            )
        if task == "point":
            raise NotImplementedError(
                "Tracking does not support point results yet. "
                "Use predict() for point models."
            )
        if task == "depth":
            raise NotImplementedError(
                "Tracking does not support depth maps yet. "
                "Use predict() for depth models."
            )
        if task == "normal":
            raise NotImplementedError(
                "Tracking does not support surface-normal maps. "
                "Use predict() for normal models."
            )
        if task == "edge":
            raise NotImplementedError(
                "Tracking does not support edge maps. Use predict() for edge models."
            )
        if task == "semantic":
            raise NotImplementedError(
                "Tracking does not support semantic segmentation yet. "
                "Use predict() for semantic models."
            )
        if task == "panoptic":
            raise NotImplementedError(
                "Tracking does not support panoptic segmentation yet. "
                "Use predict() for panoptic models."
            )
        if task == "restore":
            raise NotImplementedError(
                "Tracking does not support restoration models. Use predict()."
            )
        if task == "mesh":
            raise NotImplementedError(
                "Tracking does not support body-mesh models yet. Use predict(). "
                "Associating meshes over time also needs a temporal contract "
                "(track IDs on the mesh rows, and a world frame) that the mesh "
                "task does not define yet."
            )
        if task == "ocr":
            raise NotImplementedError(
                "Tracking does not support OCR models yet. Use predict()."
            )

        from ...tracking import (
            BoTSortConfig,
            BoTSortTracker,
            ByteTracker,
            DeepOCSortConfig,
            DeepOCSortTracker,
            OCSortConfig,
            OCSortTracker,
            TrackConfig,
        )
        from ...utils.drawing import draw_boxes, draw_masks
        from ...utils.image_loader import ImageLoader
        from ...utils.source import ImageSequenceSource, SourceKind, classify_source
        from ...utils.video import run_video_inference

        # Classified up front (rather than after the tracker is built) so a
        # non-video source can seed the tracker's frame_rate default below.
        source_spec = classify_source(source)
        image_sequence_kinds = {
            SourceKind.IMAGE,
            SourceKind.IMAGE_BATCH,
            SourceKind.IMAGE_SEQUENCE,
            SourceKind.DIRECTORY,
        }
        if source_spec.kind in image_sequence_kinds:
            try:
                fps = float(fps)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"fps must be a finite value > 0, got {fps!r}"
                ) from exc
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"fps must be a finite value > 0, got {fps!r}")

        # A provided config picks the tracker; otherwise honour the selector.
        if isinstance(tracker_config, BoTSortConfig):
            # BoTSortConfig subclasses TrackConfig, so it must be checked first.
            tracker = "botsort"
        elif isinstance(tracker_config, DeepOCSortConfig):
            tracker = "deepocsort"
        elif isinstance(tracker_config, OCSortConfig):
            tracker = "ocsort"
        elif isinstance(tracker_config, TrackConfig):
            tracker = "bytetrack"
        tracker = (tracker or "bytetrack").lower()

        if (
            tracker in ("bytetrack", "botsort")
            and source_spec.kind in image_sequence_kinds
        ):
            # update() runs once per retained frame, so frame_rate must be
            # fps / vid_stride.
            retained_fps = fps / max(1, vid_stride)
            if tracker_config is None:
                # Pass the exact fraction: tracker.py already truncates via
                # int(), so pre-rounding here would double-round and can
                # shift max_time_lost by a frame at the expiry boundary.
                tracker_kwargs.setdefault("frame_rate", retained_fps)
            else:
                # frame_rate is a typed int field, so compare against the
                # nearest achievable value rather than the exact fraction.
                nearest_retained_fps = max(1, round(retained_fps))
                current = getattr(tracker_config, "frame_rate", nearest_retained_fps)
                if current != nearest_retained_fps:
                    # tracker_config always wins and is never overwritten --
                    # just warn so a silent mismatch doesn't linger.
                    warnings.warn(
                        f"tracker_config.frame_rate={tracker_config.frame_rate} does "
                        f"not match this image sequence's retained-frame rate "
                        f"(fps / vid_stride ≈ {nearest_retained_fps}). "
                        "tracker_config is used as-is for tracker timing, so lost "
                        "tracks will be kept recoverable for the wrong duration "
                        "unless "
                        f"you set frame_rate={nearest_retained_fps} on it yourself.",
                        stacklevel=2,
                    )

        if tracker == "deepocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = DeepOCSortConfig.from_kwargs(**tracker_kwargs)
            # Deep OC-SORT has no low-score recovery band; the detector only
            # needs to produce boxes down to det_thresh.
            effective_conf = tracker_config.det_thresh
            tracker_obj = DeepOCSortTracker(
                config=tracker_config, device=str(self.device)
            )
        elif tracker == "ocsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("det_thresh", track_conf)
                tracker_config = OCSortConfig.from_kwargs(**tracker_kwargs)
            # OC-SORT consumes low-score detections (>0.1) for recovery.
            effective_conf = min(0.1, tracker_config.det_thresh)
            tracker_obj = OCSortTracker(config=tracker_config)
        elif tracker == "botsort":
            if tracker_config is None:
                tracker_kwargs.setdefault("track_high_thresh", track_conf)
                tracker_config = BoTSortConfig.from_kwargs(**tracker_kwargs)
            # BoT-SORT keeps ByteTrack's low-confidence recovery stage.
            effective_conf = tracker_config.track_low_thresh
            tracker_obj = BoTSortTracker(config=tracker_config)
        elif tracker == "bytetrack":
            if tracker_config is None:
                tracker_kwargs.setdefault("track_high_thresh", track_conf)
                tracker_config = TrackConfig.from_kwargs(**tracker_kwargs)
            # ByteTrack needs to see low-confidence detections.
            effective_conf = tracker_config.track_low_thresh
            tracker_obj = ByteTracker(config=tracker_config)
        else:
            raise ValueError(
                f"Unknown tracker {tracker!r}; "
                "choose 'bytetrack', 'botsort', 'ocsort' or 'deepocsort'."
            )

        default_stem = "sequence"
        if source_spec.kind == SourceKind.VIDEO:
            frame_source: Union[Path, ImageSequenceSource] = Path(source_spec.source)
            if not frame_source.exists():
                raise FileNotFoundError(f"Video file not found: {frame_source}")
            default_stem = frame_source.stem
        elif source_spec.kind == SourceKind.DIRECTORY:
            images = ImageLoader.collect_images(source_spec.source)
            if not images:
                return
            default_stem = Path(source_spec.source).name
            frame_source = ImageSequenceSource(
                images,
                vid_stride=vid_stride,
                fps=fps,
                save_name=default_stem,
                color_format=color_format,
            )
        elif source_spec.kind == SourceKind.IMAGE_BATCH:
            frame_source = ImageSequenceSource(
                list(source_spec.items),
                vid_stride=vid_stride,
                fps=fps,
                save_name=default_stem,
                color_format=color_format,
            )
        elif source_spec.kind == SourceKind.IMAGE_SEQUENCE:
            frame_source = ImageSequenceSource(
                source_spec.source,
                vid_stride=vid_stride,
                fps=fps,
                save_name=default_stem,
                color_format=color_format,
            )
        elif source_spec.kind == SourceKind.IMAGE:
            # A single already-loaded image or path, treated as a one-frame
            # sequence so predict()-style single-image inputs also work.
            frame_source = ImageSequenceSource(
                [source_spec.source],
                vid_stride=vid_stride,
                fps=fps,
                save_name=default_stem,
                color_format=color_format,
            )
        else:
            raise NotImplementedError(
                f"track() does not yet support {source_spec.kind.value!r} sources. "
                "Pass a video file, a directory or list of images, or an "
                "image iterator."
            )

        model_names = self.names

        def predict_and_track(pil_img):
            result = self._runner(
                pil_img,
                conf=effective_conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format="rgb",
            )
            if isinstance(tracker_obj, (BoTSortTracker, DeepOCSortTracker)):
                # BoT-SORT needs pixels for camera motion; Deep OC-SORT needs
                # them for ReID crops.
                return tracker_obj.update(result, pil_img)
            return tracker_obj.update(result)

        def annotate_tracked(pil_img, result):
            if len(result) == 0:
                return pil_img
            img = pil_img
            if result.masks is not None:
                masks_np = result.masks.data
                if isinstance(masks_np, torch.Tensor):
                    masks_np = masks_np.cpu().numpy()
                img = draw_masks(img, masks_np, result.boxes.cls.tolist())
            tid_list = result.track_id.tolist() if result.track_id is not None else None
            return draw_boxes(
                img,
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
                class_names=model_names,
                track_ids=tid_list,
            )

        # Use runs/track/ prefix instead of runs/detect/
        track_output = output_path
        if save and output_path is None:
            from ...utils.general import increment_path

            track_output = str(
                increment_path(
                    Path("runs") / "track" / f"{default_stem}.mp4",
                    exist_ok=False,
                )
            )

        yield from run_video_inference(
            frame_source,
            predict_and_track,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=track_output,
            annotate_fn=annotate_tracked,
        )

    def quantize(
        self,
        recipe: str,
        calib: str | None = "coco128.yaml",
        samples: int = 128,
        batch: int = 8,
        algorithm: str = "auto",
        keep_high_precision: tuple | list | None = None,
        allow_download_scripts: bool = False,
        verbose: bool = True,
    ):
        """Quantize the loaded model in place (PyTorch execution) and return it.

        Calibration data is separate from training data: it is a small set of
        images used forward-only to derive activation ranges and scales.
        Labels are never read. To recover accuracy afterwards, run the normal
        training step on the returned model (QAT), optionally with the
        existing ``distill_model`` kwargs (QAD).

        Args:
            recipe: Quantization recipe. Casts: "fp16", "bf16". Conv+Linear:
                "int8", "fp8". Linear-only (transformer families such as
                rfdetr): "w4a16", "w4a8", "nvfp4", "mxfp4", and the
                research preview "int2" (QAT required).
            calib: Calibration images: data.yaml path or built-in dataset
                name. Pass None to skip calibration (int8 weights-only).
            samples: Maximum number of calibration images.
            batch: Calibration batch size.
            algorithm: Activation range estimation: "minmax" (absolute
                extremes across batches; the measured best default),
                "percentile" (mean of per-batch 0.1/99.9 percentiles; measured
                to degrade transformer families), "mse" (histogram sweep
                minimizing quantization reconstruction error), "entropy"
                (histogram sweep minimizing KL divergence between the float
                and quantized distributions), or "auto" (minmax). Weight
                scales stay per-channel absolute-max in every case.
            keep_high_precision: Substring patterns of module names kept in
                float. Defaults to the family policy (first layer + heads).
            allow_download_scripts: Allow embedded Python in dataset YAML
                download blocks.
            verbose: Log a quantization summary.

        Returns:
            This model, quantized in place.

        Example::

            >>> model = LibreYOLO("LibreYOLO9s.pt")
            >>> qmodel = model.quantize(recipe="int8", calib="coco8.yaml")
            >>> qmodel.val(data="coco8.yaml")
            >>> qmodel.train(data="coco8.yaml", epochs=5)  # QAT
            >>> qmodel.save("LibreYOLO9s-int8.pt")
        """
        from libreyolo.quant import quantize_model

        self._invalidate_cuda_graphs("quantize")
        return quantize_model(
            self,
            recipe=recipe,
            calib=calib,
            samples=samples,
            batch=batch,
            algorithm=algorithm,
            keep_high_precision=(
                tuple(keep_high_precision) if keep_high_precision is not None else None
            ),
            allow_download_scripts=allow_download_scripts,
            verbose=verbose,
        )

    def quant_info(self) -> Optional[Dict[str, Any]]:
        """Return the quantization state summary, or None for float models."""
        from libreyolo.quant import quant_info

        return quant_info(self)

    def dequantize(self):
        """Restore float modules in place, keeping the master weights.

        After QAT/QAD the masters are quantization-trained, so this is the
        bridge to the deployment exporters: ``model.dequantize()`` then
        ``model.export(format="onnx", int8=True, data=...)`` produces a real
        QDQ INT8 artifact from QAT-trained weights.
        """
        from libreyolo.quant import dequantize_model

        self._invalidate_cuda_graphs("dequantize")
        return dequantize_model(self)

    def save(self, path: str) -> str:
        """Save the current model as a LibreYOLO checkpoint.

        Writes schema v1.0 metadata; quantized models additionally carry the
        ``quant`` manifest so ``LibreYOLO(path)`` restores the quantized
        structure and scales.
        """
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        native_imgsz = self._get_input_size()
        rectangular_metadata = {}
        if isinstance(native_imgsz, (tuple, list)):
            if len(native_imgsz) != 2:
                raise ValueError(
                    "Model input size must be an int or (height, width), "
                    f"got {native_imgsz!r}."
                )
            imgsz_h, imgsz_w = int(native_imgsz[0]), int(native_imgsz[1])
            checkpoint_imgsz = max(imgsz_h, imgsz_w)
            rectangular_metadata = {"imgsz_h": imgsz_h, "imgsz_w": imgsz_w}
        else:
            checkpoint_imgsz = int(native_imgsz)
        extra_metadata = {}
        save_extra = getattr(self, "_save_extra_metadata", None)
        if callable(save_extra):
            extra_metadata = dict(save_extra() or {})
        checkpoint = wrap_libreyolo_checkpoint(
            state_dict,
            model_family=self._get_model_name(),
            size=self.size,
            task=self.task,
            nc=self.nb_classes,
            names=self.names,
            imgsz=checkpoint_imgsz,
            **rectangular_metadata,
            **extra_metadata,
        )
        quant_manifest = getattr(self, "_quant_manifest", None)
        if quant_manifest:
            checkpoint["quant"] = dict(quant_manifest)

        out = Path(path)
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, out)
        logger.info("Saved checkpoint to %s", out)
        return str(out)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        license: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload this model to a Hugging Face Hub repository.

        Saves the model as a schema v1.0 checkpoint (``model.pt``) and pushes
        it together with an auto-generated model card. The repo is created if
        it does not exist. Anyone can then load it back with
        ``LibreYOLO("owner/repo")``.

        Requires the optional huggingface_hub package
        (``pip install libreyolo[hf]``) and Hub authentication with write
        scope (``hf auth login`` or the ``HF_TOKEN`` environment variable, or
        pass ``token=``).

        Args:
            repo_id: Target repository as ``"owner/name"``.
            private: Create the repo as private (existing repos keep their
                visibility).
            token: Hub token overriding the ambient login.
            commit_message: Custom commit message for the checkpoint upload.
            license: License identifier for the model card front matter
                (e.g. ``"mit"``). Omitted from the card when None.
            metrics: Optional metric name -> value mapping rendered into the
                model card.

        Returns:
            The repository URL.
        """
        from libreyolo.utils.hf_hub import push_model_to_hub

        return push_model_to_hub(
            self,
            repo_id,
            private=private,
            token=token,
            commit_message=commit_message,
            license_id=license,
            metrics=metrics,
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export model to deployment format.

        Args:
            format: Target format ("onnx", "torchscript", "executorch",
                "tensorrt", "openvino", "mnn", "ncnn", "tflite"). "litert" is
                accepted as an alias for "tflite" (LiteRT is TensorFlow
                Lite's new name).
            **kwargs: Format-specific parameters forwarded to the exporter.

        Returns:
            Path to the exported model file.
        """
        # Live LoRA adapters are folded into dense weights for export. That
        # merge is destructive, so it happens only after every request
        # rejection: BaseExporter.__call__ merges after its preflight for the
        # float formats, and quantized_export merges after its recipe/format
        # gates for the format="pt" path (its ONNX path goes through
        # BaseExporter too).
        if getattr(self, "_quant_manifest", None):
            from libreyolo.quant.api import quantized_export

            return quantized_export(self, format=format, **kwargs)

        from libreyolo.export import BaseExporter

        return BaseExporter.create(format, self)(**kwargs)

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: int | tuple[int, int] | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 4,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *,
        plots: bool | None = None,
        **kwargs,
    ) -> Dict:
        """Run validation on a dataset.

        Args:
            data: Path to data.yaml file.
            batch: Batch size.
            imgsz: Square image size or ``(height, width)`` tuple. Defaults to
                the model's native input size.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            workers: Number of dataloader workers.
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            device: Device to use (default: same as model).
            split: Dataset split ("val", "test").
            save_json: Save predictions in COCO JSON format.
            plots: Alias for save_plots.
            verbose: Print detailed metrics.
            faster_coco_eval: (kwarg) Use the faster-coco-eval C++ backend
                for COCO metrics. Default True; falls back to pycocotools
                if the package is unavailable. Pass False (or set
                LIBREYOLO_FASTER_COCO_EVAL=0) to force pycocotools. The
                backend used is surfaced as ``model.last_eval_backend``.

        Returns:
            Dictionary with metrics/precision, metrics/recall,
            metrics/mAP50, metrics/mAP50-95.

            Detection validation additionally reports the F1-optimal
            confidence threshold as a free by-product of the COCO matching:
            ``metrics/best_conf`` (global, micro-averaged over classes) and
            ``metrics/best_conf_per_class`` (class name to threshold), with
            ``metrics/best_conf_f1`` the F1 reached at the global optimum.
            F1 is defined at IoU 0.50 matching; crowd/ignore regions count
            as neither TP nor FP. Use it as a data-driven ``conf=`` for
            deployment instead of a folklore default. Entries are NaN for
            classes where no threshold reaches F1 > 0 (no predictions, no
            ground truth, or all false positives).
        """
        from libreyolo.validation import (
            ClassifyValidator,
            DepthValidator,
            DetectionValidator,
            MatteValidator,
            NormalValidator,
            EdgeValidator,
            OBBValidator,
            OCRValidator,
            PanopticValidator,
            PointValidator,
            PoseValidator,
            RestoreValidator,
            SegmentationValidator,
            SemanticValidator,
            ValidationConfig,
        )

        if imgsz is None:
            imgsz = self._get_input_size()
        if plots is not None and "save_plots" not in kwargs:
            kwargs["save_plots"] = plots
        if augment and self.task == "obb":
            raise ValueError(
                "Augmented validation does not support oriented boxes yet. "
                "Use augment=False for OBB models."
            )
        if augment and self.task == "pose":
            raise ValueError(
                "Augmented validation does not support pose keypoints yet. "
                "Use augment=False for pose models."
            )
        if augment and self.task == "point":
            raise ValueError(
                "Augmented validation does not support point-task models yet. "
                "Use augment=False for point models."
            )
        if augment and self.task == "depth":
            raise ValueError(
                "Augmented validation does not support depth estimation yet. "
                "Use augment=False for depth models."
            )
        if augment and self.task == "normal":
            raise ValueError(
                "Augmented validation does not support surface normals yet. "
                "Use augment=False for normal models."
            )
        if augment and self.task == "edge":
            raise ValueError(
                "Augmented validation does not support edge detection yet. "
                "Use augment=False for edge models."
            )
        if augment and self.task == "restore":
            raise ValueError(
                "Augmented validation does not support restoration models yet. "
                "Use augment=False for restore models."
            )
        if augment and self.task == "mesh":
            raise ValueError(
                "Augmented validation does not support body-mesh models: a "
                "horizontal flip swaps left and right body parts, so merging "
                "flipped mesh parameters is not a matter of averaging. "
                "Use augment=False for mesh models."
            )
        if augment and self.task == "matte":
            raise ValueError(
                "Augmented validation does not support matte models yet. "
                "Use augment=False for matte models."
            )
        if augment and self.task == "ocr":
            raise ValueError(
                "Augmented validation does not support OCR models yet. "
                "Use augment=False for OCR models."
            )

        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            num_workers=workers,
            allow_download_scripts=allow_download_scripts,
            device=device or str(self.device),
            split=split,
            augment=augment,
            save_json=save_json,
            verbose=verbose,
            **kwargs,
        )

        if self.task == "mesh":
            raise NotImplementedError(
                "Body-mesh validation needs a ground-truth mesh dataset, and the "
                "usual benchmarks (3DPW, EMDB, AGORA) are research-license only, "
                "so none is bundled. The metrics themselves are available as "
                "libreyolo.validation.mesh_metrics (MPJPE, PA-MPJPE, PVE) for "
                "evaluating against a dataset you already hold."
            )
        if self.task == "gaze":
            raise NotImplementedError(
                "Validation against gaze ground-truth datasets (MPIIGaze, Gaze360) "
                "is out of scope for LibreYOLO. Evaluate upstream at "
                "https://github.com/Ahmednull/L2CS-Net."
            )
        if self.validator_class is not None:
            validator_cls = self.validator_class
        elif self.task == "pose":
            validator_cls = PoseValidator
        elif self.task == "point":
            validator_cls = PointValidator
        elif self.task == "segment":
            validator_cls = SegmentationValidator
        elif self.task == "semantic":
            validator_cls = SemanticValidator
        elif self.task == "panoptic":
            validator_cls = PanopticValidator
        elif self.task == "depth":
            validator_cls = DepthValidator
        elif self.task == "normal":
            validator_cls = NormalValidator
        elif self.task == "edge":
            validator_cls = EdgeValidator
        elif self.task == "restore":
            validator_cls = RestoreValidator
        elif self.task == "matte":
            validator_cls = MatteValidator
        elif self.task == "ocr":
            validator_cls = OCRValidator
        elif self.task == "classify":
            validator_cls = ClassifyValidator
        elif self.task == "obb":
            validator_cls = OBBValidator
        else:
            validator_cls = DetectionValidator
        validator = validator_cls(model=self, config=config)
        metrics = validator()
        # Provenance: which COCO eval backend produced these metrics
        # (e.g. "faster-coco-eval 1.7.2" / "pycocotools 2.0.10"; None for
        # validators that don't run COCO evaluation).
        self.last_eval_backend = getattr(validator, "eval_backend", None)
        return metrics
