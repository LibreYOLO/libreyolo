"""Base class for the ``LibreGen`` tier: generative image models.

A generative model takes a text prompt and returns a synthesized image. This
base owns everything LibreYOLO promises about that (ADR 0030): the call
shape, the arguments every generator understands (``imgsz``, ``seed``, ``n``,
``save``, ``stream``), the rule for family-specific knobs, the ``Results``
with a ``GeneratedImage`` payload, the pinned-snapshot weights contract, and
the reserved ``train`` / ``val`` surface. Families implement two hooks,
``_load`` and ``_generate``. The tier is a sibling of ``LibreVLM`` and
``LibreVLA``: it is not a ``BaseModel`` and never enters the state-dict
factory.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import re
import threading
import time
import weakref
from numbers import Integral
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ...utils.results import GeneratedImage, Results

logger = logging.getLogger(__name__)

# One load lock per model, so threads that first call the same model at the
# same time load it once while other models load in parallel. Kept off the
# instance so models still copy and pickle.
_LOAD_LOCKS: "weakref.WeakKeyDictionary[Any, threading.RLock]" = weakref.WeakKeyDictionary()
_LOAD_LOCKS_GUARD = threading.Lock()


def _load_lock(model: Any) -> threading.RLock:
    with _LOAD_LOCKS_GUARD:
        lock = _LOAD_LOCKS.get(model)
        if lock is None:
            lock = _LOAD_LOCKS[model] = threading.RLock()
        return lock

_INSTALL_HINT = (
    "LibreGen models require the 'gen' extra. Install with:\n"
    "    pip install 'libreyolo[gen]'"
)
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SEED_RANGE = 2**32
_DTYPES = ("auto", "bf16", "fp16", "fp32")

PromptInput = Union[str, Sequence[str]]


class LibreGenModel:
    """Generative image model behind the LibreYOLO call surface."""

    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    HF_REPOS: ClassVar[Dict[str, str]] = {}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {}
    # Default square output size per size code.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {}
    TASK_INPUT_SIZES: ClassVar[Dict[str, Dict[str, int]]] = {}
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("generate",)
    DEFAULT_TASK: ClassVar[str] = "generate"
    # Output height and width are rounded up to a multiple of this.
    SIZE_MULTIPLE: ClassVar[int] = 16
    # Family knobs ``generate()`` accepts beyond the shared arguments, and
    # the family's defaults for them. Shared concepts keep shared names:
    # ``steps`` for iterative sampling, ``guidance`` for prompt guidance,
    # ``negative`` for a negative prompt.
    KNOBS: ClassVar[Tuple[str, ...]] = ()
    DEFAULTS: ClassVar[Dict[str, Any]] = {}
    # Shared arguments the unknown-knob error lists before the family knobs.
    SHARED_ARGS: ClassVar[Tuple[str, ...]] = ("imgsz", "seed", "n")
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = ()
    TRAINABLE: ClassVar[bool] = False
    # `libreyolo models` points at the dedicated verb instead of `model=` names.
    CLI_COMMAND: ClassVar[str] = "generate"
    _LICENSE_NOTICE: ClassVar[str] = ""
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False

    def __init__(
        self,
        size: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        offload: bool = False,
        task: str | None = None,
        checkpoint_dir: Optional[str] = None,
    ):
        if size is None:
            size = next(iter(self.INPUT_SIZES))
        if size not in self.INPUT_SIZES:
            raise ValueError(
                f"Invalid size {size!r} for {type(self).__name__}. "
                f"Must be one of: {', '.join(self.INPUT_SIZES)}"
            )
        from ...tasks import normalize_task

        resolved_task = normalize_task(task, default=self.DEFAULT_TASK)
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"{type(self).__name__} solves the "
                f"{' and '.join(repr(t) for t in self.SUPPORTED_TASKS)} task only. "
                f"Got task={task!r}."
            )
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {dtype!r}.")
        self.task = resolved_task
        self.family = self.FAMILY
        self.size = size
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.offload = bool(offload)
        if self.offload and self.device.type != "cuda":
            raise ValueError(
                "offload=True moves idle components off a CUDA device; it has "
                f"no effect on {self.device.type!r}."
            )
        self.input_size = self.INPUT_SIZES[size]
        self.names: Dict[int, str] = {}
        self._checkpoint_dir: Optional[Path] = (
            Path(checkpoint_dir) if checkpoint_dir else None
        )
        if self._checkpoint_dir is not None and not self._checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"{self._checkpoint_dir} is not a model directory."
            )
        self._pipeline = None
        self._save_dir: Optional[Path] = None
        self.model_path: Optional[str] = (
            str(self._checkpoint_dir)
            if self._checkpoint_dir
            else self.HF_REPOS.get(size)
        )

    # ------------------------------------------------------------------
    # Family hooks
    # ------------------------------------------------------------------

    def _load(self, snapshot_dir: str) -> None:
        """Load the upstream model from ``snapshot_dir`` into ``self._pipeline``."""
        raise NotImplementedError

    def _generate(
        self,
        prompt: str,
        height: int,
        width: int,
        seeds: Sequence[int],
        knobs: Dict[str, Any],
    ) -> List[Any]:
        """Return one RGB image (PIL or ``(H, W, 3)`` uint8) per seed."""
        raise NotImplementedError

    @classmethod
    def matches_directory(cls, path: Any) -> bool:
        """True when ``path`` is a local model directory this family loads."""
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Canonical name, ``<family>-<size>`` with dashes."""
        return f"{self.FAMILY.replace('_', '-')}-{self.size}"

    @property
    def pipeline(self):
        """The loaded upstream pipeline (loads on first access)."""
        self._ensure_loaded()
        return self._pipeline

    def _check_runtime(self) -> None:
        """Raise when the upstream runtime is missing; runs before any download."""

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            with _load_lock(self):
                if self._pipeline is None:
                    self._check_runtime()
                    self._load(self._ensure_weights())

    @staticmethod
    def _resolve_device(device: Any):
        import torch

        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            device = f"cuda:{device}"
        resolved = torch.device(device)
        # Fail before any weights are downloaded or loaded.
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"device={str(resolved)!r} needs CUDA, which is not available.")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("device='mps' needs MPS, which is not available.")
        return resolved

    def _resolve_dtype(self):
        """bf16 on MPS (macOS 14+) and on CUDA when supported (else fp16); fp32 otherwise."""
        import torch

        explicit = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if self.dtype != "auto":
            return explicit[self.dtype]
        if self.device.type == "mps":
            # fp32 doubles the memory of weights that are published in bf16,
            # but MPS only has bfloat16 from macOS 14.
            release = platform.mac_ver()[0].split(".")[0]
            if release.isdigit() and int(release) < 14:
                return torch.float32
            return torch.bfloat16
        if self.device.type != "cuda":
            return torch.float32
        # Ask the selected GPU, not whichever one is current.
        with torch.cuda.device(self.device):
            supported = torch.cuda.is_bf16_supported()
        return torch.bfloat16 if supported else torch.float16

    def to(self, device) -> "LibreGenModel":
        if self.offload:
            raise ValueError("to() is not available with offload=True.")
        self.device = self._resolve_device(device)
        if self._pipeline is not None:
            # dtype="auto" depends on the device, so it moves with it.
            self._pipeline.to(self.device, self._resolve_dtype())
        return self

    def info(self, verbose: bool = True) -> Dict[str, Any]:
        data = {
            "family": self.FAMILY,
            "size": self.size,
            "task": self.task,
            "device": str(self.device),
            "dtype": self.dtype,
            "imgsz": self.input_size,
            "knobs": {knob: self.DEFAULTS.get(knob) for knob in self.KNOBS},
            "checkpoint": str(self._checkpoint_dir) if self._checkpoint_dir else None,
        }
        if verbose:
            logger.info(json.dumps(data, indent=2))
        return data

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Hub URL of the pinned snapshot behind ``Libre<Prefix><size>``."""
        stem = str(filename)
        for ext in (".pt", ".safetensors"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        if not cls.FILENAME_PREFIX or not stem.startswith(cls.FILENAME_PREFIX):
            return None
        size = stem[len(cls.FILENAME_PREFIX) :]
        repo = cls.HF_REPOS.get(size)
        if repo is None:
            return None
        revision = cls.HF_REVISIONS.get(size) or "main"
        return f"https://huggingface.co/{repo}/tree/{revision}"

    @classmethod
    def _notify_license_once(cls) -> None:
        if cls._LICENSE_NOTICE and not cls._LICENSE_NOTICE_SHOWN:
            cls._LICENSE_NOTICE_SHOWN = True
            logger.warning(cls._LICENSE_NOTICE)

    def _snapshot_complete(
        self, local_dir: Path, *, repo: str, revision: Optional[str]
    ) -> bool:
        marker = local_dir / _SNAPSHOT_COMPLETE_MARKER
        if not marker.is_file():
            return False
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if recorded.get("repo") != repo or recorded.get("revision") != revision:
            return False
        return self.matches_directory(local_dir)

    def _ensure_weights(self) -> str:
        """Return a local model dir, downloading the pinned snapshot if needed."""
        if self._checkpoint_dir is not None:
            return str(self._checkpoint_dir)
        repo = self.HF_REPOS[self.size]
        revision = self.HF_REVISIONS.get(self.size)
        if revision is not None and not _COMMIT_SHA_RE.fullmatch(revision):
            raise ValueError(
                f"{type(self).__name__}.HF_REVISIONS[{self.size!r}] must be a "
                "40-char commit SHA."
            )
        local_dir = Path("weights") / f"{self.FILENAME_PREFIX}{self.size}"
        self._notify_license_once()
        if self._snapshot_complete(local_dir, repo=repo, revision=revision):
            return str(local_dir)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        source = f"{repo}@{revision}" if revision else repo
        logger.info(
            "Downloading %s weights from %s -> %s ...", self.FAMILY, source, local_dir
        )
        download_kwargs: Dict[str, Any] = {}
        if revision is not None:
            download_kwargs["revision"] = revision
        if self.SNAPSHOT_ALLOW_PATTERNS:
            download_kwargs["allow_patterns"] = list(self.SNAPSHOT_ALLOW_PATTERNS)
        snapshot_download(repo, local_dir=str(local_dir), **download_kwargs)
        marker = local_dir / _SNAPSHOT_COMPLETE_MARKER
        marker.write_text(
            json.dumps({"repo": repo, "revision": revision}) + "\n", encoding="utf-8"
        )
        if not self._snapshot_complete(local_dir, repo=repo, revision=revision):
            marker.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} in {local_dir} is not a "
                f"loadable {type(self).__name__} directory."
            )
        return str(local_dir)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_prompts(prompt: Any) -> Tuple[List[str], bool]:
        """Return ``(prompts, single)`` from a string or a sequence of strings."""
        if isinstance(prompt, str):
            prompts, single = [prompt], True
        elif isinstance(prompt, (list, tuple)):
            prompts, single = list(prompt), False
        else:
            raise TypeError(
                "prompt must be a string or a list of strings, got "
                f"{type(prompt).__name__}."
            )
        if not prompts:
            raise ValueError("Empty prompt list.")
        for text in prompts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every prompt must be a non-empty string.")
        return prompts, single

    def _resolve_imgsz(self, imgsz: Any) -> Tuple[int, int]:
        """Return ``(height, width)``, rounded up to ``SIZE_MULTIPLE``."""
        if imgsz is None:
            imgsz = self.input_size
        if isinstance(imgsz, bool):
            raise TypeError("imgsz must be an int or a (height, width) pair.")
        if isinstance(imgsz, Integral):
            requested = (imgsz, imgsz)
        elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
            requested = (imgsz[0], imgsz[1])
        else:
            raise TypeError("imgsz must be an int or a (height, width) pair.")
        for side in requested:
            if isinstance(side, bool) or not isinstance(side, Integral) or side <= 0:
                raise ValueError(f"imgsz sides must be positive integers, got {imgsz!r}.")
        requested = (int(requested[0]), int(requested[1]))
        multiple = self.SIZE_MULTIPLE
        height, width = (-(-side // multiple) * multiple for side in requested)
        if (height, width) != requested:
            logger.warning(
                "imgsz=%s must be a multiple of %d for %s; using %s.",
                list(requested),
                multiple,
                type(self).__name__,
                [height, width],
            )
        return height, width

    def _resolve_knobs(self, knobs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge family defaults with the call's knobs; reject unknown names."""
        unknown = sorted(set(knobs) - set(self.KNOBS))
        if unknown:
            supported = [*self.SHARED_ARGS, *self.KNOBS]
            raise TypeError(
                f"{type(self).__name__} does not accept {', '.join(map(repr, unknown))}. "
                f"Supported: {', '.join(supported)}."
            )
        settings = dict(self.DEFAULTS)
        settings.update({k: v for k, v in knobs.items() if v is not None})
        return settings

    @staticmethod
    def _seeds(seed: Optional[int], n: int) -> List[int]:
        """Image ``i`` of a prompt uses ``seed + i``; no seed draws a fresh one."""
        if seed is None:
            seed = random.SystemRandom().randrange(_SEED_RANGE)
        return [(seed + offset) % _SEED_RANGE for offset in range(n)]

    def _result(
        self,
        image: Any,
        prompt: str,
        seed: int,
        settings: Dict[str, Any],
        elapsed_ms: float,
    ) -> Results:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        data = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        shape = (int(data.shape[0]), int(data.shape[1]))
        generated = GeneratedImage(
            data,
            shape,
            prompt=prompt,
            seed=seed,
            params={
                "model": self.model_path if self._checkpoint_dir else self.model_name,
                "imgsz": list(shape),
                **settings,
            },
        )
        return Results(
            None,
            shape,
            names={},
            speed={"inference": elapsed_ms},
            generated=generated,
        )

    def generate(
        self,
        prompt: Optional[PromptInput] = None,
        *,
        source: Any = None,
        imgsz: Union[int, Sequence[int], None] = None,
        seed: Optional[int] = None,
        n: int = 1,
        stream: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        **knobs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Generate images from text.

        Args:
            prompt: The text to render, or a list of texts. A string is always
                a prompt, never a path.
            source: Reserved for image-conditioned generation (editing,
                image-to-image). Not supported yet.
            imgsz: Output size: an int for a square image or ``(height,
                width)``. Defaults to the model's native size.
            seed: The same seed, model and settings reproduce the same image.
                Image ``i`` of a prompt uses ``seed + i``. ``None`` draws a
                fresh seed, recorded on the result.
            n: Images per prompt.
            stream: Return a generator instead of a list.
            save: Write each image under ``runs/generate/predict``.
            output_path: Explicit output file for a single image.
            **knobs: Family-specific settings such as ``steps``, ``guidance``
                or ``negative``. A name the family does not have raises.

        Returns:
            ``Results`` for one prompt with ``n=1``, a list otherwise, a
            generator with ``stream=True``. Each carries ``Results.generated``.
        """
        if source is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support image-conditioned "
                "generation (source=) yet; generate from a text prompt."
            )
        if prompt is None:
            raise ValueError('generate() needs a prompt, e.g. model("a red fox").')
        prompts, single = self._coerce_prompts(prompt)
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        n = int(n)
        if seed is not None:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, Integral)
                or not 0 <= seed < _SEED_RANGE
            ):
                raise ValueError(
                    f"seed must be an integer in [0, 2**32), got {seed!r}."
                )
            seed = int(seed)
        height, width = self._resolve_imgsz(imgsz)
        settings = self._resolve_knobs(knobs)
        one_image = single and n == 1
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        if output_path is not None and not one_image:
            raise ValueError("output_path names one file; it needs one prompt and n=1.")
        if output_path is not None:
            from PIL import Image

            suffix = Path(output_path).suffix.lower()
            if suffix not in Image.registered_extensions():
                raise ValueError(
                    f"output_path needs an image extension such as .png, got {output_path!r}."
                )

        def run() -> Generator[Results, None, None]:
            self._ensure_loaded()
            index = 0
            # One run directory per generate() call, like the shared predict
            # runner; it lives here so concurrent streams never share one.
            save_dir: Optional[Path] = None
            for text in prompts:
                seeds = self._seeds(seed, n)
                started = time.perf_counter()
                images = list(self._generate(text, height, width, seeds, settings))
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(len(images), 1)
                if len(images) != len(seeds):
                    raise RuntimeError(
                        f"{type(self).__name__} returned {len(images)} images "
                        f"for {len(seeds)} seeds."
                    )
                for image, image_seed in zip(images, seeds):
                    result = self._result(image, text, image_seed, settings, elapsed_ms)
                    if save:
                        save_dir = self._save_image(
                            result, index, output_path, save_dir
                        )
                    index += 1
                    yield result

        if stream:
            return run()
        results = list(run())
        return results[0] if one_image else results

    predict = generate
    __call__ = generate

    def _save_image(
        self,
        result: Results,
        index: int,
        output_path: Optional[str],
        save_dir: Optional[Path],
    ) -> Optional[Path]:
        """Write one image; return the call's run directory once it exists."""
        from ...utils.general import increment_path, log_saved_result

        if output_path is not None:
            destination = Path(output_path)
        else:
            if save_dir is None:
                save_dir = increment_path(Path("runs/generate/predict"), mkdir=True)
                self._save_dir = save_dir
            generated = result.generated
            slug = re.sub(r"[^a-z0-9]+", "-", generated.prompt.lower()).strip("-")[:40]
            name = f"{index:03d}_{slug.rstrip('-') or 'image'}_{generated.seed}.png"
            destination = save_dir / name
        result.generated.save(destination)
        log_saved_result(result, destination)
        return save_dir

    # ------------------------------------------------------------------
    # Train / val / unsupported
    # ------------------------------------------------------------------

    def train(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Reserved: fine-tune on images with captions (ADR 0030)."""
        raise NotImplementedError(
            f"Training is not supported for {type(self).__name__} yet. "
            "train(data=...) is reserved for fine-tuning on captioned images; "
            "a fine-tune saved in the family's upstream format already loads "
            "with LibreGen(path)."
        )

    def val(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Reserved: generation metrics on a captioned image set (ADR 0030)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no validation metric yet (ADR 0030)."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not export to {format!r}: a sampling "
            "loop over several networks has no exportable graph contract yet "
            "(ADR 0030). Run it through generate()."
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} generates images; tracking does not apply."
        )

    def benchmark(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not covered by the export benchmark."
        )
