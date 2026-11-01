"""Base class for image-to-3D families in the ``LibreGen`` tier (ADR 0031).

An image-to-3D model takes one picture of an object and returns the whole
object. This base owns the call shape (``model(source)``, sources as YOLO
predict takes them), the shared arguments (``seed``, ``n``, ``save``,
``stream``), the family-knob rule from ADR 0030, and the ``Results`` with a
``Gaussians`` payload. Families implement ``_load`` and ``_generate_3d``.
"""

from __future__ import annotations

import logging
import time
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

from ...utils.results import Gaussians, Results
from .base import _SEED_RANGE, LibreGenModel

logger = logging.getLogger(__name__)

_SAVE_SUFFIXES = (".ply", ".splat")


class LibreGen3DModel(LibreGenModel):
    """Image-to-3D model behind the LibreYOLO call surface."""

    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("generate3d",)
    DEFAULT_TASK: ClassVar[str] = "generate3d"
    # The input size is fixed, so ``imgsz`` is not a shared argument here.
    SHARED_ARGS: ClassVar[Tuple[str, ...]] = ("seed", "n")

    def __init__(
        self,
        size: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        compile: bool = False,
        offload: bool = False,
        task: str | None = None,
        checkpoint_dir: Optional[str] = None,
    ):
        if offload:
            raise ValueError(
                f"{type(self).__name__} fits on one GPU; offload=True is a "
                "text-to-image option."
            )
        if dtype != "auto":
            raise ValueError(
                f"{type(self).__name__} runs in its published mixed precision; "
                f"dtype must be 'auto', got {dtype!r}."
            )
        super().__init__(
            size, device=device, dtype=dtype, task=task, checkpoint_dir=checkpoint_dir
        )
        self.compile = bool(compile)
        if self.compile and self.device.type != "cuda":
            raise ValueError(
                f"compile=True needs CUDA; the model is on {self.device.type!r}."
            )

    # ------------------------------------------------------------------
    # Family hook
    # ------------------------------------------------------------------

    def _generate_3d(
        self, image, seeds: Sequence[int], knobs: Dict[str, Any]
    ) -> List[Gaussians]:
        """Return one ``Gaussians`` per seed for one source image (PIL, RGB or RGBA)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to(self, device) -> "LibreGen3DModel":
        device = self._resolve_device(device)
        if self.compile and device.type != "cuda":
            raise ValueError(
                f"compile=True needs CUDA; cannot move to {device.type!r}."
            )
        self.device = device
        if self._pipeline is not None:
            self._move_loaded(device)
        return self

    def _move_loaded(self, device) -> None:
        """Move loaded networks to ``device``; families with several modules override it."""
        self._pipeline.to(device)

    def info(self, verbose: bool = True) -> Dict[str, Any]:
        data = super().info(verbose=False)
        data.pop("imgsz", None)
        data["compile"] = self.compile
        if verbose:
            import json

            logger.info(json.dumps(data, indent=2))
        return data

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_sources(source: Any) -> Tuple[List[Any], bool]:
        """Return ``(items, single)``; a directory or a batch expands to its images."""
        import numpy as np
        import torch

        from ...utils.image_loader import ImageLoader

        if isinstance(source, (list, tuple)):
            items: List[Any] = []
            for item in source:
                expanded, _ = LibreGen3DModel._expand_sources(item)
                items.extend(expanded)
            if not items:
                raise ValueError("Empty source list.")
            return items, False
        if isinstance(source, (str, Path)) and not str(source).startswith(
            ("http://", "https://", "s3://", "gs://")
        ):
            path = Path(source)
            if path.is_dir():
                images = ImageLoader.collect_images(path)
                if not images:
                    raise FileNotFoundError(f"No images found in {path}.")
                return list(images), False
        # An (N, ...) batch is N images, as predict reads it; a batch of one
        # is one image.
        is_batch = (isinstance(source, np.ndarray) and source.ndim == 4) or (
            isinstance(source, torch.Tensor) and source.dim() == 4
        )
        if is_batch and source.shape[0] > 1:
            return [source[i] for i in range(source.shape[0])], False
        return [source], True

    @staticmethod
    def _load_source(source: Any):
        """Return ``(PIL image, path or None)``, keeping a file's or PIL image's alpha.

        A real alpha channel is the object's cutout, so files open as they are
        and PIL images pass through; every other input goes through the shared
        loader (NumPy arrays are BGR by default, like predict).
        """
        from PIL import Image

        from ...utils.image_loader import ImageLoader

        if isinstance(source, Image.Image):
            return source, None
        if isinstance(source, (str, Path)) and not str(source).startswith(
            ("http://", "https://", "s3://", "gs://")
        ):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = Image.open(path)
            image.load()
            return image, str(path)
        return ImageLoader.load(source), (
            str(source) if isinstance(source, str) else None
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        source: Any = None,
        *,
        seed: Optional[int] = None,
        n: int = 1,
        stream: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        prompt: Any = None,
        imgsz: Any = None,
        **knobs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Generate the 3D object in each source image.

        Args:
            source: An image path, URL, ``PIL.Image``, NumPy array (BGR),
                tensor, a directory of images, or a list of any of these. A
                file or ``PIL.Image`` with a real alpha channel is used as the
                object's cutout.
            seed: The same seed, model and settings reproduce the same object.
                Object ``i`` of a source uses ``seed + i``. ``None`` draws a
                fresh seed, recorded on the result.
            n: Objects per source.
            stream: Return a generator instead of a list.
            save: Write each object as ``.ply`` under ``runs/generate/predict``.
            output_path: Explicit ``.ply`` or ``.splat`` file for one object.
            **knobs: Family settings such as ``steps``, ``guidance`` or
                ``gaussians``. A name the family does not have raises.

        Returns:
            ``Results`` for one source with ``n=1``, a list otherwise, a
            generator with ``stream=True``. Each carries ``Results.gaussians``.
        """
        if prompt is not None:
            raise TypeError(
                f"{type(self).__name__} generates from an image and has no text input; "
                'pass the image, e.g. model("chair.jpg").'
            )
        if imgsz is not None:
            raise TypeError(
                f"{type(self).__name__} has a fixed input size; the output detail is a "
                f"knob ({', '.join(self.KNOBS)})."
            )
        if source is None:
            raise ValueError(
                'generate() needs a source image, e.g. model("chair.jpg").'
            )
        sources, single = self._expand_sources(source)
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
        settings = self._resolve_knobs(knobs)
        one_object = single and n == 1
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        if output_path is not None and not one_object:
            raise ValueError("output_path names one file; it needs one source and n=1.")
        if (
            output_path is not None
            and Path(output_path).suffix.lower() not in _SAVE_SUFFIXES
        ):
            raise ValueError(
                f"output_path needs a .ply or .splat extension, got {output_path!r}."
            )

        def run() -> Generator[Results, None, None]:
            self._ensure_loaded()
            index = 0
            save_dir: Optional[Path] = None
            for item in sources:
                image, path = self._load_source(item)
                seeds = self._seeds(seed, n)
                started = time.perf_counter()
                objects = list(self._generate_3d(image, seeds, settings))
                elapsed_ms = (
                    (time.perf_counter() - started) * 1000.0 / max(len(objects), 1)
                )
                if len(objects) != len(seeds):
                    raise RuntimeError(
                        f"{type(self).__name__} returned {len(objects)} objects for {len(seeds)} seeds."
                    )
                for gaussians in objects:
                    result = Results(
                        None,
                        (int(image.height), int(image.width)),
                        path=path,
                        names={},
                        speed={"inference": elapsed_ms},
                        orig_img=image.convert("RGB"),
                        gaussians=gaussians,
                    )
                    if save:
                        save_dir = self._save_object(
                            result, index, output_path, save_dir
                        )
                    index += 1
                    yield result

        if stream:
            return run()
        results = list(run())
        return results[0] if one_object else results

    predict = generate
    __call__ = generate

    def _save_object(
        self,
        result: Results,
        index: int,
        output_path: Optional[str],
        save_dir: Optional[Path],
    ) -> Optional[Path]:
        """Write one object; return the call's run directory once it exists."""
        from ...utils.general import increment_path, log_saved_result

        if output_path is not None:
            destination = Path(output_path)
        else:
            if save_dir is None:
                save_dir = increment_path(Path("runs/generate/predict"), mkdir=True)
                self._save_dir = save_dir
            stem = Path(result.path).stem if result.path else "image"
            destination = save_dir / f"{index:03d}_{stem}_{result.gaussians.seed}.ply"
        result.gaussians.save(destination)
        log_saved_result(result, destination)
        return save_dir

    # ------------------------------------------------------------------
    # Reserved surface
    # ------------------------------------------------------------------

    def train(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Reserved: fine-tune on image and object pairs (ADR 0031)."""
        raise NotImplementedError(
            f"Training is not supported for {type(self).__name__} yet. train(data=...) is "
            "reserved for fine-tuning on image and 3D object pairs (ADR 0031)."
        )

    def val(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Reserved: rendered-view metrics on a held-out set (ADR 0031)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no validation metric yet (ADR 0031)."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not export to {format!r}: a sampling loop over "
            "several networks has no exportable graph contract yet (ADR 0031). Run it "
            "through generate()."
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} generates 3D objects; tracking does not apply."
        )
