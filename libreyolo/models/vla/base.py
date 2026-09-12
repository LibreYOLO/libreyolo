"""Base class for the ``LibreVLA`` tier: vision-language-action policies.

A VLA policy takes camera frames, the robot's proprioceptive state and an
instruction, and returns an action chunk: the next ``T`` actions of ``D``
dimensions each. This base owns everything LibreYOLO promises about that
(ADR 0028): the observation contract (``observation.py``), the ``Results``
with an ``Actions`` payload, source handling for frames, folders, videos and
live streams, the checkpoint directory contract, and the ``train`` / ``val``
surface. Families implement four small hooks over their upstream policy
API. The tier is a sibling of ``LibreVLM``: it is not a ``BaseModel`` and
never enters the state-dict factory.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from PIL import Image

from ...utils.image_loader import SUPPORTED_EXTENSIONS, ImageInput
from ...utils.results import Actions, Results
from ...utils.source import SourceKind, build_stream_source, classify_source
from .checkpoint import read_contract
from .observation import Observation, coerce_state, load_frames, map_cameras

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "LibreVLA models require the 'vla' extra and Python 3.12 or newer "
    "(the lerobot package's floor). Install with:\n"
    "    pip install 'libreyolo[vla]'"
)
_SNAPSHOT_COMPLETE_MARKER = ".libreyolo_snapshot_complete"
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_UNSUPPORTED_TASK = "LibreVLA models solve the 'act' task only."


class LibreVLAModel:
    """Vision-language-action policy behind the LibreYOLO predict surface."""

    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    HF_REPOS: ClassVar[Dict[str, str]] = {}
    HF_REVISIONS: ClassVar[Dict[str, str]] = {}
    # Nominal per-size resize the family applies; the processor owns resize.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {}
    TASK_INPUT_SIZES: ClassVar[Dict[str, Dict[str, int]]] = {}
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("act",)
    DEFAULT_TASK: ClassVar[str] = "act"
    SNAPSHOT_ALLOW_PATTERNS: ClassVar[Tuple[str, ...]] = (
        "*.json",
        "*.safetensors",
        "README.md",
    )
    TRAINABLE: ClassVar[bool] = True
    REQUIRES_INSTRUCTION: ClassVar[bool] = True
    PRETRAINED_BASE: ClassVar[bool] = True
    TRAIN_UNSUPPORTED_REASON: ClassVar[str] = ""
    _LICENSE_NOTICE: ClassVar[str] = ""
    _LICENSE_NOTICE_SHOWN: ClassVar[bool] = False

    def __init__(
        self,
        size: str,
        *,
        device: str = "auto",
        task: str | None = None,
        instruction: Optional[str] = None,
        cameras: Optional[Sequence[str]] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs,
    ):
        sizes = self.HF_REPOS if self.PRETRAINED_BASE else self.INPUT_SIZES
        if size not in sizes:
            raise ValueError(
                f"Invalid size {size!r} for {type(self).__name__}. "
                f"Must be one of: {', '.join(sizes)}"
            )
        from ...tasks import normalize_task

        resolved_task = normalize_task(task, default=self.DEFAULT_TASK)
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(f"{_UNSUPPORTED_TASK} Got task={task!r}.")
        self.task = resolved_task
        self.family = self.FAMILY
        self.size = size
        self.device = self._resolve_device(device)
        self.input_size = self.INPUT_SIZES.get(size)
        self.names: Dict[int, str] = {}
        self.instruction: Optional[str] = None
        if instruction is not None:
            self.set_instruction(instruction)
        self.cameras: Optional[List[str]] = (
            [str(c) for c in cameras] if cameras is not None else None
        )
        self._checkpoint_dir: Optional[Path] = (
            Path(checkpoint_dir) if checkpoint_dir else None
        )
        self.contract: Dict[str, Any] = {}
        if self._checkpoint_dir is not None:
            self.contract = read_contract(self._checkpoint_dir)
            if self.contract.get("family") != self.FAMILY:
                raise ValueError(
                    f"{self._checkpoint_dir} was trained on family "
                    f"{self.contract.get('family')!r}, not {self.FAMILY!r}."
                )
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._state_warned = False
        self.model_path: Optional[str] = (
            str(self._checkpoint_dir)
            if self._checkpoint_dir
            else self.HF_REPOS.get(size)
        )

    # ------------------------------------------------------------------
    # Family hooks
    # ------------------------------------------------------------------

    def _load_policy(self, snapshot_dir: str) -> None:
        """Load the upstream policy and processors from ``snapshot_dir``.

        Must set ``self._policy`` (an ``nn.Module`` in eval mode on
        ``self.device``) and may set ``self._preprocessor`` /
        ``self._postprocessor``.
        """
        raise NotImplementedError

    def _predict_chunk(self, observation: Observation) -> Any:
        """Return the ``(T, D)`` action chunk for one observation."""
        raise NotImplementedError

    def _pretrained_config(self, snapshot_dir: str) -> Any:
        """Return the upstream policy config stored in ``snapshot_dir``."""
        raise NotImplementedError

    def _scratch_config(self, meta: Any) -> Any:
        """Return the family's default config for training without a base policy."""
        raise NotImplementedError

    @property
    def camera_slots(self) -> List[str]:
        """Ordered camera slot names the loaded policy expects."""
        raise NotImplementedError

    @property
    def state_dim(self) -> int:
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    def chunk_size(self) -> int:
        raise NotImplementedError

    @property
    def action_names(self) -> Optional[List[str]]:
        return self.contract.get("action_names") if self.contract else None

    @property
    def fps(self) -> Optional[float]:
        return self.contract.get("fps") if self.contract else None

    # ------------------------------------------------------------------
    # Instruction and lifecycle
    # ------------------------------------------------------------------

    def set_instruction(self, instruction: str) -> "LibreVLAModel":
        """Set the sticky task instruction used by later ``predict`` calls."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")
        self.instruction = instruction.strip()
        return self

    def set_classes(self, *args, **kwargs):
        raise AttributeError(
            f"{type(self).__name__} has no class vocabulary. Use "
            "set_instruction(text) to tell the policy what to do."
        )

    def reset(self) -> None:
        """Clear any family-side action queue between episodes."""
        policy = self._policy
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()

    @property
    def model(self):
        """The loaded upstream policy module (loads on first access)."""
        self._ensure_loaded()
        return self._policy

    def _ensure_loaded(self) -> None:
        if self._policy is None:
            self._load_policy(self._ensure_weights())

    @staticmethod
    def _resolve_device(device: Any):
        import torch

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            return torch.device(f"cuda:{device}")
        return torch.device(device)

    def to(self, device) -> "LibreVLAModel":
        self.device = self._resolve_device(device)
        if self._policy is not None:
            self._policy.to(self.device)
        return self

    def info(self, verbose: bool = True) -> Dict[str, Any]:
        data = {
            "family": self.FAMILY,
            "size": self.size,
            "task": self.task,
            "device": str(self.device),
            "instruction": self.instruction,
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
        """Hub URL of the pinned snapshot behind ``Libre<Prefix><size>``.

        VLA weights are snapshot directories, not ``.pt`` files, so this
        returns the upstream repository tree at the pinned revision.
        """
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

    @staticmethod
    def _snapshot_complete(
        local_dir: Path, *, repo: str, revision: Optional[str]
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
        has_config = (local_dir / "config.json").is_file()
        has_weights = any(local_dir.glob("*.safetensors"))
        return has_config and has_weights

    def _ensure_weights(self) -> str:
        """Return a local policy dir, downloading the pinned snapshot if needed."""
        if self._checkpoint_dir is not None:
            return str(self._checkpoint_dir)
        if not self.PRETRAINED_BASE:
            raise ValueError(
                f"{type(self).__name__} is untrained. Call train(data=...) first "
                "and load its best or last checkpoint with LibreVLA(path)."
            )
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
        snapshot_download(
            repo,
            local_dir=str(local_dir),
            allow_patterns=list(self.SNAPSHOT_ALLOW_PATTERNS),
            **download_kwargs,
        )
        (local_dir / _SNAPSHOT_COMPLETE_MARKER).write_text(
            json.dumps({"repo": repo, "revision": revision}) + "\n", encoding="utf-8"
        )
        if not self._snapshot_complete(local_dir, repo=repo, revision=revision):
            (local_dir / _SNAPSHOT_COMPLETE_MARKER).unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Downloaded snapshot for {repo} is missing config.json or "
                f"safetensors files in {local_dir}."
            )
        return str(local_dir)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _warn_state(self, message: str) -> None:
        if not self._state_warned:
            self._state_warned = True
            logger.warning(message)

    def _resolve_instruction(self, instruction: Optional[str]) -> str:
        text = instruction if instruction is not None else self.instruction
        if not text or not str(text).strip():
            if not self.REQUIRES_INSTRUCTION:
                return ""
            raise ValueError(
                "No instruction given. Pass instruction='...' to predict() or "
                "call set_instruction('...') once."
            )
        return str(text).strip()

    def _build_observation(
        self,
        frames: Dict[str, Any],
        state: Any,
        instruction: str,
        *,
        cameras: Optional[Sequence[str]],
        color_format: str,
        path: Optional[str] = None,
        frame_idx: Optional[int] = None,
    ) -> Observation:
        mapped = map_cameras(frames, self.camera_slots, cameras)
        loaded = load_frames(mapped, color_format=color_format)
        vector = coerce_state(state, self.state_dim, warn=self._warn_state)
        return Observation(loaded, vector, instruction, path=path, frame_idx=frame_idx)

    def _result(self, observation: Observation, chunk: Any) -> Results:
        import torch

        data = torch.as_tensor(chunk).detach().cpu().float()
        if data.ndim == 3:
            if data.shape[0] != 1:
                raise ValueError(
                    f"{type(self).__name__} returned a batch of {data.shape[0]} "
                    "chunks for one observation."
                )
            data = data[0]
        primary = observation.primary
        shape = (primary.height, primary.width)
        actions = Actions(
            data,
            shape,
            names=self.action_names,
            fps=self.fps,
            instruction=observation.instruction,
        )
        return Results(
            None,
            shape,
            path=observation.path,
            names={},
            frame_idx=observation.frame_idx,
            actions=actions,
        )

    def _predict_one(
        self,
        frames: Dict[str, Any],
        state: Any,
        instruction: str,
        *,
        cameras: Optional[Sequence[str]],
        color_format: str,
        path: Optional[str] = None,
        frame_idx: Optional[int] = None,
    ) -> Results:
        self._ensure_loaded()
        observation = self._build_observation(
            frames,
            state,
            instruction,
            cameras=cameras,
            color_format=color_format,
            path=path,
            frame_idx=frame_idx,
        )
        chunk = self._predict_chunk(observation)
        result = self._result(observation, chunk)
        result._libreyolo_frame = observation.primary  # for save=True
        return result

    def predict(
        self,
        source: Union[
            ImageInput, Dict[str, ImageInput], int, Sequence[Any], None
        ] = None,
        *,
        state: Any = None,
        instruction: Optional[str] = None,
        cameras: Optional[Sequence[str]] = None,
        stream: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        vid_stride: int = 1,
        stream_buffer: bool = False,
        color_format: str = "auto",
        **kwargs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Predict an action chunk per observation.

        Args:
            source: One frame (any image input) for a single camera, a
                ``{camera: frame}`` dict for several cameras, a list of
                either, an image directory, a video file, a webcam index or
                a stream URL (live sources need ``stream=True``).
            state: The proprioceptive vector, or a callable returning it,
                called once per frame. ``None`` uses zeros and warns.
            instruction: The task text; else the sticky ``set_instruction``.
            cameras: User camera names in slot order; fixes the mapping of
                dict keys onto the family's camera slots.
            stream: Return a generator instead of a list.
            save: Write ``result.plot()`` renders under ``runs/act/predict``.
            output_path: Explicit output file for a single observation.
            vid_stride: Process every N-th frame of a video or stream.

        Returns:
            ``Results`` for one observation, a list for many, a generator
            with ``stream=True``. Each result carries ``Results.actions``.
        """
        if kwargs:
            logger.warning("Ignoring unknown predict() kwargs: %s", sorted(kwargs))
        text = self._resolve_instruction(instruction)
        cams = [str(c) for c in cameras] if cameras is not None else self.cameras
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        if source is None:
            raise ValueError(
                "predict() needs a source: a frame, a dict of frames, a folder, a video or a stream."
            )
        # One run directory per predict() call, like the shared runner.
        self._save_dir = None

        def one(frames, path=None, frame_idx=None):
            return self._predict_one(
                frames,
                state,
                text,
                cameras=cams,
                color_format=color_format,
                path=path,
                frame_idx=frame_idx,
            )

        # A single multi-camera observation.
        if isinstance(source, dict):
            result = one(source)
            if save:
                self._save_render(result, 0, output_path)
            return result

        spec = classify_source(source)
        is_many = True
        generator: Iterable[Results]

        if spec.kind == SourceKind.IMAGE:
            is_many = False
            generator = iter(
                [one({self.camera_slots[0]: source}, self._path_of(source))]
            )
        elif spec.kind == SourceKind.IMAGE_BATCH:
            items = list(spec.items)
            if not items:
                raise ValueError("Empty source list.")

            def batch_gen():
                for idx, item in enumerate(items):
                    frames = (
                        item if isinstance(item, dict) else {self.camera_slots[0]: item}
                    )
                    yield one(frames, self._path_of(item), idx)

            generator = batch_gen()
        elif spec.kind == SourceKind.DIRECTORY:
            paths = sorted(
                p
                for p in Path(source).iterdir()
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No images found in directory {source}.")
            generator = (
                one({self.camera_slots[0]: p}, str(p), idx)
                for idx, p in enumerate(paths)
            )
        elif spec.kind == SourceKind.IMAGE_SEQUENCE:
            generator = (
                one(
                    item if isinstance(item, dict) else {self.camera_slots[0]: item},
                    None,
                    idx,
                )
                for idx, item in enumerate(source)
            )
        elif spec.kind == SourceKind.VIDEO:
            generator = self._video_frames(source, one, vid_stride)
        elif spec.kind in (SourceKind.STREAM, SourceKind.STREAMS):
            if not stream:
                raise ValueError(
                    "Live sources are unbounded; call predict(..., stream=True) "
                    "and iterate the generator."
                )
            generator = self._stream_frames(spec, one, vid_stride, stream_buffer)
        else:
            raise ValueError(
                f"Source kind {spec.kind.value!r} is not supported by LibreVLA yet."
            )

        if save:
            generator = self._saving(generator, output_path if not is_many else None)
        if stream:
            return generator
        results = list(generator)
        return results if is_many else results[0]

    __call__ = predict

    @staticmethod
    def _path_of(item: Any) -> Optional[str]:
        return str(item) if isinstance(item, (str, Path)) else None

    def _video_frames(self, source, one, vid_stride: int):
        from ...utils.video import VideoSource

        with VideoSource(source, vid_stride=vid_stride) as frames:
            for frame_bgr, frame_idx in frames:
                image = Image.fromarray(frame_bgr[:, :, ::-1])
                yield one({self.camera_slots[0]: image}, str(source), frame_idx)

    def _stream_frames(self, spec, one, vid_stride: int, stream_buffer: bool):
        capture = build_stream_source(
            spec, vid_stride=vid_stride, stream_buffer=stream_buffer
        )
        with capture as frames:
            for frame in frames:
                image = Image.fromarray(frame.frame_bgr[:, :, ::-1])
                yield one(
                    {self.camera_slots[0]: image}, frame.source_label, frame.frame_idx
                )

    def _saving(self, results: Iterable[Results], output_path: Optional[str]):
        for idx, result in enumerate(results):
            self._save_render(result, idx, output_path)
            yield result

    def _save_render(
        self, result: Results, index: int, output_path: Optional[str]
    ) -> None:
        from ...utils.general import increment_path, log_saved_result

        if output_path is not None:
            destination = Path(output_path)
        else:
            if getattr(self, "_save_dir", None) is None:
                self._save_dir = increment_path(Path("runs/act/predict"), mkdir=True)
            stem = (
                Path(result.path).stem
                if result.path and not str(result.path).isdigit()
                else "frame"
            )
            destination = self._save_dir / f"{stem}_{index}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame = getattr(result, "_libreyolo_frame", None)
        result.plot(frame).save(destination)
        log_saved_result(result, destination)

    # ------------------------------------------------------------------
    # Train / val / unsupported
    # ------------------------------------------------------------------

    def train(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Fine-tune on a LeRobot dataset (Hub repo id or local directory).

        Args:
            data: LeRobot dataset repo id (``"lerobot/svla_so101_pickplace"``)
                or a local directory in the LeRobot v3 layout.
            **kwargs: ``epochs``, ``batch``, ``lr0``, ``accumulate``,
                ``val_split`` / ``val_episodes`` / ``train_episodes``,
                ``output_dir`` / ``project`` / ``name`` / ``exist_ok``,
                ``workers``, ``seed``, ``device``, ``callbacks``, ``loggers``,
                ``max_steps`` (cap steps per epoch, for smoke runs).

        Returns:
            The standard results dict with ``save_dir``, ``best``, ``last``
            and the final metrics. Load a checkpoint with
            ``LibreVLA(results["best"])``.
        """
        if not self.TRAINABLE:
            raise NotImplementedError(
                self.TRAIN_UNSUPPORTED_REASON
                or f"Training is not supported for {type(self).__name__} yet."
            )
        if not data:
            raise ValueError(
                "train() requires data=<LeRobot dataset repo id or directory>, "
                'e.g. train(data="lerobot/svla_so101_pickplace").'
            )
        from .training.trainer import VLATrainer

        return VLATrainer(self, data=data, **kwargs).run()

    def val(self, data: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Offline action error on held-out episodes of a LeRobot dataset.

        Returns mean L1 / MSE between predicted and recorded action chunks
        in the dataset's units, overall and per dimension. See
        ``libreyolo.models.vla.metrics.action_error``.
        """
        if not data:
            data = self.contract.get("data") if self.contract else None
        if not data:
            raise ValueError(
                "val() requires data=<LeRobot dataset repo id or directory>."
            )
        from .training.trainer import VLAValidator

        return VLAValidator(self, data=data, **kwargs).run()

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not export to {format!r}: the action "
            "expert's sampling loop has no exportable graph contract yet "
            "(ADR 0028). Run it through predict()."
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} returns action chunks, not boxes; "
            "tracking does not apply."
        )

    def benchmark(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not covered by the export benchmark."
        )
