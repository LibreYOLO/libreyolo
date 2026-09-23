"""LibreVJEPA2: V-JEPA 2.0 video embedding and attentive-probe classification.

V-JEPA 2 is a self-supervised *video* encoder. It is not a text-image model and
provides no text embeddings, so there is no shared text space and no retrieval
benchmark is claimed for the vector this family returns.

Three different representations are involved and are deliberately kept
distinct:

* **Native tokens** -- the encoder's spatiotemporal patch tokens, shaped
  ``(B, T', H', W', D)``. Reached only through :meth:`LibreVJEPA2.embed_tokens`,
  never through the generic ``Embeddings`` container, which is a 2D contract.
* **The LibreYOLO global vector** -- ``normalize(tokens.mean(dim=1), dim=-1)``.
  This is a LibreYOLO *pooling contract*, not an upstream retrieval head;
  upstream designates no global vector.
* **Attentive-probe logits** -- from the released classification heads.

Image input is supported as an explicitly documented single-frame
representation. Every frame is identical in that case, so it is a static
appearance embedding and must not be described as a motion representation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...tasks import normalize_task
from ...utils.image_loader import ImageLoader
from ..base.model import BaseModel
from .nn import (
    VJEPA2_CONFIGS,
    LibreVJEPA2Classifier,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)
from .validator import VJEPA2ClipValidator
from .preprocess import (
    clip_frame_indices,
    DEFAULT_FRAME_STRIDE,
    image_to_clip,
    preprocess_frames,
    validate_clip_tensor,
)

logger = logging.getLogger(__name__)

# Frames per clip for each *released* artifact. The encoders are all 64-frame;
# the probes vary, and a probe must be run at the frame count it was trained
# for or its pooled statistics are wrong.
ENCODER_FRAMES: int = 64
PROBE_FRAMES: Dict[Tuple[str, str], int] = {
    ("l256", "ssv2"): 16,
    ("l256", "diving48"): 32,
    ("g384", "ssv2"): 64,
    ("g384", "diving48"): 32,
}
PROBE_CLASSES: Dict[str, int] = {"ssv2": 174, "diving48": 48}


class LibreVJEPA2(BaseModel):
    """V-JEPA 2.0 video encoder (``embed``) and attentive probe (``classify``)."""

    FAMILY: ClassVar[str] = "vjepa2"
    FILENAME_PREFIX: ClassVar[str] = "LibreVJEPA2"
    WEIGHT_EXT: ClassVar[str] = ".pt"

    # Sizes encode the crop, because g256 and g384 share a width and differ
    # only in input resolution.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "l256": 256,
        "h256": 256,
        "g256": 256,
        "g384": 384,
    }
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("embed", "classify")
    WEIGHT_TASKS: ClassVar[Tuple[str, ...]] = ("embed", "classify")
    DEFAULT_TASK: ClassVar[str] = "embed"
    # Every published artifact carries a task suffix; a bare ``LibreVJEPA2l256.pt``
    # is not canonical and must not resolve.
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    WEIGHT_VARIANTS: ClassVar[Tuple[str, ...]] = ("ssv2", "diving48")

    # Self-supervised pretraining is out of scope; the probe trainer is wired
    # separately in trainer.py.
    TRAIN_CONFIG: ClassVar[Optional[type]] = None

    # Opt this family into clip-mode finite-video handling. Every other family
    # keeps the default "frames" behaviour and its per-frame result cardinality.
    VIDEO_EMBED_MODE: ClassVar[str] = "clip"

    # val() and epoch validation read the video val manifest (see validator.py).
    validator_class: ClassVar[Optional[type]] = VJEPA2ClipValidator

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    # An ``-embed`` checkpoint is rooted at the encoder itself, while a
    # ``-cls`` checkpoint nests it under ``encoder.``. Both roots are accepted
    # so one discriminator serves both artifact shapes.
    _PATCH_EMBED_KEYS = (
        "embeddings.patch_embeddings.proj.weight",
        "encoder.embeddings.patch_embeddings.proj.weight",
    )
    _FINAL_NORM_KEYS = ("layernorm.weight", "encoder.layernorm.weight")

    @classmethod
    def _lookup(cls, weights_dict: dict, keys: Tuple[str, ...]):
        for key in keys:
            value = weights_dict.get(key)
            if value is not None:
                return value
        return None

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match on the 3D tubelet patch embedding, which is unique to V-JEPA 2.

        A 5D conv weight under this key does not occur in any other registered
        family: the image ViTs all use a 4D Conv2d patch embedding, so the
        rank check is what makes this safe rather than the name alone.
        """
        weight = cls._lookup(weights_dict, cls._PATCH_EMBED_KEYS)
        if weight is None:
            return False
        # Guard the structure too, not just the name.
        return getattr(weight, "ndim", 0) == 5

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer size from encoder width.

        Width alone cannot separate ``g256`` from ``g384`` -- they are the same
        network at different input resolutions. That pair is resolved from
        checkpoint metadata or the filename, never guessed here.
        """
        weight = cls._lookup(weights_dict, cls._FINAL_NORM_KEYS)
        if weight is None:
            return None
        hidden = int(weight.shape[0])
        if hidden == 1024:
            return "l256"
        if hidden == 1280:
            return "h256"
        # 1408 is ambiguous between g256 and g384 by design.
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("classifier.weight")
        if weight is None:
            return None
        return int(weight.shape[0])

    @classmethod
    def detect_task_from_state_dict(cls, weights_dict: dict) -> Optional[str]:
        if any(k.startswith("pooler.") for k in weights_dict):
            return "classify"
        if cls.can_load(weights_dict):
            return "embed"
        return None

    @classmethod
    def validate_artifact_name(cls, size: str, task: str, variant: Optional[str]) -> None:
        """Reject impossible size/task/variant combinations.

        The base filename regex can parse combinations that no released
        artifact provides. Parsing is not the same as existing, so the valid
        matrix is enforced here on top of it.
        """
        task = normalize_task(task)
        if size not in cls.INPUT_SIZES:
            raise ValueError(
                f"unknown V-JEPA 2 size {size!r}; expected one of "
                f"{sorted(cls.INPUT_SIZES)}"
            )
        if task == "embed":
            if variant:
                raise ValueError(
                    f"'-embed' artifacts carry no dataset variant, got {variant!r}. "
                    f"Canonical name: LibreVJEPA2{size}-embed.pt"
                )
            return
        if task == "classify":
            if not variant:
                raise ValueError(
                    "published '-cls' artifacts require a dataset variant "
                    f"({' or '.join(cls.WEIGHT_VARIANTS)}), e.g. "
                    f"LibreVJEPA2{size}-cls-ssv2.pt. A locally trained probe "
                    "should be loaded from its explicit path instead."
                )
            if (size, variant) not in PROBE_FRAMES:
                published = ", ".join(
                    f"LibreVJEPA2{s}-cls-{v}.pt" for s, v in sorted(PROBE_FRAMES)
                )
                raise ValueError(
                    f"no released V-JEPA 2 probe for size={size!r} variant={variant!r}. "
                    f"Published probes are: {published}"
                )
            return
        raise ValueError(f"V-JEPA 2 supports 'embed' and 'classify'; got {task!r}")

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "l256",
        nb_classes: int = 174,
        device: str = "auto",
        task: str | None = None,
        variant: str | None = None,
        clip_frames: int | None = None,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else self.DEFAULT_TASK
        if resolved_task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"LibreVJEPA2 supports task in {self.SUPPORTED_TASKS}; got {task!r}."
            )
        self.variant = variant
        self._requested_clip_frames = clip_frames
        self.frame_stride = frame_stride
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        # BaseModel.__init__ only records the path (and loads an inline state
        # dict); loading from a file is the family's job, as in every sibling.
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
            self._adopt_checkpoint_clip_geometry(str(model_path))

    def _adopt_checkpoint_clip_geometry(self, model_path: str) -> None:
        """Recover the dataset variant and frame count from checkpoint metadata.

        A probe must run at the frame count it was trained for -- an SSv2 L/256
        probe is a 16-frame artifact, and silently running it at the encoder's
        64 frames would change its pooled statistics and its logits. The
        factory does not thread ``variant`` through, so it is read back here
        rather than guessed from the size.
        """
        try:
            from ...utils.serialization import load_trusted_torch_file

            checkpoint = load_trusted_torch_file(model_path, map_location="cpu")
        except Exception:  # pragma: no cover - metadata is best-effort
            return
        if not isinstance(checkpoint, dict):
            return

        variant = checkpoint.get("variant")
        if isinstance(variant, str) and variant in self.WEIGHT_VARIANTS:
            self.variant = variant

        frames = checkpoint.get("frames_per_clip")
        if isinstance(frames, int) and frames > 0:
            self._requested_clip_frames = frames

        stride = checkpoint.get("frame_stride")
        if isinstance(stride, int) and stride > 0:
            self.frame_stride = stride

    @property
    def clip_frames(self) -> int:
        """Frames the loaded artifact expects for one clip."""
        if self._requested_clip_frames is not None:
            return int(self._requested_clip_frames)
        if self.task == "classify" and self.variant:
            return PROBE_FRAMES.get((self.size, self.variant), ENCODER_FRAMES)
        return ENCODER_FRAMES

    @property
    def crop_size(self) -> int:
        return self.INPUT_SIZES[self.size]

    @property
    def embedding_dim(self) -> int:
        return VJEPA2_CONFIGS[self.size]["hidden_size"]

    def _build_config(self) -> VJEPA2Config:
        return VJEPA2Config.for_size(self.size, frames_per_clip=self.clip_frames)

    def _init_model(self) -> nn.Module:
        config = self._build_config()
        if self.task == "classify":
            return LibreVJEPA2Classifier(config, nc=self.nb_classes)
        return LibreVJEPA2Encoder(config)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        if self.task == "classify":
            return {
                "encoder": self.model.encoder,
                "pooler": self.model.pooler,
                "classifier": self.model.classifier,
            }
        return {"encoder": self.model}

    # =========================================================================
    # Preprocess / forward / postprocess
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_frames

    def _preprocess(
        self,
        image,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        """Preprocess ONE frame to ``(1, C, H, W)``, or pass through a 5D clip.

        Returns the shared 4-tuple
        ``(input_tensor, original_image, original_size, ratio)``.

        The per-frame shape is deliberate: the shared whole-clip runner
        preprocesses each sampled frame and concatenates them into
        ``(1, F, C, H, W)``. Returning a whole clip from here would break that
        contract. A still image therefore also arrives as one frame, and
        ``_forward`` is what expands it into a static clip.
        """
        del input_size  # clip geometry is fixed by the checkpoint, not the call

        if isinstance(image, torch.Tensor) and image.ndim == 5:
            tensor = validate_clip_tensor(image, self.crop_size)
            size = (int(tensor.shape[-1]), int(tensor.shape[-2]))
            return tensor.to(self.device), image, size, 1.0

        if isinstance(image, (list, tuple)):
            frames = [
                np.asarray(ImageLoader.load(f, color_format=color_format))
                for f in image
            ]
            original = frames[-1]
            height, width = original.shape[:2]
            tensor = preprocess_frames(frames, self.crop_size)
            return tensor.to(self.device), original, (width, height), 1.0

        loaded = ImageLoader.load(image, color_format=color_format)
        frame = np.asarray(loaded)
        height, width = frame.shape[:2]
        # (1, 1, C, H, W) -> (1, C, H, W): one frame, stackable by the runner.
        tensor = preprocess_frames([frame], self.crop_size)[0]
        return tensor.to(self.device), loaded, (width, height), 1.0

    def _get_eval_transform(
        self, img_size: Optional[int] = None, crop_pct: Optional[float] = None
    ):
        """Still-image eval transform: the frame code ``predict()`` runs (#886).

        One frame through :func:`preprocess_frames` (short side to
        ``crop_size``, center crop, PIXEL_MEAN/STD). The crop geometry is fixed
        by the checkpoint, so another size or a ``crop_pct`` is rejected rather
        than ignored.
        """
        if img_size is not None:
            size = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
            if int(size) != int(self.crop_size):
                raise ValueError(
                    f"LibreVJEPA2 {self.size!r} preprocesses at its fixed crop "
                    f"size {self.crop_size}; got imgsz={img_size}."
                )
        if crop_pct is not None:
            raise ValueError(
                "LibreVJEPA2 has no crop_pct: frames are resized short side to "
                "the crop size and center-cropped."
            )
        crop_size = self.crop_size

        def _transform(image) -> torch.Tensor:
            frame = np.asarray(image.convert("RGB"))
            return preprocess_frames([frame], crop_size)[0, 0]

        return _transform

    def _as_clip(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Promote a 4D still frame to a 5D static clip; pass 5D through.

        Every caller that reaches the encoder must go through here. The encoder
        is 5D-only, and ``_preprocess`` returns a single frame so the shared
        whole-clip runner can concatenate frames itself, so a still image
        arrives here as 4D and would otherwise hit a bare rank error.

        A static clip repeats one frame, so it contains no motion at all. That
        is a documented compatibility behaviour, not a motion representation.
        """
        if input_tensor.ndim == 4:
            return input_tensor.unsqueeze(1).repeat(1, self.clip_frames, 1, 1, 1)
        if input_tensor.ndim != 5:
            raise ValueError(
                "V-JEPA 2 consumes a 5D clip (B, F, C, H, W) or a 4D still "
                f"frame (B, C, H, W); got {input_tensor.ndim}D shape "
                f"{tuple(input_tensor.shape)}."
            )
        return input_tensor

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        input_tensor = self._as_clip(input_tensor)
        if self.task == "classify":
            return self.model(input_tensor)
        tokens = self.model(input_tensor)
        return self.pool_tokens(tokens)

    def sample_clip_frames(self, source, clip_frames: int) -> list:
        """Family-local temporal sampling: a centered, strided window.

        The shared runner samples uniformly across the whole video, which is
        right for a general clip embedder but wrong here: the released
        checkpoints were trained on a fixed window (64 frames at stride 2,
        spanning 127 source frames), and spreading those frames across a long
        video would feed the model statistics it never saw.
        """
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                # Header has no frame count: count without decoding pixels,
                # so memory stays proportional to the clip, not the video.
                total = 0
                while capture.grab():
                    total += 1
                capture.release()
                capture = cv2.VideoCapture(str(source))
            if total < 1:
                raise ValueError(
                    f"Video decoded zero frames: {source}; refusing to treat a "
                    "corrupt or empty video as a black clip."
                )

            indices = clip_frame_indices(total, int(clip_frames), self.frame_stride)
            wanted = set(indices)
            decoded = {}
            position = 0
            highest = max(wanted)
            while position <= highest:
                ok, frame = capture.read()
                if not ok:
                    break
                if position in wanted:
                    decoded[position] = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                position += 1
            if not decoded:
                raise ValueError(f"Could not decode any frame from {source}.")

            held = decoded[min(decoded)]
            frames = []
            for index in indices:
                held = decoded.get(index, held)
                frames.append(held)
            return frames
        finally:
            capture.release()

    @staticmethod
    def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """The LibreYOLO pooling contract: mean over all tokens, then L2.

        Averaging is over every temporal *and* spatial position, so the result
        is a single float32 row per clip.
        """
        return F.normalize(tokens.mean(dim=1), dim=-1).float()

    def embed_tokens(self, source, **kwargs) -> torch.Tensor:
        """Family escape hatch returning native tokens as ``(B, T', H', W', D)``.

        Ordering is upstream's: time-major, then height, then width. This is
        deliberately not routed through ``Embeddings``, which is a 2D contract
        and cannot represent a token grid.
        """
        if self.task != "embed":
            raise ValueError(
                "embed_tokens() requires task='embed'; this model is "
                f"task={self.task!r}."
            )
        tensor, _, _, _ = self._preprocess(source, **kwargs)
        # A still image arrives as a 4D frame; the encoder is 5D-only.
        tensor = self._as_clip(tensor)
        with torch.no_grad():
            encoder = self.model
            tokens = encoder(tensor)
        config = self._build_config()
        grid = config.grid_size
        depth = tensor.shape[1] // config.tubelet_size
        batch, num_tokens, dim = tokens.shape
        expected = depth * grid * grid
        if num_tokens != expected:
            raise RuntimeError(
                f"expected {expected} tokens for a {tensor.shape[1]}-frame clip at "
                f"{config.crop_size}px, got {num_tokens}"
            )
        return tokens.reshape(batch, depth, grid, grid, dim)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self.task == "classify":
            logits = output
            probs = torch.softmax(logits.float(), dim=1)[0]
            return {"probs": probs.cpu()}
        return self._postprocess_embeddings(
            output,
            gallery=kwargs.get("gallery"),
            threshold=kwargs.get("threshold"),
        )

    # =========================================================================
    # Training
    # =========================================================================

    def train(
        self,
        data: str,
        *,
        epochs: int = 10,
        batch: int = 2,
        lr0: float = 1e-3,
        device: str = "",
        workers: int = 0,
        seed: int = 0,
        project: str = "runs",
        name: str = "train",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        freeze: int = 1,
        callbacks=None,
        **kwargs: Any,
    ) -> dict:
        """Train the attentive probe on a user-supplied video dataset.

        The encoder is frozen and kept in eval mode; only the three-layer
        attentive pooler and the linear classifier are optimized. ``data`` is a
        dataset YAML whose manifests are validated in full before the first
        epoch. Nothing is downloaded: the videos are yours.
        """
        # Reject the unsupported training stories before building a dataset.
        if self.task == "embed":
            raise NotImplementedError(
                "V-JEPA 2 embedding training is self-supervised pretraining: it "
                "needs web-scale mixed video corpora, two training phases and "
                "distributed recipes, so LibreYOLO does not implement it. Train "
                "the attentive probe instead:\n"
                "    model = LibreYOLO('LibreVJEPA2l256-embed.pt', task='classify')\n"
                "    model.train(data='video_dataset.yaml')"
            )
        if kwargs.pop("pretrain", False) or kwargs.pop("predictor", False):
            raise NotImplementedError(
                "V-JEPA 2 predictor / self-supervised pretraining is not "
                "implemented and is out of scope for this family."
            )
        if freeze == 0:
            raise NotImplementedError(
                "Full encoder fine-tuning (freeze=0) is not wired for V-JEPA 2. "
                "The supported recipe trains the attentive pooler and classifier "
                "with the encoder frozen; unfreezing a 1B-parameter video "
                "encoder needs far more memory than the probe recipe and has no "
                "validated schedule here."
            )

        from .trainer import VJEPA2Trainer

        trainer = VJEPA2Trainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=self.crop_size,
            lr0=lr0,
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            callbacks=callbacks,
            **kwargs,
        )
        return trainer.train()
