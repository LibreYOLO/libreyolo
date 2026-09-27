"""LibreLeVJEPA: inference-only block-causal video embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.image_loader import ImageLoader
from ...utils.video import is_video_file
from ..base.model import BaseModel
from .nn import LEVJEPA_CONFIGS, LeVJEPAConfig, LeVJEPAModel
from .preprocess import (
    TARGET_FPS,
    clip_frame_indices,
    preprocess_frames,
    validate_clip_tensor,
)


class LibreLeVJEPA(BaseModel):
    """LeVJEPA Large video encoder for clip and patch embeddings."""

    FAMILY: ClassVar[str] = "levjepa"
    FILENAME_PREFIX: ClassVar[str] = "LibreLeVJEPA"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"l": 224}
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("embed",)
    WEIGHT_TASKS: ClassVar[tuple[str, ...]] = ("embed",)
    DEFAULT_TASK: ClassVar[str] = "embed"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    TRAIN_CONFIG: ClassVar[None] = None
    VIDEO_EMBED_MODE: ClassVar[str] = "clip"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        patch = weights_dict.get("encoder.patch_embed.proj.weight")
        return bool(
            getattr(patch, "ndim", 0) == 5
            and tuple(patch.shape[2:]) == (1, 16, 16)
            and "encoder.cls_token" in weights_dict
            and "encoder.blocks.0.attn.qkv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        norm = weights_dict.get("encoder.norm.weight")
        if norm is not None and tuple(norm.shape) == (1024,):
            return "l"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_task_from_state_dict(cls, weights_dict: dict) -> Optional[str]:
        return "embed" if cls.can_load(weights_dict) else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str:
        return (
            f"{Path(filename).name} contains LeVJEPA weights released under "
            "CC BY-NC 4.0. They require attribution, are for NON-COMMERCIAL "
            "use only, and are not covered by LibreYOLO's MIT license. "
            "LibreYOLO's native inference code remains permissively licensed."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "l",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task or self.DEFAULT_TASK,
            **kwargs,
        )
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    @property
    def clip_frames(self) -> int:
        return int(LEVJEPA_CONFIGS[self.size]["num_frames"])

    @property
    def crop_size(self) -> int:
        return self.INPUT_SIZES[self.size]

    @property
    def embedding_dim(self) -> int:
        return int(LEVJEPA_CONFIGS[self.size]["embed_dim"])

    def _build_config(self) -> LeVJEPAConfig:
        return LeVJEPAConfig.for_size(self.size)

    def _init_model(self) -> nn.Module:
        return LeVJEPAModel(self._build_config())

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {"encoder": self.model.encoder}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_frames

    def _preprocess(
        self,
        image,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        del input_size
        if isinstance(image, torch.Tensor) and image.ndim == 5:
            tensor = validate_clip_tensor(
                image, frames=self.clip_frames, size=self.crop_size
            )
            size = (int(tensor.shape[-1]), int(tensor.shape[-2]))
            return tensor.to(self.device), image, size, 1.0

        if isinstance(image, (list, tuple)):
            if len(image) != self.clip_frames:
                raise ValueError(
                    f"LeVJEPA requires exactly {self.clip_frames} frames, "
                    f"got {len(image)}"
                )
            frames = [
                np.asarray(ImageLoader.load(frame, color_format=color_format))
                for frame in image
            ]
            original = frames[-1]
            height, width = original.shape[:2]
            tensor = preprocess_frames(frames, self.crop_size)
            return tensor.to(self.device), original, (width, height), 1.0

        loaded = ImageLoader.load(image, color_format=color_format)
        frame = np.asarray(loaded)
        height, width = frame.shape[:2]
        tensor = preprocess_frames([frame], self.crop_size)[0]
        return tensor.to(self.device), loaded, (width, height), 1.0

    def _as_clip(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            return tensor.unsqueeze(1).repeat(1, self.clip_frames, 1, 1, 1)
        return validate_clip_tensor(
            tensor, frames=self.clip_frames, size=self.crop_size
        ).to(self.device)

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        tokens = self.model(self._as_clip(input_tensor))
        return F.normalize(tokens[:, 0].float(), dim=-1)

    def embed_tokens(self, source, **kwargs) -> torch.Tensor:
        """Return patch tokens as ``(B, F, H, W, D)`` without the CLS token."""

        if is_video_file(source):
            source = self.sample_clip_frames(source, self.clip_frames)
        tensor, _, _, _ = self._preprocess(source, **kwargs)
        tensor = self._as_clip(tensor)
        with torch.no_grad():
            tokens = self.model(tensor)[:, 1:]
        config = self._build_config()
        return tokens.reshape(
            tokens.shape[0],
            config.temporal_grid_size,
            config.grid_size,
            config.grid_size,
            config.embed_dim,
        )

    def sample_clip_frames(self, source, clip_frames: int) -> list:
        """Decode a centered approximately-7.5-FPS window from a finite video."""

        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if total < 1:
                total = 0
                while capture.grab():
                    total += 1
                capture.release()
                capture = cv2.VideoCapture(str(source))
            indices = clip_frame_indices(
                total,
                clip_frames,
                source_fps=source_fps,
                target_fps=TARGET_FPS,
            )
            wanted = set(indices)
            decode_start = min(wanted)
            if decode_start:
                seeked = capture.set(cv2.CAP_PROP_POS_FRAMES, decode_start)
                reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if not seeked or abs(reported - decode_start) > 1:
                    capture.release()
                    capture = cv2.VideoCapture(str(source))
                    if not capture.isOpened():
                        raise ValueError(f"Could not reopen video: {source}")
                    decode_start = 0
            decoded = {}
            for position in range(decode_start, max(wanted) + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if position in wanted:
                    decoded[position] = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
            if not decoded:
                raise ValueError(f"Could not decode any frame from {source}")
            held = decoded[min(decoded)]
            frames = []
            for index in indices:
                held = decoded.get(index, held)
                frames.append(held)
            return frames
        finally:
            capture.release()

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size,
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        del conf_thres, iou_thres, original_size, max_det
        return self._postprocess_embeddings(
            output,
            gallery=kwargs.get("gallery"),
            threshold=kwargs.get("threshold"),
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LeVJEPA is inference-only in LibreYOLO; self-supervised pretraining "
            "is intentionally out of scope."
        )
