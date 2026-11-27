"""LibreLoopDVR — BaseModel wrapper for LoopDVR (LoopDet, LoopSeg, LoopPose).

LoopDVR (arXiv:2611.32067, Great Bay University / Intellindust AI Lab, 2026)
turns EdgeCrafter's ViT backbone and recurrent decoder into looped architectures
with per-iteration rank-modulated LoRA adapters, reducing parameters by ~30%
across object detection, instance segmentation, and human pose estimation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from libreyolo.training.ddp_spawn import ddp_aware

from ...postprocess.loopdvr import postprocess, postprocess_pose, postprocess_seg
from ...preprocess.ec import preprocess_numpy
from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import ECValPreprocessor
from ..base import BaseModel
from .nn import (
    LibreLoopDVRModel,
    LibreLoopDVRPoseModel,
    LibreLoopDVRSegModel,
)
from .trainer import LoopDVRConfig

_POSE_HEAD_KEY = "decoder.keypoint_embedding.weight"
_SEG_HEAD_KEY = "decoder.decoder.segmentation_head.bias"


def unwrap_loopdvr_checkpoint(checkpoint: Mapping | Any):
    """Extract the model state_dict from upstream or LibreYOLO LoopDVR checkpoints."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        module = ema.get("module")
        if isinstance(module, Mapping):
            return module
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> tuple[torch.Tensor, Any, tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio


class LibreLoopDVR(BaseModel):
    """LibreYOLO wrapper for LoopDVR (LoopDet / LoopSeg / LoopPose)."""

    FAMILY = "loopdvr"
    RESUME_RESTORES_TRAIN_ARGS = True
    FILENAME_PREFIX = "LibreLoopDVR"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"s": 640, "m": 640, "l": 640, "x": 640}
    POSE_INPUT_SIZES: ClassVar[dict[str, int]] = {
        "s": 640,
        "m": 640,
        "l": 640,
        "x": 640,
    }
    SEG_INPUT_SIZES: ClassVar[dict[str, int]] = {"s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect", "pose", "segment")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {
        "detect": INPUT_SIZES,
        "pose": POSE_INPUT_SIZES,
        "segment": SEG_INPUT_SIZES,
    }
    TRAIN_CONFIG: ClassVar[type[LoopDVRConfig]] = LoopDVRConfig
    TTA_FIXED_SIZE = True  # fixed 640x640 eval spatial grid
    POSE_NUM_KEYPOINTS = 17
    KEYPOINT_DIM = 3
    val_preprocessor_class = ECValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Fingerprint LoopDVR's looped ViT backbone and per-iteration adapters.

        ``looped_vit.register_token`` together with a looped-adapter coefficient
        weight is unique to LoopDVR and disjoint from EdgeCrafter
        (``backbone.backbone.register_token``) and every other DETR family.
        """
        return (
            "looped_vit.register_token" in weights_dict
            and "looped_vit.blocks.0.loop_adapters.0.coefficient1.bias"
            in weights_dict
        )

    @classmethod
    def is_pose_state_dict(cls, weights_dict: dict) -> bool:
        return _POSE_HEAD_KEY in weights_dict

    @classmethod
    def is_seg_state_dict(cls, weights_dict: dict) -> bool:
        return _SEG_HEAD_KEY in weights_dict or any(
            k.startswith("decoder.decoder.segmentation_head.") for k in weights_dict
        )

    @classmethod
    def detect_task_from_state_dict(cls, weights_dict: dict) -> str | None:
        if cls.is_pose_state_dict(weights_dict):
            return "pose"
        if cls.is_seg_state_dict(weights_dict):
            return "segment"
        return None

    @classmethod
    def detect_checkpoint_task(cls, weights_dict: dict) -> str | None:
        return cls.detect_task_from_state_dict(weights_dict)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> str | None:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        match = re.search(
            r"loop(?:det|seg|pose)[-_]([smlx])(?=[._-]|$)",
            Path(filename).name.lower(),
        )
        if match:
            return match.group(1)
        return None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> str | None:
        detected = super().detect_task_from_filename(filename)
        if detected is not None:
            return detected
        name = Path(filename).name.lower()
        if re.search(r"loopseg[-_][smlx](?=[._-]|$)", name):
            return "segment"
        if re.search(r"looppose[-_][smlx](?=[._-]|$)", name):
            return "pose"
        if re.search(r"loopdet[-_][smlx](?=[._-]|$)", name):
            return "detect"
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        """Infer model size (s/m/l/x) from backbone embedding, projector, and MLP shapes.

        S: embed=192, proj=192
        M: embed=256, proj=256
        L: embed=384, proj=256, mlp.fc1 out=1536 (ffn_ratio=4.0), fpn cv4 fan-in=704
        X: embed=384, proj=256, mlp.fc1 out=2304 (ffn_ratio=6.0), fpn cv4 fan-in=896
        """
        reg_key = "looped_vit.register_token"
        proj_key = "looped_vit.projector.0.conv.weight"
        if reg_key not in weights_dict or proj_key not in weights_dict:
            return None
        embed_dim = int(weights_dict[reg_key].shape[-1])
        proj_dim = int(weights_dict[proj_key].shape[0])

        if embed_dim == 192 and proj_dim == 192:
            return "s"
        if embed_dim == 256 and proj_dim == 256:
            return "m"
        if embed_dim == 384 and proj_dim == 256:
            mlp_key = "looped_vit.blocks.0.mlp.fc1.weight"
            if mlp_key in weights_dict:
                mlp_hidden = int(weights_dict[mlp_key].shape[0])
                if mlp_hidden == 1536:
                    return "l"
                if mlp_hidden == 2304:
                    return "x"
            cv4_key = "encoder.fpn_blocks.0.cv4.conv.weight"
            if cv4_key in weights_dict:
                fan_in = int(weights_dict[cv4_key].shape[1])
                if fan_in == 704:
                    return "l"
                if fan_in == 896:
                    return "x"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        if cls.is_pose_state_dict(weights_dict):
            return 1
        key = "decoder.dec_score_head.0.bias"
        if key in weights_dict:
            return int(weights_dict[key].shape[0])
        return None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        checkpoint = model_path if isinstance(model_path, dict) else None
        if isinstance(model_path, dict):
            model_path = unwrap_loopdvr_checkpoint(model_path)
        resolved_task = normalize_task(task) if task is not None else None
        if resolved_task == "pose":
            nb_classes = 1
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = self.KEYPOINT_DIM
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if checkpoint is not None:
            self._cache_checkpoint_train_config(checkpoint)
        if self.task == "pose":
            self.names = {0: "person"}
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        if self.task == "pose":
            return LibreLoopDVRPoseModel(
                config=self.size,
                eval_spatial_size=(self.input_size, self.input_size),
            )
        if self.task == "segment":
            return LibreLoopDVRSegModel(
                config=self.size,
                nb_classes=self.nb_classes,
                eval_spatial_size=(self.input_size, self.input_size),
            )
        return LibreLoopDVRModel(
            config=self.size,
            nb_classes=self.nb_classes,
            eval_spatial_size=(self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> dict[str, nn.Module]:
        layers: dict[str, nn.Module] = {
            "backbone": self.model.looped_vit,
            "looped_vit": self.model.looped_vit,
            "backbone_projector": self.model.looped_vit.projector,
            "encoder": self.model.encoder,
            "encoder_fpn": self.model.encoder.fpn_blocks,
            "encoder_pan": self.model.encoder.pan_blocks,
            "decoder": self.model.decoder,
        }
        if self.task == "pose":
            layers["pose_embed"] = self.model.decoder.pose_embed
            layers["class_embed"] = self.model.decoder.class_embed
        else:
            layers["decoder_input_proj"] = self.model.decoder.input_proj
            layers["dec_bbox_head"] = self.model.decoder.dec_bbox_head
            layers["dec_score_head"] = self.model.decoder.dec_score_head
        return layers

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        if self.task == "pose":
            return
        super()._rebuild_for_new_classes(new_nb_classes)

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | None = None,
    ) -> tuple[torch.Tensor, Any, tuple[int, int], float]:
        effective = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=effective, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        if self.task == "pose":
            return postprocess_pose(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                max_det=max_det,
                num_keypoints=self.POSE_NUM_KEYPOINTS,
            )
        if self.task == "segment":
            return postprocess_seg(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                max_det=max_det,
            )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
        )

    def _strict_loading(self) -> bool:
        return False

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 74,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 5e-4,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "loopdvr_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune LoopDVR on a YOLO-format dataset (detect / segment / pose).

        The task is taken from ``self.task`` (set at load time): ``detect`` uses
        :class:`LoopDVRTrainer` (MAL+L1+GIoU+FGL+DDF + contrastive denoising),
        ``segment`` uses :class:`LoopDVRSegTrainer` (adding point-sampled
        BCE+Dice mask loss), and ``pose`` delegates to :meth:`_train_pose`
        (:class:`LoopDVRPoseTrainer`, DETRPose Hungarian matcher + VFL/L1/OKS
        + contrastive keypoint denoising).
        """
        if self.task == "pose":
            return self._train_pose(
                data=data,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                lr0=lr0,
                device=device,
                workers=workers,
                seed=seed,
                project=project,
                name=name,
                exist_ok=exist_ok,
                resume=resume,
                amp=amp,
                patience=patience,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )

        from libreyolo.data import load_data_config

        if self.task == "segment":
            from .trainer import LoopDVRSegTrainer as trainer_cls
        else:
            from .trainer import LoopDVRTrainer as trainer_cls

        native_imgsz = int(self.input_size)
        if self.task == "segment" and int(imgsz) != native_imgsz:
            raise ValueError(
                "LoopDVR segmentation fine-tuning currently requires "
                f"imgsz={native_imgsz}; the eval anchor grid is built at "
                "model construction for the native input size."
            )

        try:
            data_config = load_data_config(
                data,
                autodownload=True,
                single_cls=bool(kwargs.get("single_cls", False)),
            )
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}") from e

        yaml_nc = data_config.get("nc")
        yaml_names = data_config.get("names")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if yaml_names is not None:
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, self.nb_classes)

        if seed > 0:
            import random as _r

            _r.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = trainer_cls(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
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
            loggers=loggers,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        self.model.to(self.device)
        return results

    def _train_pose(
        self,
        data: str,
        *,
        epochs: int = 74,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 5e-4,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "loopdvr_pose_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        callbacks=None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune LoopPose on a YOLO-format keypoint dataset."""
        from libreyolo.data import load_data_config

        from .trainer import LoopDVRPoseTrainer

        native_imgsz = int(self.input_size)
        if int(imgsz) != native_imgsz:
            raise ValueError(
                "LoopDVR pose fine-tuning currently requires "
                f"imgsz={native_imgsz}; the eval anchor grid is built at "
                "model construction for the native input size."
            )

        flip_idx_override = object()
        explicit_flip_idx = kwargs.pop("flip_idx", flip_idx_override)

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}") from e

        kpt_shape = data_config.get("kpt_shape")
        if not kpt_shape or len(kpt_shape) < 1:
            raise ValueError(
                "Pose training requires 'kpt_shape: [num_keypoints, 2|3]' in the "
                "dataset data.yaml."
            )
        num_keypoints = int(kpt_shape[0])
        keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        if keypoint_dim not in (2, 3):
            raise ValueError(
                "Pose training requires kpt_shape second value to be 2 or 3 "
                f"(got {keypoint_dim})."
            )
        if num_keypoints != self.POSE_NUM_KEYPOINTS:
            raise ValueError(
                f"LoopDVR pose fine-tuning supports {self.POSE_NUM_KEYPOINTS} "
                f"keypoints only (dataset declares {num_keypoints})."
            )

        yaml_nc = data_config.get("nc")
        if yaml_nc is not None and int(yaml_nc) != 1:
            raise ValueError(
                "LoopDVR pose fine-tuning supports single-class pose datasets "
                f"only (dataset declares nc={yaml_nc})."
            )

        yaml_names = data_config.get("names")
        if yaml_names is not None:
            if len(yaml_names) != 1:
                raise ValueError(
                    "LoopDVR pose fine-tuning supports single-class pose datasets "
                    f"only (dataset declares {len(yaml_names)} names)."
                )
            if isinstance(yaml_names, list):
                yaml_names = {i: n for i, n in enumerate(yaml_names)}
            self.names = self._sanitize_names(yaml_names, 1)

        flip_idx = (
            explicit_flip_idx
            if explicit_flip_idx is not flip_idx_override
            else data_config.get("flip_idx")
        )

        if seed > 0:
            import random as _r

            _r.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if str(device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = LoopDVRPoseTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=1,
            num_keypoints=num_keypoints,
            keypoint_dim=keypoint_dim,
            flip_idx=flip_idx,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
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
            loggers=loggers,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        self.model.to(self.device)
        return results

    def _load_weights(self, model_path: str):
        model_path = self._resolve_weights_path(model_path)
        path = Path(model_path)
        download_error = None
        if not path.exists():
            from ...utils.download import download_weights

            try:
                download_weights(model_path, self.size)
            except Exception as exc:  # noqa: BLE001
                download_error = exc
        if not path.exists():
            if download_error is not None:
                raise FileNotFoundError(
                    f"LoopDVR weights file not found: {model_path}\n"
                    f"Auto-download failed: {download_error}"
                ) from download_error
            raise FileNotFoundError(f"LoopDVR weights file not found: {model_path}")

        try:
            loaded = load_untrusted_torch_file(
                model_path, map_location="cpu", context="LoopDVR weights"
            )
            self._cache_checkpoint_train_config(loaded)
            state_dict = unwrap_loopdvr_checkpoint(loaded)
            state_dict = self._strip_ddp_prefix(dict(state_dict))

            ckpt_task = self.detect_task_from_state_dict(state_dict) or "detect"
            if ckpt_task != self.task:
                raise RuntimeError(
                    f"Checkpoint is a LoopDVR-{ckpt_task} model but this instance "
                    f"was initialized for task='{self.task}'. Pass the matching "
                    "task or use a different checkpoint."
                )

            if isinstance(loaded, dict):
                ckpt_family = loaded.get("model_family", "")
                own_family = self._get_model_name()
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'."
                    )
                ckpt_nc = loaded.get("nc")
                ckpt_names = loaded.get("names")
                if self.task == "pose":
                    effective_nc = int(ckpt_nc) if ckpt_nc is not None else 1
                    if effective_nc != 1:
                        raise RuntimeError(
                            "LoopDVR pose checkpoints must declare nc=1 "
                            f"(got nc={effective_nc})."
                        )
                    if ckpt_names is not None:
                        if len(ckpt_names) != 1:
                            raise RuntimeError(
                                "LoopDVR pose checkpoints must carry exactly one class "
                                f"name (got {len(ckpt_names)})."
                            )
                        self.names = self._sanitize_names(ckpt_names, 1)
                else:
                    if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                        self._rebuild_for_new_classes(int(ckpt_nc))
                    effective_nc = (
                        int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                    )
                    if ckpt_names is not None:
                        self.names = self._sanitize_names(ckpt_names, effective_nc)

            from ...utils.input_channels import restore_channels

            restore_channels(self, loaded)
            _missing, unexpected = self.model.load_state_dict(
                state_dict, strict=self._strict_loading()
            )
            if unexpected:
                raise RuntimeError(
                    "Unexpected keys when loading LoopDVR weights: "
                    f"{sorted(unexpected)[:10]}"
                    + (
                        f" (+{len(unexpected) - 10} more)"
                        if len(unexpected) > 10
                        else ""
                    )
                )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LoopDVR weights from {model_path}: {e}"
            ) from e
