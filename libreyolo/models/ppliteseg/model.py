"""LibreYOLO wrapper for PP-LiteSeg semantic segmentation.

PP-LiteSeg (arXiv:2204.02681) pairs an STDC backbone with a Simple Pyramid
Pooling Module and a Unified Attention Fusion Module decoder. This family ships
the four released Cityscapes 19-class checkpoints as
``LibrePPLiteSeg{t50,b50,t75,b75}-sem.pt``.

Sizes are ``t``/``b`` for the STDC1/STDC2 backbone and ``50``/``75`` for the
source validation scale against Cityscapes' native 1024x2048: ``50`` models
run on a 512x1024 canvas, ``75`` models on 768x1536. These are genuinely
rectangular models; nothing here collapses them to a square.

Licensing: the architecture, loss, and training recipe are adapted from the
Apache-2.0 SuperGradients implementation, with STDC lineage from the MIT
STDC-Seg repository (see ``NOTICE`` in this directory). The *released weights*
are a different surface: they are trained on Cityscapes, whose license allows
redistributing abstract derivative models but restricts the dataset and its
derivatives to NON-COMMERCIAL use. LibreYOLO itself and this code stay
permissively licensed; only these pretrained checkpoints are restricted. Train
from scratch, or fine-tune on your own data, for weights free of that term.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from ..base.model import BaseModel
from .nn import SIZE_CONFIGS, STRIDE, LibrePPLiteSegNet

logger = logging.getLogger(__name__)

CITYSCAPES_NAMES: Dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

CITYSCAPES_LICENSE_URL = "https://www.cityscapes-dataset.com/license/"
WEIGHT_LICENSE = "Cityscapes dataset terms, non-commercial"


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = (512, 1024),
) -> tuple[np.ndarray, float]:
    """Direct-resize an RGB image to the checkpoint canvas as CHW ``[0, 1]``.

    The source validation pipeline rescales the whole frame to the native
    rectangle without letterbox padding, so this stretches rather than fits.
    The returned ratio is always ``1.0``: there is no padded region to trim,
    and the logits are restored to the original canvas by a plain resize.

    ImageNet standardization is *not* applied here. It lives inside
    ``LibrePPLiteSegNet.forward`` on the raw ``[0, 1]`` tensor so that it
    happens exactly once and travels with every exported graph.
    """
    input_h, input_w = _input_size_hw(input_size)
    resized = cv2.resize(img_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32), 1.0


class LibrePPLiteSeg(BaseModel):
    """PP-LiteSeg t50/b50/t75/b75 for dense semantic segmentation."""

    FAMILY: ClassVar[str] = "ppliteseg"
    FILENAME_PREFIX: ClassVar[str] = "LibrePPLiteSeg"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, Tuple[int, int]]] = {
        size: config["imgsz"] for size, config in SIZE_CONFIGS.items()
    }

    # Source training recipe, read by BaseTrainer._setup_semantic_data.
    #  - rescale_crop: random absolute rescale of the source image, ignore-pad
    #    up to the crop, then a random crop -- the SegRandomRescale /
    #    SegPadShortToCropSize / SegCropImageAndMask chain of the recipe.
    #  - the scale range and the train crop differ per size, so both are
    #    resolved from SIZE_CONFIGS rather than pinned as class constants.
    semantic_resize_mode: ClassVar[str] = "rescale_crop"
    semantic_imgsz_divisor: ClassVar[int] = STRIDE
    # The recipe uses brightness/contrast/saturation jitter (magnitude 0.5),
    # not the HSV-gain jitter SemanticDataset defaults to. The family-local
    # transform is supplied through semantic_photometric below, so the shared
    # HSV path is switched off explicitly rather than left at its default.
    semantic_hsv_prob: ClassVar[float] = 0.0

    # ------------------------------------------------------------------
    # Registry / can_load interface
    # ------------------------------------------------------------------

    @classmethod
    def _strip_module_prefix(cls, weights_dict: dict) -> dict:
        if any(str(k).startswith("module.") for k in weights_dict):
            return {
                (str(k)[len("module.") :] if str(k).startswith("module.") else str(k)): v
                for k, v in weights_dict.items()
            }
        return weights_dict

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Require independent STDC + SPPM + UAFM + PP-LiteSeg head evidence.

        Every marker below is specific to this decoder: the SPPM branch convs,
        the 4->2->1 UAFM spatial-attention stack, and the three encoder
        projection convolutions. A generic key such as ``backbone`` would
        happily claim other families' checkpoints.
        """
        keys = set(cls._strip_module_prefix(weights_dict))
        return (
            "encoder.backbone.stages.block_s2.0.seq.conv.weight" in keys
            and "encoder.context_module.conv_out.seq.conv.weight" in keys
            and any(k.startswith("encoder.context_module.branches.") for k in keys)
            and "encoder.proj_convs.0.seq.conv.weight" in keys
            and "decoder.up_stages.0.conv_atten.0.seq.conv.weight" in keys
            and "seg_head.0.seg_head.2.weight" in keys
        )

    @classmethod
    def detect_backbone(cls, weights_dict: dict) -> Optional[str]:
        """Return ``"stdc1"`` / ``"stdc2"`` from decoder and projection widths.

        Deliberately independent of the filename: the two backbones differ in
        STDC block counts, in the first projection conv width (64 vs 96), and
        in the decoder stage widths.
        """
        state = cls._strip_module_prefix(weights_dict)
        proj = state.get("encoder.proj_convs.0.seq.conv.weight")
        if proj is None or getattr(proj, "ndim", 0) < 1:
            return None
        proj_ch = int(proj.shape[0])
        block_counts = {
            int(key.split(".")[4])
            for key in state
            if key.startswith("encoder.backbone.stages.block_s8.")
        }
        stage_s8_blocks = (max(block_counts) + 1) if block_counts else None
        if proj_ch == 64 and stage_s8_blocks == 2:
            return "stdc1"
        if proj_ch == 96 and stage_s8_blocks == 4:
            return "stdc2"
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Only the backbone is recoverable from tensors; 50 vs 75 is not.

        The two resolution recipes share architecture and class count, so the
        state dict cannot tell them apart. Returning the ``50`` variant here
        would silently mislabel a ``75`` checkpoint, so this reports ``None``
        and the size comes from checkpoint metadata or the canonical filename.
        """
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        """Read nc from the main head and require every aux head to agree."""
        state = cls._strip_module_prefix(weights_dict)
        head = state.get("seg_head.0.seg_head.2.weight")
        if head is None or getattr(head, "ndim", 0) < 1:
            return None
        nc = int(head.shape[0])
        for index in range(3):
            aux = state.get(f"aux_heads.{index}.0.seg_head.2.weight")
            if aux is not None and int(aux.shape[0]) != nc:
                raise RuntimeError(
                    "PP-LiteSeg checkpoint is inconsistent: main head predicts "
                    f"{nc} classes but aux head {index} predicts {int(aux.shape[0])}."
                )
        return nc

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        return (
            f"{Path(filename).name} is a converted PP-LiteSeg checkpoint trained on "
            "Cityscapes. The Cityscapes license restricts the dataset and its "
            "derivatives, including this checkpoint, to NON-COMMERCIAL use "
            f"({CITYSCAPES_LICENSE_URL}). The restriction applies to this "
            "pretrained checkpoint, not to LibreYOLO's MIT code or the "
            "PP-LiteSeg architecture. Train from scratch or fine-tune on your "
            "own data for weights without that term."
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "t50",
        nb_classes: int = 19,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibrePPLiteSeg supports only task='semantic'; got {task!r}.")
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        self.weight_license: Optional[str] = None
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)

    def _init_model(self) -> nn.Module:
        return LibrePPLiteSegNet(size=self.size, num_classes=self.nb_classes, use_aux_heads=True)

    # ------------------------------------------------------------------
    # Family recipe accessors (read by the trainer)
    # ------------------------------------------------------------------

    @property
    def semantic_scale_jitter(self) -> Tuple[float, float]:
        return tuple(SIZE_CONFIGS[self.size]["rescale_range"])

    @property
    def semantic_train_imgsz(self) -> Tuple[int, int]:
        """Source train crop: 512x1024 for the 50 sizes, 768x768 for the 75s."""
        return tuple(SIZE_CONFIGS[self.size]["train_crop"])

    @property
    def semantic_val_imgsz(self) -> Tuple[int, int]:
        """Validation runs on the native rectangle, not the train crop."""
        return tuple(SIZE_CONFIGS[self.size]["imgsz"])

    @property
    def semantic_photometric(self):
        from .transforms import SegColorJitter

        return SegColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)

    def _rebuild_for_new_size(self, new_size: str) -> None:
        if new_size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown PP-LiteSeg size {new_size!r}; expected one of {tuple(SIZE_CONFIGS)}"
            )
        self.size = new_size
        self.input_size = self.INPUT_SIZES[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        """Re-head the main and all three auxiliary classifiers together.

        Fine-tuning on another dataset must not leave the auxiliary heads at
        the old class count: they are still supervised, so a mismatch either
        crashes the loss or silently trains against stale logits.
        """
        self.model.replace_num_classes(int(new_nb_classes))
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(int(new_nb_classes))}
        self.model.to(self.device)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {
            "backbone": self.model.encoder.backbone,
            "context_module": self.model.encoder.context_module,
            "decoder": self.model.decoder,
            "seg_head": self.model.seg_head,
        }
        if hasattr(self.model, "aux_heads"):
            layers["aux_heads"] = self.model.aux_heads
        return layers

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | tuple[int, int] | None = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        input_h, input_w = _input_size_hw(effective_res)
        if input_h % STRIDE or input_w % STRIDE:
            raise ValueError(
                f"LibrePPLiteSeg imgsz={effective_res} must have both sides divisible "
                f"by {STRIDE} (encoder stride product)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), (input_h, input_w))
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
        ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Resize the main logits back to the original canvas, pre-argmax.

        Direct resize means there is no padded region to trim, so ``ratio`` is
        unused. Interpolating the logits and taking argmax afterwards (rather
        than nearest-resizing an argmax map) is what the source evaluation
        does, and it is what the exported backends reproduce.
        """
        logits = output
        if isinstance(logits, (tuple, list)):
            # Training forward returns (main, aux_s8, aux_s16, aux_s32).
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("predictions"))
        orig_w, orig_h = original_size
        return F.interpolate(
            logits.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

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
        logits = self._postprocess_semantic_logits(output, original_size, ratio=ratio, **kwargs)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_model(self, imgsz: int | tuple[int, int] | None = None) -> nn.Module:
        """Return an inference-only copy with fixed-shape SPPM pooling.

        Two things have to happen before tracing, and both on a *copy*: the
        auxiliary heads are training-only and must not become graph outputs,
        and SPPM's adaptive pooling has no ONNX lowering, so it is rewritten to
        fixed-kernel pooling derived from the rectangle actually being
        exported. A module prepared for 512x1024 is invalid for 768x1536,
        which is exactly why the live model is never mutated here.
        """
        target = imgsz if imgsz is not None else self._get_input_size()
        input_h, input_w = _input_size_hw(target)
        export_model = copy.deepcopy(self.model).eval()
        export_model.remove_aux_heads()
        export_model.encoder.context_module.prep_model_for_conversion((input_h, input_w))
        return export_model

    def export(self, format: str = "onnx", **kwargs) -> str:
        original_model = self.model
        self.model = self._export_model(kwargs.get("imgsz"))
        try:
            return super().export(format=format, **kwargs)
        finally:
            self.model = original_model

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _strict_loading(self) -> bool:
        return True

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError("Checkpoint does not look like a PP-LiteSeg semantic model.")

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_trusted_torch_file(
                model_path, map_location="cpu", context="PP-LiteSeg semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibrePPLiteSeg checkpoints must be dictionaries")

        ckpt_family = loaded.get("model_family")
        if isinstance(ckpt_family, str) and ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but LibrePPLiteSeg is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded
        state = self._strip_module_prefix(state)

        ckpt_size = loaded.get("size")
        if ckpt_size is not None and str(ckpt_size) != self.size:
            self._rebuild_for_new_size(str(ckpt_size))
        backbone = self.detect_backbone(state)
        expected = SIZE_CONFIGS[self.size]["backbone"]
        if backbone is not None and backbone != expected:
            raise RuntimeError(
                f"Checkpoint holds a {backbone.upper()} backbone but size={self.size!r} "
                f"expects {expected.upper()}. Pass the matching size, or use the "
                "canonical filename so the size is resolved for you."
            )

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError("Checkpoint does not look like a PP-LiteSeg semantic model.")
        # Auxiliary heads are part of the trainable checkpoint, so a released
        # artifact loads strictly. An inference-only artifact that dropped them
        # would fail here rather than load a silently half-initialized model.
        self.model.load_state_dict(state, strict=True)

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)
        self.weight_license = loaded.get("weight_license")
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 800,
        batch: int = 8,
        imgsz: Optional[int | Tuple[int, int]] = None,
        lr0: Optional[float] = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: Optional[str] = None,
        name: Optional[str] = None,
        exist_ok: bool = False,
        resume: bool | str | Path = False,
        amp: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> Dict:
        """Train PP-LiteSeg with the source recipe.

        ``imgsz`` defaults to the size's *train crop* (512x1024 for t50/b50,
        768x768 for t75/b75), not its validation canvas: the 75 recipe trains
        on a square crop and validates on a rectangle on purpose. Validation
        always runs at the native rectangle from ``semantic_val_imgsz``.

        The released recipe keeps mixed precision off; ``amp`` defaults to
        ``False`` to match it rather than silently changing the recipe.

        ``resume=True`` resumes the loaded checkpoint and, when that is a run's
        ``weights/*.pt``, keeps writing into that run; a path resumes from that
        file. ``project``/``name`` default to ``runs/train/ppliteseg_exp``.
        """
        from .trainer import PPLiteSegTrainer

        resume_path = None
        if resume:
            if resume is True:
                run_checkpoint = self._loaded_run_checkpoint()
                if run_checkpoint is not None and project is None and name is None:
                    project = str(run_checkpoint.parent.parent.parent)
                    name = run_checkpoint.parent.parent.name
                    # Never let _get_save_dir() increment away from the run.
                    exist_ok = True
                resume_path = self.model_path
            else:
                resume_path = resume
            if not resume_path:
                raise ValueError(
                    "resume=True requires a checkpoint; load last.pt before training."
                )
        if project is None:
            project = "runs/train"
        if name is None:
            name = "ppliteseg_exp"

        train_imgsz = imgsz if imgsz is not None else self.semantic_train_imgsz
        train_h, train_w = _input_size_hw(train_imgsz)
        if train_h % STRIDE or train_w % STRIDE:
            raise ValueError(
                f"PP-LiteSeg training imgsz={train_imgsz!r} must have both sides "
                f"divisible by {STRIDE}."
            )

        train_kwargs = dict(
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=(train_h, train_w),
            size=self.size,
            num_classes=self.nb_classes,
            device=device,
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=bool(resume_path),
            amp=amp,
            **kwargs,
        )
        if lr0 is not None:
            train_kwargs["lr0"] = lr0

        trainer = PPLiteSegTrainer(
            model=self.model,
            wrapper_model=self,
            callbacks=callbacks,
            loggers=loggers,
            **train_kwargs,
        )
        if resume_path:
            trainer.setup()
            trainer.resume(str(resume_path))
        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break
        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)
        self.model.to(self.device)
        self.model.eval()


__all__ = [
    "CITYSCAPES_LICENSE_URL",
    "CITYSCAPES_NAMES",
    "WEIGHT_LICENSE",
    "LibrePPLiteSeg",
    "preprocess_numpy",
]
