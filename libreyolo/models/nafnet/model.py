"""LibreNAFNet image-restoration model family."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.nafnet import postprocess as _nafnet_postprocess
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput
from ...utils.serialization import load_untrusted_torch_file
from ..base import BaseModel
from .config import NAFNetConfig
from .nn import NAFNetLocal
from .utils import preprocess_image, preprocess_numpy

logger = logging.getLogger(__name__)


NAFNET_SIZE_CONFIGS: dict[str, dict[str, object]] = {
    "s": {
        "width": 32,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
    "l": {
        "width": 64,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1, 28],
        "dec_blk_nums": [1, 1, 1, 1],
    },
}
_TRAIN_DEFAULTS = NAFNetConfig()


def infer_nafnet_config(state_dict: dict) -> Optional[dict]:
    """Infer a NAFNet architecture config (width + block layout) from a state dict.

    Different NAFNet checkpoints share the same tensor names but use different
    block layouts (the GoPro deblur models use ``enc=[1,1,1,28], middle=1`` while
    the SIDD denoise models use ``enc=[2,2,4,8], middle=12``). Counting the
    per-stage block indices lets one family class load any of them.
    """

    intro = state_dict.get("intro.weight")
    if intro is None or getattr(intro, "ndim", 0) != 4:
        return None

    def _stage_counts(prefix: str) -> list[int]:
        stages: dict[int, set[int]] = {}
        marker = ".beta"
        plen = len(prefix) + 1
        for key in state_dict:
            if not key.startswith(prefix + ".") or not key.endswith(marker):
                continue
            rest = key[plen:-len(marker)]
            parts = rest.split(".")
            if len(parts) != 2:
                continue
            try:
                stage, block = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            stages.setdefault(stage, set()).add(block)
        return [len(stages[i]) for i in sorted(stages)]

    middle = {
        int(k[len("middle_blks.") :].split(".")[0])
        for k in state_dict
        if k.startswith("middle_blks.") and k.endswith(".beta")
    }
    enc = _stage_counts("encoders")
    dec = _stage_counts("decoders")
    if not enc or not dec or not middle:
        return None
    return {
        "width": int(intro.shape[0]),
        "middle_blk_num": len(middle),
        "enc_blk_nums": enc,
        "dec_blk_nums": dec,
    }


class LibreNAFNet(BaseModel):
    """NAFNet RGB image restoration.

    Predict runs at native image resolution, pads to the network downsample
    factor, and returns ``Results.restored`` on the original canvas.
    """

    FAMILY = "nafnet"
    FILENAME_PREFIX = "LibreNAFNet"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 256, "l": 256}
    SUPPORTED_TASKS = ("restore",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = NAFNetConfig
    RESUME_RESTORES_TRAIN_ARGS = True
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False
    # Dataset/degradation weight variants (e.g. SIDD denoise vs the default GoPro
    # deblur weights) get a filename + HF-repo suffix such as
    # ``LibreNAFNetl-restore-sidd.pt``.
    WEIGHT_VARIANTS: ClassVar[Tuple[str, ...]] = ("sidd",)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "intro.weight" in weights_dict
            and "ending.weight" in weights_dict
            and any(k.startswith("encoders.") for k in weights_dict)
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        intro = weights_dict.get("intro.weight")
        if intro is None or getattr(intro, "ndim", 0) != 4:
            return None
        return {32: "s", 64: "l"}.get(int(intro.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        # Peek the checkpoint to infer the block layout before the model is
        # built. GoPro and SIDD NAFNet checkpoints share tensor names but use
        # different block counts, so a fixed size config cannot load both.
        self._arch_config: Optional[dict] = self._peek_arch_config(model_path, size)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}

    @classmethod
    def _peek_arch_config(cls, model_path, size) -> Optional[dict]:
        """Best-effort read of the block config from a checkpoint before build."""

        try:
            if isinstance(model_path, dict):
                state_dict = model_path.get("model", model_path)
            elif isinstance(model_path, (str, Path)):
                path = Path(cls._resolve_weights_path(str(model_path)))
                if not path.exists():
                    # Resolve auto-download so the block layout is known before
                    # the architecture is built; _load_weights reuses the cache.
                    from ...utils.download import download_weights

                    download_weights(str(model_path), size)
                    path = Path(cls._resolve_weights_path(str(model_path)))
                if not path.exists():
                    return None
                # Safe peek: weights_only=True forbids pickle code execution.
                # This only needs the tensor state dict to infer the block
                # layout; anything unloadable falls back to the default config.
                loaded = load_untrusted_torch_file(
                    str(path), map_location="cpu", context="NAFNet arch peek"
                )
                state_dict = loaded.get("model", loaded) if isinstance(loaded, dict) else None
            else:
                return None
            if not isinstance(state_dict, dict):
                return None
            return infer_nafnet_config(state_dict)
        except Exception as exc:
            # Best-effort pre-pass: any real problem resurfaces in
            # _load_weights with a proper error, but keep the root cause
            # visible instead of silently falling back to the default layout.
            logger.warning(
                "Could not infer NAFNet block config from %s (%s); "
                "falling back to the default size config.",
                model_path,
                exc,
            )
            return None

    def _init_model(self) -> nn.Module:
        cfg = self._arch_config or NAFNET_SIZE_CONFIGS[self.size]
        return NAFNetLocal(
            img_channel=3,
            width=int(cfg["width"]),
            middle_blk_num=int(cfg["middle_blk_num"]),
            enc_blk_nums=cfg["enc_blk_nums"],
            dec_blk_nums=cfg["dec_blk_nums"],
            train_size=(1, 3, self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "intro": self.model.intro,
            "encoders": self.model.encoders,
            "middle_blks": self.model.middle_blks,
            "decoders": self.model.decoders,
            "ending": self.model.ending,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        del input_size
        return preprocess_image(
            image,
            pad_multiple=int(getattr(self.model, "padder_size", 16)),
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

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
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return {"restored": _nafnet_postprocess(output, original_size)}

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = _TRAIN_DEFAULTS.epochs,
        batch: int = _TRAIN_DEFAULTS.batch,
        imgsz: int | None = None,
        lr0: float = _TRAIN_DEFAULTS.lr0,
        optimizer: str = _TRAIN_DEFAULTS.optimizer,
        device: str = "",
        workers: int = _TRAIN_DEFAULTS.workers,
        seed: int = _TRAIN_DEFAULTS.seed,
        project: str = _TRAIN_DEFAULTS.project,
        name: str = _TRAIN_DEFAULTS.name,
        exist_ok: bool = _TRAIN_DEFAULTS.exist_ok,
        resume: bool = _TRAIN_DEFAULTS.resume,
        amp: bool = _TRAIN_DEFAULTS.amp,
        patience: int = _TRAIN_DEFAULTS.patience,
        degradation: str | None = None,
        dataset: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune NAFNet on paired degraded/target RGB images."""

        from .trainer import NAFNetTrainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = NAFNetTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=1,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            degradation=degradation,
            dataset=dataset,
            **kwargs,
        )

        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results


__all__ = ["LibreNAFNet", "NAFNET_SIZE_CONFIGS"]
