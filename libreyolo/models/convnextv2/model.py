"""Native ConvNeXt V2 classification and official checkpoint recognition."""

import re
from pathlib import Path
from typing import Any, ClassVar

from ...training.callbacks import TrainCallbacks
from ...training.ddp_spawn import ddp_aware
from ..convnext.model import LibreConvNeXt
from .config import ConvNeXtV2Config
from .nn import ARCH_DEFS, ConvNeXtV2

_TRAIN_DEFAULTS = ConvNeXtV2Config()

_WEIGHT_KEYS = (
    "weight_license",
    "weight_license_url",
    "weight_commercial_use",
    "weight_dataset",
    "source",
    "source_commit",
    "source_sha256",
)


class LibreConvNeXtV2(LibreConvNeXt):
    """ConvNeXt V2 atto/femto/pico/n/t/b/l/h classifiers at 224 pixels.

    Official ImageNet-1K weights are CC-BY-NC-4.0; the code is MIT.
    Supervised fine-tuning uses LibreYOLO's shared classification recipe,
    not the upstream FCMAE pretraining or ImageNet reproduction recipe.
    """

    FAMILY = "convnextv2"
    FILENAME_PREFIX = "LibreConvNeXtV2"
    INPUT_SIZES: ClassVar[dict[str, int]] = {size: 224 for size in ARCH_DEFS}
    CROP_PCT: ClassVar[dict[str, float]] = {size: 0.875 for size in ARCH_DEFS}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    SUPPORTS_CUDA_GRAPH = False
    TRAIN_CONFIG = ConvNeXtV2Config

    def __init__(
        self,
        model_path=None,
        size="atto",
        nb_classes=1000,
        device="auto",
        task=None,
        **kwargs,
    ):
        self._weight_metadata = {}
        super().__init__(
            str(model_path) if model_path is not None else None,
            size,
            nb_classes,
            device,
            task,
            **kwargs,
        )

    @classmethod
    def can_load(cls, weights_dict):
        return (
            "downsample_layers.0.0.weight" in weights_dict
            and "head.weight" in weights_dict
            and "stages.0.0.grn.gamma" in weights_dict
            and "stages.0.0.grn.beta" in weights_dict
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict):
        stem = weights_dict.get("downsample_layers.0.0.weight")
        if stem is None or stem.ndim != 4:
            return None
        indices = [set() for _ in range(4)]
        for key in weights_dict:
            match = re.fullmatch(r"stages\.([0-3])\.(\d+)\.grn\.gamma", key)
            if match:
                indices[int(match[1])].add(int(match[2]))
        for size, (depths, dims) in ARCH_DEFS.items():
            if stem.shape[0] == dims[0] and all(
                blocks == set(range(depth)) for blocks, depth in zip(indices, depths)
            ):
                return size
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict):
        head = weights_dict.get("head.weight")
        return int(head.shape[0]) if head is not None else None

    @classmethod
    def get_download_notice(cls, filename, url):
        return (
            f"{Path(filename).name}: official ConvNeXt V2 weights are licensed "
            "CC-BY-NC-4.0 (NON-COMMERCIAL). The architecture code is MIT. "
            "See the Hugging Face repository LICENSE and NOTICE."
        )

    def _init_model(self):
        return ConvNeXtV2(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self):
        return {
            "stem": self.model.downsample_layers,
            "stages": self.model.stages,
            "head_norm": self.model.norm,
            "classifier": self.model.head,
        }

    def _prepare_model_for_state_dict(self, state_dict):
        pass

    def _prepare_scratch_init(self):
        super()._prepare_scratch_init()
        self._weight_metadata = {}

    def _validate_loaded_state_dict_for_task(self, state_dict, loaded):
        super()._validate_loaded_state_dict_for_task(state_dict, loaded)
        self._weight_metadata = {
            key: loaded[key] for key in _WEIGHT_KEYS if key in loaded
        }

    def _save_extra_metadata(self):
        return {
            **self._weight_metadata,
            "crop_pct": self.crop_pct,
            "interpolation": self.interpolation,
        }

    def _ddp_bootstrap_checkpoint(self, state):
        from ...utils.serialization import wrap_libreyolo_checkpoint

        return wrap_libreyolo_checkpoint(
            state,
            model_family=self.FAMILY,
            size=self.size,
            task=self.task,
            nc=self.nb_classes,
            names=self.names,
            imgsz=self.input_size,
            **self._save_extra_metadata(),
        )

    @classmethod
    def default_checkpoint_names(cls, nc, task=None):
        if nc == 1000:
            from ...data.imagenet import imagenet1k_names

            return imagenet1k_names()
        return None

    @classmethod
    def upstream_checkpoint_metadata(cls, loaded, *, source=None):
        from .utils import SOURCE_COMMIT, SOURCE_REPO, SOURCE_SHA256, checkpoint_sha256

        metadata = {key: loaded[key] for key in _WEIGHT_KEYS if key in loaded}
        metadata.update(crop_pct=0.875, interpolation="bicubic")
        if source is not None:
            digest = checkpoint_sha256(Path(source))
            if digest in SOURCE_SHA256.values():
                metadata.update(
                    source=SOURCE_REPO,
                    source_commit=SOURCE_COMMIT,
                    source_sha256=digest,
                    weight_license="cc-by-nc-4.0",
                    weight_license_url="https://creativecommons.org/licenses/by-nc/4.0/",
                    weight_commercial_use=False,
                    weight_dataset="ImageNet-1K",
                )
        return metadata

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
        callbacks: TrainCallbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Fine-tune the classifier on an ImageFolder-style dataset.

        ``data`` is a dataset root (``train/`` + ``val/`` folder-per-class), a
        known name (e.g. ``"imagenette160"``), or a ``.zip`` URL. The head is
        rebuilt to the dataset's class count automatically. Cross-entropy +
        AdamW + cosine; the ImageNet-pretrained backbone transfers cleanly.

        ``cls_pw`` (classification only, float in [0, 1], default 0) controls
        inverse-frequency weighting strength with mean-one class weights.
        ``class_weights=True`` retains legacy sample-normalized weighting and
        cannot be combined with ``cls_pw>0``. Neither option changes sampling.
        See docs/classification_training.md for compatibility and resume rules.
        """
        from .trainer import ConvNeXtV2Trainer

        if imgsz is None:
            imgsz = self.input_size

        trainer = ConvNeXtV2Trainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
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
            callbacks=callbacks,
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreConvNeXtV2('path/to/last.pt', size='t'); "
                    "model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))

        results = trainer.train()
        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self._load_weights(best_ckpt)
        return results
