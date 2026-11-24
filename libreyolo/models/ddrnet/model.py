"""LibreYOLO wrapper for DDRNet semantic segmentation.

DDRNet (Hong et al., arXiv:2101.06085) is a real-time two-branch CNN: a
high-resolution branch at stride 8 and a low-resolution context branch exchange
features through repeated bilateral fusion, and a Deep Aggregation Pyramid
Pooling Module (DAPPM) closes the context branch. Sizes ``s`` / ``m`` / ``l``
are upstream's DDRNet-23-slim / DDRNet-23 / DDRNet-39.

Inference runs whole frames at the upstream evaluation canvas (1024x2048) and
``weights/parity_ddrnet.py`` proves the graph bit-identical to the official
implementation. ``LibreDDRNets-sem.pt`` and ``LibreDDRNetm-sem.pt`` are the
official Cityscapes checkpoints; upstream released no Cityscapes DDRNet-39
checkpoint, so ``l`` is code-ready and trains from the official ImageNet
backbone.

Licensing: the code is ported from the MIT-licensed official repository
(see ``NOTICE``). The official checkpoints are linked from that repository with
no separate terms, so LibreYOLO hosts them under MIT on that implied basis; the
Cityscapes dataset they were trained on has its own terms and is not
redistributed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn

from ...postprocess.ddrnet import postprocess as _postprocess
from ...postprocess.ddrnet import resize_logits
from ...tasks import normalize_task
from ...training.callbacks import TrainCallbacks
from ...training.config import DDRNetConfig
from ...training.ddp_spawn import ddp_aware
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.input_channels import restore_channels
from ...utils.serialization import (
    load_untrusted_torch_file,
    wrap_libreyolo_checkpoint,
)
from ..base.model import BaseModel
from .convert import (
    IMAGENET_BACKBONES,
    UPSTREAM_LICENSE_URL,
    UPSTREAM_REPO,
    convert_upstream_ddrnet_state_dict,
    imagenet_backbone_state,
    is_ddrnet_state_dict,
    official_cityscapes_size,
    sha256_file,
    strip_wrapper_prefixes,
)
from .nn import ARCH_CONFIGS, SIZE_CONFIGS, STRIDE, LibreDDRNetNet

logger = logging.getLogger(__name__)

CITYSCAPES_NAMES: dict[int, str] = {
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

WEIGHT_LICENSE = "MIT"
WEIGHT_LICENSE_URL = UPSTREAM_LICENSE_URL
_WEIGHT_METADATA_KEYS = (
    "weight_license",
    "weight_license_url",
    "weight_dataset",
    "weight_commercial_use",
)


def _input_size_hw(input_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(f"input_size must be int or (height, width), got {input_size!r}")
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = (1024, 2048),
) -> tuple[np.ndarray, float]:
    """Direct-resize an RGB image to the evaluation canvas as CHW ``[0, 1]``.

    The upstream Cityscapes evaluation feeds whole 1024x2048 frames with no
    padding, so a native frame passes through unchanged. ImageNet
    standardization lives inside ``LibreDDRNetNet``.
    """
    import cv2

    input_h, input_w = _input_size_hw(input_size)
    if img_rgb_hwc.shape[:2] == (input_h, input_w):
        resized = img_rgb_hwc
    else:
        resized = cv2.resize(img_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    arr = np.ascontiguousarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32), 1.0


class LibreDDRNet(BaseModel):
    """DDRNet-23-slim / DDRNet-23 / DDRNet-39 for dense semantic segmentation."""

    FAMILY: ClassVar[str] = "ddrnet"
    FILENAME_PREFIX: ClassVar[str] = "LibreDDRNet"
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {
        size: config["imgsz"] for size, config in SIZE_CONFIGS.items()
    }
    TRAIN_CONFIG: ClassVar[type[DDRNetConfig]] = DDRNetConfig
    RESUME_RESTORES_TRAIN_ARGS: ClassVar[bool] = True
    UNPUBLISHED_WEIGHTS: ClassVar[dict[str, str]] = {
        "LibreDDRNetl-sem": (
            "Upstream released no Cityscapes DDRNet-39 checkpoint. Build "
            "LibreDDRNet(size='l') and train it; a fresh model starts from the "
            "official ImageNet backbone."
        ),
    }

    # Training samples the paper's Cityscapes recipe: rescale the source frame
    # by a factor in ``rescale_range``, then random-crop ``train_crop``, with a
    # horizontal flip. Validation and inference direct-resize the whole frame
    # to ``imgsz`` (identity on Cityscapes' 1024x2048).
    semantic_resize_mode: ClassVar[str] = "rescale_crop"
    semantic_imgsz_divisor: ClassVar[int] = STRIDE

    # ------------------------------------------------------------------
    # Registry / can_load interface
    # ------------------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return is_ddrnet_state_dict(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        state = strip_wrapper_prefixes(weights_dict)
        stem = state.get("conv1.0.weight")
        head = state.get("final_layer.conv1.weight")
        if stem is None or head is None:
            return None
        planes, head_planes = int(stem.shape[0]), int(head.shape[0])
        split = "layer3_1.0.conv1.weight" in state
        for size, cfg in ARCH_CONFIGS.items():
            if (cfg.planes, cfg.head_planes, cfg.split_stage3) == (planes, head_planes, split):
                return size
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        state = strip_wrapper_prefixes(weights_dict)
        head = state.get("final_layer.conv2.weight")
        if head is None or getattr(head, "ndim", 0) < 1:
            return None
        nc = int(head.shape[0])
        aux = state.get("seghead_extra.conv2.weight")
        if aux is not None and int(aux.shape[0]) != nc:
            raise RuntimeError(
                "DDRNet checkpoint is inconsistent: main head predicts "
                f"{nc} classes but auxiliary head predicts {int(aux.shape[0])}."
            )
        return nc

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> dict | None:
        return convert_upstream_ddrnet_state_dict(state_dict)

    @classmethod
    def upstream_checkpoint_metadata(cls, loaded: dict, *, source: Path | None = None) -> dict:
        """Stamp the official artifacts' terms; identify them by SHA-256.

        Architecture or class count alone cannot identify a weight license:
        users can train this graph from scratch on their own data.
        """
        height, width = SIZE_CONFIGS["s"]["imgsz"]
        metadata = {
            "imgsz": max(height, width),
            "imgsz_h": height,
            "imgsz_w": width,
            **{
                key: loaded[key]
                for key in _WEIGHT_METADATA_KEYS
                if isinstance(loaded, dict) and loaded.get(key) is not None
            },
        }
        if source is not None and Path(source).is_file():
            size = official_cityscapes_size(source)
            if size is not None:
                metadata.update(
                    names=dict(CITYSCAPES_NAMES),
                    weight_license=WEIGHT_LICENSE,
                    weight_license_url=WEIGHT_LICENSE_URL,
                    weight_dataset="Cityscapes",
                    source_sha256=sha256_file(source),
                )
        return metadata

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 19,
        device: str = "auto",
        task: str | None = None,
        pretrained_backbone: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Build a DDRNet, from a checkpoint or fresh.

        ``pretrained_backbone`` is a local copy of the official ImageNet
        classifier for this size (``DDRNet23s_imagenet.pth``,
        ``DDRNet23_imagenet.pth`` or ``DDRNet39_imagenet.pth``); it initializes
        every tensor the two networks share. Without it, a fresh model fetches
        the LibreYOLO mirror of that file when ``train()`` starts.
        """
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(f"LibreDDRNet supports only task='semantic'; got {task!r}.")
        if model_path is not None and pretrained_backbone is not None:
            raise ValueError(
                "Pass either model_path= (a full DDRNet checkpoint) or "
                "pretrained_backbone= (the ImageNet classifier), not both."
            )
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        self.weight_license: str | None = None
        self.weight_license_url: str | None = None
        self.weight_dataset: str | None = None
        self.weight_commercial_use: bool | None = None
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)
        if pretrained_backbone is not None:
            self.load_imagenet_backbone(pretrained_backbone)

    def _init_model(self) -> nn.Module:
        # A fresh build that is neither loading weights, re-heading, nor a
        # scratch-training reset still owes the upstream ImageNet backbone.
        # It is fetched when training starts, not here, so constructing a
        # model never touches the network.
        self._imagenet_init_pending = not (
            getattr(self, "_loading_from_weights", False)
            or getattr(self, "_in_rebuild", False)
            or self._is_scratch_build()
        )
        return LibreDDRNetNet(size=self.size, num_classes=self.nb_classes)

    @property
    def semantic_scale_jitter(self) -> tuple[float, float]:
        return tuple(SIZE_CONFIGS[self.size]["rescale_range"])

    @property
    def semantic_train_imgsz(self) -> tuple[int, int]:
        """Source train crop (1024x1024)."""
        return tuple(SIZE_CONFIGS[self.size]["train_crop"])

    @property
    def semantic_val_imgsz(self) -> tuple[int, int]:
        """Whole-frame evaluation canvas (1024x2048), not the train crop."""
        return tuple(SIZE_CONFIGS[self.size]["imgsz"])

    def _prepare_scratch_init(self) -> None:
        for key in _WEIGHT_METADATA_KEYS:
            setattr(self, key, None)

    def _checkpoint_metadata(self) -> dict[str, Any]:
        height, width = self.semantic_val_imgsz
        return {
            "imgsz": max(height, width),
            "imgsz_h": height,
            "imgsz_w": width,
            **{
                key: getattr(self, key)
                for key in _WEIGHT_METADATA_KEYS
                if getattr(self, key, None) is not None
            },
        }

    def _save_extra_metadata(self) -> dict[str, Any]:
        """Weight terms ``save()`` writes, so a saved copy keeps its source terms."""
        return {
            key: getattr(self, key)
            for key in _WEIGHT_METADATA_KEYS
            if getattr(self, key, None) is not None
        }

    def _ddp_bootstrap_checkpoint(self, state_dict: dict) -> dict:
        return wrap_libreyolo_checkpoint(
            state_dict,
            model_family=self.FAMILY,
            size=self.size,
            task=self.task,
            nc=self.nb_classes,
            names=self.names,
            **self._checkpoint_metadata(),
        )

    def _rebuild_for_new_size(self, new_size: str) -> None:
        if new_size not in SIZE_CONFIGS:
            raise ValueError(
                f"Unknown DDRNet size {new_size!r}; expected one of {tuple(SIZE_CONFIGS)}"
            )
        self.size = new_size
        self.input_size = self.INPUT_SIZES[new_size]
        self.model = self._init_model()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        """Re-head both classifiers; the backbone (and any ImageNet init) stays."""
        self.model.replace_num_classes(int(new_nb_classes))
        self.nb_classes = int(new_nb_classes)
        self.names = {i: f"class_{i}" for i in range(int(new_nb_classes))}
        self.model.to(self.device)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        net = self.model
        if net.split_stage3:
            stage3 = [net.layer3_1, net.layer3_2]
            highres3 = [net.layer3_1_, net.layer3_2_]
        else:
            stage3 = [net.layer3]
            highres3 = [net.layer3_]
        return {
            "stem": net.conv1,
            "low_res_branch": nn.ModuleList(
                [net.layer1, net.layer2, *stage3, net.layer4, net.layer5]
            ),
            "high_res_branch": nn.ModuleList([*highres3, net.layer4_, net.layer5_]),
            "dappm": net.spp,
            "head": net.final_layer,
            "aux_head": net.seghead_extra,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        input_h, input_w = _input_size_hw(effective_res)
        if input_h % STRIDE or input_w % STRIDE:
            raise ValueError(
                f"LibreDDRNet imgsz={effective_res} must have both sides divisible "
                f"by {STRIDE} (high-resolution branch stride)."
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
        original_size: tuple[int, int],
        ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del ratio, kwargs
        return resize_logits(output, original_size)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        return _postprocess(
            output,
            conf_thres,
            iou_thres,
            original_size,
            max_det=max_det,
            **kwargs,
        )

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
            raise RuntimeError("Checkpoint does not look like a DDRNet semantic model.")

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_untrusted_torch_file(
                model_path, map_location="cpu", context="DDRNet semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibreDDRNet checkpoints must be dictionaries")

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
                "but LibreDDRNet is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            state = loaded["state_dict"]
        else:
            state = loaded
        if "model_family" not in loaded:
            # A raw official checkpoint: rename its keys and stamp its terms.
            converted = convert_upstream_ddrnet_state_dict(state)
            if converted is not None:
                state = converted
                source = Path(model_path) if isinstance(model_path, str) else None
                loaded = {**loaded, **self.upstream_checkpoint_metadata(loaded, source=source)}
        state = strip_wrapper_prefixes(state)

        ckpt_size = loaded.get("size") or self.detect_size(state)
        if ckpt_size is not None and str(ckpt_size) != self.size:
            self._rebuild_for_new_size(str(ckpt_size))

        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        if not self.can_load(state):
            raise RuntimeError("Checkpoint does not look like a DDRNet semantic model.")
        # A grayscale checkpoint carries a one-channel stem; rebuild it first.
        restore_channels(self, loaded)
        self.model.load_state_dict(state, strict=True)
        self._imagenet_init_pending = False

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        elif self.nb_classes == len(CITYSCAPES_NAMES):
            self.names = dict(CITYSCAPES_NAMES)
        self.weight_license = loaded.get("weight_license")
        self.weight_license_url = loaded.get("weight_license_url")
        self.weight_dataset = loaded.get("weight_dataset")
        commercial = loaded.get("weight_commercial_use")
        self.weight_commercial_use = None if commercial is None else bool(commercial)
        self._restore_checkpoint_input_size(loaded)
        self._cache_checkpoint_train_config(loaded)
        if isinstance(model_path, str):
            self.model_path = model_path
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # ImageNet backbone initialization
    # ------------------------------------------------------------------

    @classmethod
    def imagenet_backbone_path(cls, size: str, weights_dir: str | Path = "weights") -> Path:
        """Local path of the official ImageNet backbone, downloading it if absent.

        The file is a byte-for-byte mirror of the upstream release, checked
        against its pinned SHA-256 before it is published at the final path.
        """
        from ...utils.download import download_url_to_path
        from ...utils.hf_hub import hf_url

        spec = IMAGENET_BACKBONES[size]
        path = Path(weights_dir) / spec["filename"]
        if not path.exists():
            url = hf_url(f"{spec['repo']}/resolve/main/{spec['filename']}")

            def _verify(local_path, source_url):
                digest = sha256_file(local_path)
                if digest != spec["sha256"]:
                    raise RuntimeError(
                        f"{spec['filename']} from {source_url} has SHA-256 {digest}, "
                        f"expected {spec['sha256']}."
                    )

            download_url_to_path(url, path, verify=_verify)
        return path

    def load_imagenet_backbone(self, path: str | Path | None = None) -> None:
        """Initialize the shared tensors from the official ImageNet classifier.

        DAPPM and both segmentation heads keep their random initialization, as
        upstream trains them from scratch.
        """
        source = Path(path) if path is not None else self.imagenet_backbone_path(self.size)
        if path is not None:
            digest = sha256_file(source)
            if digest != IMAGENET_BACKBONES[self.size]["sha256"]:
                logger.warning(
                    "%s is not the official DDRNet ImageNet backbone for size %r "
                    "(SHA-256 %s); loading it anyway.",
                    source,
                    self.size,
                    digest,
                )
        loaded = load_untrusted_torch_file(
            str(source), map_location="cpu", context="DDRNet ImageNet backbone"
        )
        if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError(f"{source} does not hold a state dict.")
        target = self.model.state_dict()
        state = imagenet_backbone_state(loaded, set(target))
        stem = "conv1.0.weight"
        if stem in state and state[stem].shape[1] != target[stem].shape[1]:
            # A grayscale dataset already gave the stem one input channel; sum
            # the RGB filters, as the shared channel adapter does.
            if target[stem].shape[1] != 1 or state[stem].shape[1] != 3:
                raise RuntimeError(
                    f"Cannot map the ImageNet stem {tuple(state[stem].shape)} onto "
                    f"{tuple(target[stem].shape)}."
                )
            state[stem] = state[stem].sum(dim=1, keepdim=True)
        self.model.load_state_dict(state, strict=False)
        self._imagenet_init_pending = False
        logger.info(
            "Initialized LibreDDRNet-%s from the ImageNet backbone %s (%d tensors).",
            self.size,
            source.name,
            len(state),
        )

    def _maybe_init_imagenet_backbone(self) -> None:
        if getattr(self, "_imagenet_init_pending", False):
            self.load_imagenet_backbone()

    def _ddp_prepare_train_kwargs(self, train_kw: dict) -> dict:
        # DDP workers rebuild from this process's weights, so the backbone
        # must be in place before they are spawned.
        self._maybe_init_imagenet_backbone()
        return train_kw

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @ddp_aware()
    def train(
        self,
        data: str,
        *,
        epochs: int = 484,
        batch: int = 12,
        imgsz: int | tuple[int, int] | None = None,
        lr0: float | None = None,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "ddrnet_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = False,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Train DDRNet with the paper's OHEM + auxiliary-head Cityscapes recipe.

        A model built without a checkpoint first loads the official ImageNet
        backbone (downloaded once into ``weights/``), as the upstream recipe
        does; ``pretrained=False`` trains from random initialization instead.
        """
        from .trainer import DDRNetTrainer

        if not resume:
            self._maybe_init_imagenet_backbone()

        train_imgsz = imgsz if imgsz is not None else self.semantic_train_imgsz
        train_h, train_w = _input_size_hw(train_imgsz)
        if train_h % STRIDE or train_w % STRIDE:
            raise ValueError(
                f"DDRNet training imgsz={train_imgsz!r} must have both sides "
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
            resume=resume,
            amp=amp,
            **kwargs,
        )
        if lr0 is not None:
            train_kwargs["lr0"] = lr0

        trainer = DDRNetTrainer(
            model=self.model,
            wrapper_model=self,
            callbacks=callbacks,
            loggers=loggers,
            **train_kwargs,
        )
        if resume:
            trainer.setup()
            trainer.resume(self._resume_checkpoint(resume))
        result = trainer.train()
        self._restore_after_training(result)
        return result

    def _restore_after_training(self, result: dict) -> None:
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path:
                checkpoint = path
                break
        if checkpoint:
            self._load_weights(str(checkpoint))


__all__ = [
    "CITYSCAPES_NAMES",
    "UPSTREAM_REPO",
    "WEIGHT_LICENSE",
    "LibreDDRNet",
    "preprocess_numpy",
]
