"""Marigold V2 depth, normals and albedo through the LibreYOLO model factory."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from ...tasks import normalize_task
from ...utils.image_loader import ImageLoader
from ...utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)
from ..base.model import BaseModel
from .config import FILENAMES, VARIANTS, canonical_filename, upstream_url
from .convert import PROMPT_KEYS, VARIANT_KEY, convert_checkpoint, variant_from_state
from .nn import BASE_REPO, BASE_REVISION, MarigoldV2Net, build_components
from .utils import output_map, preprocess_numpy, resize_output


class LibreMarigoldV2(BaseModel):
    """One-step diffusion-transformer dense prediction with a frozen Qwen base.

    ``imgsz=0`` (default prediction) keeps native resolution, rounded up to
    multiples of 16. Positive ``imgsz`` stretches to a fixed square canvas.
    Depth preserves the checkpoint's affine-relative numeric encoding.
    The default NF4 runtime needs CUDA; unquantized CPU execution is explicit
    and requires enough memory for the 20B base. Checkpoints store adapters,
    optional decoder weights and prompt tensors; the base downloads separately.
    """

    FAMILY = "marigold_v2"
    FILENAME_PREFIX = "LibreMarigoldV2"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"b": 1024}
    SUPPORTED_TASKS = ("depth", "normal", "albedo")
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    WEIGHT_VARIANTS = tuple(
        key
        for key, value in VARIANTS.items()
        if value.task == "depth" and key != "log-stage2"
    )
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False
    TRAIN_CONFIG = None
    depth_imgsz_divisor = 16
    normal_imgsz_divisor = 16
    depth_resize_mode = "stretch"
    normal_resize_mode = "stretch"

    @classmethod
    def can_load(cls, state):
        return (
            variant_from_state(state) is not None
            and all(key in state for key in PROMPT_KEYS)
            and "Diffuser.transformer_blocks.0.attn.to_q.lora_A.default.weight" in state
        )

    @classmethod
    def detect_size(cls, state):
        return "b" if cls.can_load(state) else None

    @classmethod
    def detect_nb_classes(cls, state):
        return 1 if cls.can_load(state) else None

    @classmethod
    def detect_checkpoint_task(cls, state):
        variant = variant_from_state(state)
        return VARIANTS[variant].task if variant is not None else None

    @classmethod
    def get_download_url(cls, filename):
        variant = FILENAMES.get(Path(filename).name.lower())
        return upstream_url(variant) if variant else None

    @classmethod
    def verify_downloaded_file(cls, local_path, source_url):
        variant = next(
            (name for name in VARIANTS if upstream_url(name) == source_url), None
        )
        if variant is None:
            raise ValueError("Unrecognized Marigold V2 checkpoint URL.")
        # Downloaded safetensors are wrapped atomically at the canonical .pt path.
        convert_checkpoint(local_path, local_path, variant=variant)

    def __init__(
        self,
        model_path=None,
        size="b",
        nb_classes=1,
        device="auto",
        task=None,
        *,
        variant=None,
        base_path=None,
        quantization="4bit",
        seed=2025,
        **kwargs,
    ):
        # The generic metadata factory passes its legacy nc=80 constructor
        # default; dense checkpoints still enforce nc=1 below.
        if size != "b" or nb_classes not in (1, 80):
            raise ValueError("Marigold V2 has size='b' and nc=1 for its dense tasks.")
        if kwargs:
            raise TypeError(
                f"Unsupported Marigold V2 option(s): {', '.join(sorted(kwargs))}"
            )
        if task is not None:
            task = normalize_task(task)
        if variant is not None and variant not in VARIANTS:
            raise ValueError(f"Unknown Marigold V2 variant: {variant!r}")
        if quantization not in {"4bit", "none"}:
            raise ValueError("Marigold V2 quantization must be '4bit' or 'none'.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        if quantization == "4bit" and torch.device(device).type != "cuda":
            raise ValueError(
                "Marigold V2's default NF4 model requires CUDA. Use a CUDA host."
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if model_path is None:
            selected = variant or ("log-stage2" if task in (None, "depth") else task)
            if selected not in VARIANTS:
                raise ValueError(f"Unsupported Marigold V2 task: {task!r}")
            model_path = canonical_filename(selected)
        path = None
        if isinstance(model_path, dict):
            packet = model_path
            if "model" not in packet:
                selected = variant_from_state(packet)
                if selected is None:
                    raise ValueError(
                        "Raw Marigold state dictionaries require a variant marker."
                    )
                packet = wrap_libreyolo_checkpoint(
                    packet,
                    model_family=self.FAMILY,
                    size="b",
                    task=VARIANTS[selected].task,
                    nc=1,
                    names={0: VARIANTS[selected].task},
                    imgsz=1024,
                    variant=selected,
                    base_model=BASE_REPO,
                    base_revision=BASE_REVISION,
                )
        else:
            path = Path(self._resolve_weights_path(str(model_path)))
            if not path.is_file():
                from ...utils.download import download_weights

                download_weights(str(path), "b")
            packet = load_untrusted_torch_file(
                path, map_location="cpu", context="Marigold V2 checkpoint"
            )
        validate_checkpoint_metadata(packet, strict=True)
        if (
            packet["model_family"] != self.FAMILY
            or packet["size"] != "b"
            or packet["nc"] != 1
        ):
            raise ValueError("Checkpoint is not a Marigold V2 base dense model.")
        selected = variant_from_state(packet["model"])
        if selected is None or selected != packet.get("variant"):
            raise ValueError("Marigold V2 variant metadata and tensor marker disagree.")
        actual_task = VARIANTS[selected].task
        if packet["task"] != actual_task or (task is not None and task != actual_task):
            raise ValueError("Marigold V2 checkpoint and requested task disagree.")
        if variant is not None and variant != selected:
            raise ValueError("Marigold V2 checkpoint and requested variant disagree.")
        if (packet.get("base_model"), packet.get("base_revision")) != (
            BASE_REPO,
            BASE_REVISION,
        ):
            raise ValueError(
                "Marigold V2 checkpoint requires a different or unpinned base model."
            )
        self.variant = selected
        self.depth_encoding = VARIANTS[selected].encoding or "inverse_depth"
        self.base_path = base_path
        self.quantization = quantization
        self.seed = seed
        self._initial_state = packet["model"]
        try:
            super().__init__(
                self._initial_state,
                size="b",
                nb_classes=1,
                device=device,
                task=actual_task,
            )
        finally:
            del self._initial_state
        self.model_path = str(path) if path is not None else None
        self._cache_checkpoint_train_config(packet)
        self.names = {0: actual_task}
        self.model.eval()
        from .validation import VALIDATORS

        self.validator_class = VALIDATORS[actual_task]

    def _init_model(self):
        vae, transformer = build_components(
            self.device, base_path=self.base_path, quantization=self.quantization
        )
        return MarigoldV2Net(
            vae,
            transformer,
            *(self._initial_state[key] for key in PROMPT_KEYS),
            seed=self.seed,
        )

    def _load_state_dict_logged(self, state_dict, *, source):
        if variant_from_state(state_dict) != self.variant:
            raise ValueError("Marigold V2 checkpoint has the wrong variant marker.")
        known = {VARIANT_KEY, *PROMPT_KEYS}
        self.model.load_trainables(
            {key: value for key, value in state_dict.items() if key not in known}
        )

    def _get_available_layers(self):
        return {"vae": self.model.VAE, "transformer": self.model.Diffuser}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(self, image, color_format="auto", input_size=None):
        image = ImageLoader.load(image, color_format=color_format)
        array, ratio = preprocess_numpy(np.asarray(image), input_size or 0)
        return torch.from_numpy(array)[None], image, image.size, ratio

    def _forward(self, input_tensor):
        # Keep conditioning and seeded VAE sampling independent of the loader's
        # batch size, and bound the 20B graph's activation memory to one image.
        if input_tensor.shape[0] > 1:
            value = torch.cat(
                [
                    output_map(self.model(image[None]), self.task)
                    for image in input_tensor
                ]
            )
        else:
            value = output_map(self.model(input_tensor), self.task)
        if self.task == "normal":
            value = value.float() * value.new_tensor([1, -1, -1]).view(1, 3, 1, 1)
        return {self.task: value}

    def _postprocess(self, output, conf_thres, iou_thres, original_size, **kwargs):
        array = resize_output(output[self.task][0], original_size)
        if self.task == "depth":
            return {"depth": array[0], "depth_encoding": self.depth_encoding}
        array = array.transpose(1, 2, 0)
        if self.task == "normal":
            magnitude = np.linalg.norm(array, axis=-1, keepdims=True)
            if np.any(magnitude <= 1e-6) or not np.isfinite(array).all():
                raise ValueError("Marigold V2 produced an invalid normal vector.")
            array = array / magnitude
        elif self.task == "albedo":
            array = array.clip(0, 1)
        return {self.task: array}

    def __call__(self, source=None, *, imgsz=None, **kwargs):
        return super().__call__(source, imgsz=0 if imgsz is None else imgsz, **kwargs)

    predict = __call__

    def save(self, filename):
        state = self.model.trainable_state_dict()
        state[VARIANT_KEY] = torch.tensor(
            tuple(VARIANTS).index(self.variant), dtype=torch.int64
        )
        state[PROMPT_KEYS[0]] = self.model.prompt_embeds.detach().cpu()
        state[PROMPT_KEYS[1]] = self.model.prompt_mask.detach().cpu()
        packet = wrap_libreyolo_checkpoint(
            state,
            model_family=self.FAMILY,
            size="b",
            task=self.task,
            nc=1,
            names=self.names,
            imgsz=self.input_size,
            variant=self.variant,
            base_model=BASE_REPO,
            base_revision=BASE_REVISION,
            depth_encoding=VARIANTS[self.variant].encoding,
        )
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(packet, destination)
        return str(destination)

    def export(self, *args, **kwargs):
        raise NotImplementedError(
            "Marigold V2 quantized diffusion export is not integrated."
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Marigold V2 fine-tuning is not integrated. This port supports pretrained inference and validation."
        )

    def val(self, data=None, batch=1, imgsz=1024, workers=0, **kwargs):
        imgsz = 1024 if imgsz is None else imgsz
        if not isinstance(imgsz, int) or imgsz <= 0 or imgsz % 16:
            raise ValueError(
                "Marigold V2 validation imgsz must be a positive multiple of 16."
            )
        return super().val(data, batch=batch, imgsz=imgsz, workers=workers, **kwargs)
