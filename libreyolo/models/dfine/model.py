"""LibreDFINE — BaseModel wrapper for the D-FINE native detection family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import DFINEValPreprocessor
from ..base import BaseModel
from .nn import LibreDFINEModel
from .utils import postprocess, preprocess_image, unwrap_dfine_checkpoint


class LibreDFINE(BaseModel):
    """LibreYOLO wrapper for D-FINE.

    Loads upstream ``dfine_{n,s,m,l,x}_coco.pth`` checkpoints and supports
    inference, fine-tuning (via ``DFINETrainer``), validation, and export
    (ONNX / TensorRT / OpenVINO).
    """

    FAMILY = "dfine"
    FILENAME_PREFIX = "LibreDFINE"
    INPUT_SIZES = {"n": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    val_preprocessor_class = DFINEValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # D-FINE-unique decoder key prefixes absent from RF-DETR / YOLOX /
        # YOLOv9 / YOLO-NAS state dicts.
        markers = (
            "decoder.dec_bbox_head.",
            "decoder.denoising_class_embed",
            "decoder.pre_bbox_head.",
            "decoder.enc_bbox_head.",
        )
        return any(any(m in k for m in markers) for k in weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer size from distinctive shapes.

        Strategy: encoder hidden_dim (via ``encoder.lateral_convs.0.conv.weight``
        in-channels) separates N (128) / X (384) / the rest (256). Among the
        256-group, encoder input_proj[0] in-channels (backbone stage out
        channels) distinguishes S (256, B0) / M (384, B2) / L (512, B4).
        """
        enc_hidden_key = "encoder.lateral_convs.0.conv.weight"
        if enc_hidden_key not in weights_dict:
            return None
        enc_hidden = int(weights_dict[enc_hidden_key].shape[1])

        enc_levels = sum(
            1
            for k in weights_dict
            if k.startswith("encoder.input_proj.") and "conv.weight" in k
        )

        if enc_hidden == 128 and enc_levels == 2:
            return "n"
        if enc_hidden == 384 and enc_levels == 3:
            return "x"
        if enc_hidden == 256 and enc_levels == 3:
            enc0_key = "encoder.input_proj.0.conv.weight"
            if enc0_key in weights_dict:
                enc0_in = int(weights_dict[enc0_key].shape[1])
                return {256: "s", 384: "m", 512: "l"}.get(enc0_in)
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
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
        **kwargs,
    ):
        if isinstance(model_path, dict):
            model_path = unwrap_dfine_checkpoint(model_path)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return LibreDFINEModel(
            config=self.size,
            nb_classes=self.nb_classes,
            eval_spatial_size=(self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "backbone_stem": self.model.backbone.stem,
            "encoder": self.model.encoder,
            "encoder_input_proj": self.model.encoder.input_proj,
            "encoder_fpn": self.model.encoder.fpn_blocks,
            "encoder_pan": self.model.encoder.pan_blocks,
            "decoder": self.model.decoder,
            "decoder_input_proj": self.model.decoder.input_proj,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        return preprocess_image(
            image, input_size=effective_size, color_format=color_format,
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
        **kwargs,
    ) -> Dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
        )

    def _strict_loading(self) -> bool:
        # D-FINE checkpoints carry buffers (anchors, valid_mask) that are
        # regenerated at forward time from eval_spatial_size. Tolerate drift.
        return False

    def train(
        self,
        data: str,
        *,
        epochs: int = 132,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 2e-4,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "dfine_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = False,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        """Fine-tune or train D-FINE on a YOLO-format dataset config.

        For v1 inference-only usage, just don't call this. To fine-tune from
        upstream weights, pass ``data="coco128.yaml"`` (or your own data yaml).
        """
        from libreyolo.data import load_data_config

        from .trainer import DFINETrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if seed > 0:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = DFINETrainer(
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
            **kwargs,
        )

        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LibreDFINE('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        # The trainer may have forced a different device than the wrapper's
        # (e.g., MPS-fallback to CPU). Restore the model to the wrapper's
        # device so subsequent ``model.val()`` / ``model.predict()`` calls
        # don't hit input/weight device mismatches.
        self.model.to(self.device)

        return results

    def _load_weights(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"D-FINE weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = unwrap_dfine_checkpoint(loaded)
            state_dict = self._strip_ddp_prefix(dict(state_dict))

            if isinstance(loaded, dict):
                ckpt_family = loaded.get("model_family", "")
                own_family = self._get_model_name()
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'."
                    )
                ckpt_nc = loaded.get("nc")
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(int(ckpt_nc))
                ckpt_names = loaded.get("names")
                effective_nc = (
                    int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                )
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)

            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=self._strict_loading()
            )
            # Loudly surface unexpected keys during bring-up — silent drift here
            # has historically masked whole-module misalignment.
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys when loading D-FINE weights: {sorted(unexpected)[:10]}"
                    + (f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "")
                )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load D-FINE weights from {model_path}: {e}"
            ) from e
