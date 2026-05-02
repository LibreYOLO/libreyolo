"""LibreECDET — BaseModel wrapper for the ECDET (EdgeCrafter detection) family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import ECDETValPreprocessor
from ..base import BaseModel
from .nn import LibreECDETModel
from .postprocess import postprocess, preprocess_image, unwrap_ecdet_checkpoint


class LibreECDET(BaseModel):
    """LibreYOLO wrapper for EdgeCrafter ECDET."""

    FAMILY = "ecdet"
    FILENAME_PREFIX = "LibreECDET"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    val_preprocessor_class = ECDETValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ECDET-unique: ECViT register token in the backbone. Distinct from
        # D-FINE's HGNetv2 stem and RT-DETR's resnet/dinov2 backbones.
        return "backbone.backbone.register_token" in weights_dict

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer size from backbone embedding dim and projector out-dim.

        S: embed=192, proj=192        M: embed=256, proj=256
        L: embed=384, proj=256, enc_exp=0.75 (encoder fpn cv4 weight has fewer fan-in)
        X: embed=384, proj=256, enc_exp=1.5
        """
        reg_key = "backbone.backbone.register_token"
        proj_key = "backbone.projector.0.conv.weight"
        if reg_key not in weights_dict or proj_key not in weights_dict:
            return None
        embed_dim = int(weights_dict[reg_key].shape[-1])
        proj_dim = int(weights_dict[proj_key].shape[0])

        if embed_dim == 192 and proj_dim == 192:
            return "s"
        if embed_dim == 256 and proj_dim == 256:
            return "m"
        if embed_dim == 384 and proj_dim == 256:
            # Distinguish L vs X by encoder fusion-block expansion. With
            # expansion=0.75 (L), Fuse_Block c4=round(0.75*256/2)=96 → cv4
            # in_channels = c3 + 2*c4 = 512+192 = 704; with expansion=1.5 (X),
            # c4=192 → cv4 in_channels = 512+384 = 896.
            cv4_key = "encoder.fpn_blocks.0.cv4.conv.weight"
            if cv4_key in weights_dict:
                fan_in = int(weights_dict[cv4_key].shape[1])
                if fan_in == 704:
                    return "l"
                if fan_in == 896:
                    return "x"
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
            model_path = unwrap_ecdet_checkpoint(model_path)
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
        return LibreECDETModel(
            config=self.size,
            nb_classes=self.nb_classes,
            eval_spatial_size=(self.input_size, self.input_size),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "backbone_vit": self.model.backbone.backbone,
            "backbone_projector": self.model.backbone.projector,
            "encoder": self.model.encoder,
            "encoder_fpn": self.model.encoder.fpn_blocks,
            "encoder_pan": self.model.encoder.pan_blocks,
            "decoder": self.model.decoder,
            "decoder_input_proj": self.model.decoder.input_proj,
            "dec_bbox_head": self.model.decoder.dec_bbox_head,
            "dec_score_head": self.model.decoder.dec_score_head,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .postprocess import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        effective = input_size if input_size is not None else self.input_size
        return preprocess_image(image, input_size=effective, color_format=color_format)

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
        # ECDET checkpoints carry anchors/valid_mask buffers regenerated at
        # forward time. Mirror the D-FINE policy.
        return False

    def train(
        self,
        data: str,
        *,
        allow_experimental: bool = False,
        epochs: int = 74,
        batch: int = 16,
        imgsz: int = 640,
        lr0: float = 5e-4,
        device: str = "",
        workers: int = 4,
        seed: int = 0,
        project: str = "runs/train",
        name: str = "ecdet_exp",
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,
        **kwargs,
    ) -> dict:
        """Fine-tune ECDET on a YOLO-format dataset.

        **EXPERIMENTAL.** This training path follows upstream EdgeCrafter's
        published recipe (AdamW, FlatCosine, MAL+L1+GIoU+FGL+DDF, EMA 0.9999,
        Mosaic+Mixup, all strong augs disabled past stop_epoch) and passes
        loss-parity vs upstream's criterion at 1e-5 on synthetic input — but
        a full fine-tune has not been run end-to-end. Pass
        ``allow_experimental=True`` to acknowledge.
        """
        if not allow_experimental:
            raise RuntimeError(
                "ECDET training is experimental and has not been validated by a "
                "full fine-tune. Pass allow_experimental=True to proceed.\n"
                "What's been validated: inference parity (1e-5 vs upstream on all "
                "4 sizes), ONNX export round-trip, COCO val2017 mAP. What's NOT "
                "validated: full fine-tune convergence, multi-GPU, the "
                "stop_aug_epoch best-reload trick, Obj365→COCO class remap."
            )

        from pathlib import Path

        from libreyolo.data import load_data_config
        from .trainer import ECDETTrainer

        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get("yaml_file", data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        yaml_nc = data_config.get("nc")
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        if seed > 0:
            import random as _r
            import numpy as _np

            _r.seed(seed)
            _np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trainer = ECDETTrainer(
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
                raise ValueError("resume=True requires a checkpoint. Load one first.")
            trainer.setup()
            trainer.resume(str(self.model_path))
            return trainer.train()

        results = trainer.train()

        best_ckpt = results.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)

        self.model.to(self.device)
        return results

    def _load_weights(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ECDET weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = unwrap_ecdet_checkpoint(loaded)
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
                effective_nc = int(ckpt_nc) if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)

            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=self._strict_loading()
            )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys when loading ECDET weights: {sorted(unexpected)[:10]}"
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
                f"Failed to load ECDET weights from {model_path}: {e}"
            ) from e
