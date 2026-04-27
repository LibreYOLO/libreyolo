"""LibreECDet — BaseModel wrapper for the ECDet (EdgeCrafter detection) family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import ECDetValPreprocessor
from ..base import BaseModel
from .nn import LibreECDetModel
from .postprocess import postprocess, preprocess_image, unwrap_ecdet_checkpoint


class LibreECDet(BaseModel):
    """LibreYOLO wrapper for EdgeCrafter ECDet."""

    FAMILY = "ecdet"
    FILENAME_PREFIX = "LibreECDet"
    INPUT_SIZES = {"s": 640, "m": 640, "l": 640, "x": 640}
    val_preprocessor_class = ECDetValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # ECDet-unique: ECViT register token in the backbone. Distinct from
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
        return LibreECDetModel(
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
            output, conf_thres=conf_thres, iou_thres=iou_thres,
            original_size=original_size, max_det=max_det,
        )

    def _strict_loading(self) -> bool:
        # ECDet checkpoints carry anchors/valid_mask buffers regenerated at
        # forward time. Mirror the D-FINE policy.
        return False

    def _load_weights(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ECDet weights file not found: {model_path}")

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
                    f"Unexpected keys when loading ECDet weights: {sorted(unexpected)[:10]}"
                    + (f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "")
                )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load ECDet weights from {model_path}: {e}") from e
