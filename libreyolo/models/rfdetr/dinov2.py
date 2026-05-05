# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoBackbone

from .dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)
import logging

logger = logging.getLogger(__name__)

size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}


def get_config(size, use_registers):
    widths = {"small": 384, "base": 768, "large": 1024}
    heads = {"small": 6, "base": 12, "large": 16}
    layers = {"small": 12, "base": 12, "large": 24}
    return {
        "apply_layernorm": True,
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": widths[size],
        "image_size": 518,
        "initializer_range": 0.02,
        "interpolate_antialias": True,
        "interpolate_offset": 0.0,
        "layer_norm_eps": 1e-6,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_attention_heads": heads[size],
        "num_channels": 3,
        "num_hidden_layers": layers[size],
        "num_register_tokens": 4 if use_registers else 0,
        "patch_size": 14,
        "qkv_bias": True,
        "reshape_hidden_states": True,
        "use_swiglu_ffn": False,
    }


class DinoV2(nn.Module):
    def __init__(
        self,
        shape=(640, 640),
        out_feature_indexes=[2, 4, 5, 9],
        size="base",
        use_registers=True,
        use_windowed_attn=True,
        gradient_checkpointing=False,
        load_dinov2_weights=True,
        patch_size=14,
        num_windows=4,
        positional_encoding_size=37,
        drop_path_rate=0.0,
    ):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows

        # Create the encoder

        if not use_windowed_attn:
            assert not gradient_checkpointing, "Gradient checkpointing is not supported for non-windowed attention"
            assert load_dinov2_weights, "Using non-windowed attention requires loading dinov2 weights from hub"
            if drop_path_rate > 0.0:
                logger.warning(
                    "drop_path_rate > 0.0 is not supported for non-windowed DinoV2 backbones."
                    " drop_path will be ignored."
                )
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = get_config(size, use_registers)

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]
            dino_config["drop_path_rate"] = drop_path_rate

            implied_resolution = positional_encoding_size * patch_size

            if implied_resolution != dino_config["image_size"]:
                if load_dinov2_weights:
                    logger.warning(
                        "Using a different number of positional encodings than DINOv2, which means"
                        " we're not loading DINOv2 backbone weights. This is not a problem if"
                        " finetuning a pretrained RF-DETR model."
                    )
                dino_config["image_size"] = implied_resolution
                load_dinov2_weights = False

            if patch_size != 14:
                if load_dinov2_weights:
                    logger.warning(
                        f"Using patch size {patch_size} instead of 14, which means we're not loading"
                        " DINOv2 backbone weights. This is not a problem if finetuning a pretrained"
                        " RF-DETR model."
                    )
                dino_config["patch_size"] = patch_size
                load_dinov2_weights = False

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=num_windows,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            self.encoder = (
                WindowedDinov2WithRegistersBackbone.from_pretrained(
                    name,
                    config=windowed_dino_config,
                )
                if load_dinov2_weights
                else WindowedDinov2WithRegistersBackbone(windowed_dino_config)
            )

        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(position_embeddings, patch_size, height, width):

            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            # Reshape and permute
            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            # Use bicubic interpolation, disabling antialias only on MPS devices
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=patch_pos_embed.device.type != "mps",
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        # If the shape of self.encoder.embeddings.position_embeddings
        # matches the shape of your new tensor, use copy_:
        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        # Create a new Parameter with the new size
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding

        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return self_mod.position_embeddings
            return old_interpolate_pos_encoding(embeddings, height, width)

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x):
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}"
        )
        x = self.encoder(x)
        return list(x[0])


if __name__ == "__main__":
    model = DinoV2()
    model.export()
    x = torch.randn(1, 3, 640, 640)
    logger.info(model(x))
    for j in model(x):
        logger.info(j.shape)
