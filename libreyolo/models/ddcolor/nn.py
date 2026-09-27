"""Checkpoint-compatible DDColor colorization network.

Adapted from ``piddnad/DDColor/ddcolor/model.py`` at commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13`` under Apache-2.0. The port is
inference-only, retains official module/parameter names, and excludes training
utilities and unlicensed demo assets. Construction uses known ConvNeXt channel
dimensions instead of an upstream random dummy forward; inference numerics are
unchanged. See the family ``NOTICE`` for full provenance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .convnext import ConvNeXt
from .transformer import (
    CrossAttentionLayer,
    FFNLayer,
    MLP,
    PositionEmbeddingSine,
    SelfAttentionLayer,
)
from .unet import (
    CustomPixelShuffleICNR,
    NormType,
    UnetBlockWide,
    custom_conv_layer,
)


class DDColor(nn.Module):
    """DDColor's ConvNeXt encoder and dual color decoder."""

    def __init__(
        self,
        encoder_name: str = "convnext-l",
        decoder_name: str = "MultiScaleColorDecoder",
        num_input_channels: int = 3,
        input_size: Sequence[int] = (256, 256),
        nf: int = 512,
        num_output_channels: int = 3,
        last_norm: str = "Weight",
        do_normalize: bool = False,
        num_queries: int = 256,
        num_scales: int = 3,
        dec_layers: int = 9,
    ) -> None:
        super().__init__()
        del num_input_channels, input_size
        if decoder_name != "MultiScaleColorDecoder":
            raise ValueError("LibreDDColor supports MultiScaleColorDecoder only.")

        self.encoder = ImageEncoder(encoder_name)
        self.encoder.eval()
        self.decoder = DuelDecoder(
            self.encoder.channels,
            nf=nf,
            last_norm=last_norm,
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
            decoder_name=decoder_name,
        )
        self.refine_net = nn.Sequential(
            custom_conv_layer(
                num_queries + 3,
                num_output_channels,
                kernel_size=1,
                use_activation=False,
                norm_type=NormType.Spectral,
            )
        )

        self.do_normalize = do_normalize
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.std + self.mean

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        model_input = self.normalize(image) if image.shape[1] == 3 else image
        features = self.encoder(model_input)
        decoded = self.decoder(features)
        output = self.refine_net(torch.cat((decoded, model_input), dim=1))
        return self.denormalize(output) if self.do_normalize else output


class ImageEncoder(nn.Module):
    """ConvNeXt feature encoder returning DDColor's normalized skips."""

    def __init__(self, encoder_name: str) -> None:
        super().__init__()
        if encoder_name == "convnext-t":
            depths = (3, 3, 9, 3)
            dims = (96, 192, 384, 768)
        elif encoder_name == "convnext-l":
            depths = (3, 3, 27, 3)
            dims = (192, 384, 768, 1536)
        else:
            raise ValueError(
                f"Unsupported DDColor encoder {encoder_name!r}; choose convnext-t or convnext-l."
            )

        self.arch = ConvNeXt(depths=depths, dims=dims)
        self.encoder_name = encoder_name
        self.channels = tuple(dims)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, features = self.arch.forward_features(image, return_intermediates=True)
        return features


class DuelDecoder(nn.Module):
    """DDColor UNet decoder plus multi-scale color-query decoder."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        nf: int = 512,
        blur: bool = True,
        last_norm: str = "Weight",
        num_queries: int = 256,
        num_scales: int = 3,
        dec_layers: int = 9,
        decoder_name: str = "MultiScaleColorDecoder",
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.decoder_name = decoder_name

        self.layers = self.make_layers()
        embed_dim = nf // 2
        self.last_shuf = CustomPixelShuffleICNR(
            embed_dim,
            embed_dim,
            blur=self.blur,
            norm_type=self.last_norm,
            scale=4,
        )
        if decoder_name != "MultiScaleColorDecoder":
            raise ValueError("LibreDDColor supports MultiScaleColorDecoder only.")
        self.color_decoder = MultiScaleColorDecoder(
            in_channels=(512, 512, 256),
            num_queries=num_queries,
            num_scales=num_scales,
            dec_layers=dec_layers,
        )

    def make_layers(self) -> nn.Sequential:
        decoder_layers: list[nn.Module] = []
        in_channels = self.encoder_channels[-1]
        out_channels = self.nf
        skip_channels = self.encoder_channels[-2::-1]
        for layer_index, feature_channels in enumerate(skip_channels):
            if layer_index == len(skip_channels) - 1:
                out_channels //= 2
            decoder_layers.append(
                UnetBlockWide(
                    in_channels,
                    feature_channels,
                    out_channels,
                    blur=self.blur,
                    norm_type=NormType.Spectral,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*decoder_layers)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 4:
            raise RuntimeError(
                f"DDColor encoder must return four decoder features, got {len(features)}."
            )
        encoded = features[-1]
        skips = features[-2::-1]
        out0 = self.layers[0](encoded, skips[0])
        out1 = self.layers[1](out0, skips[1])
        out2 = self.layers[2](out1, skips[2])
        out3 = self.last_shuf(out2)
        return self.color_decoder((out0, out1, out2), out3)


class MultiScaleColorDecoder(nn.Module):
    """Transformer queries that project multi-scale features into color maps."""

    def __init__(
        self,
        in_channels: Sequence[int],
        hidden_dim: int = 256,
        num_queries: int = 100,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        pre_norm: bool = False,
        color_embed_dim: int = 256,
        enforce_input_project: bool = True,
        num_scales: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = dec_layers
        self.num_feature_levels = num_scales
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.level_embed = nn.Embedding(num_scales, hidden_dim)
        self.input_proj = nn.ModuleList(
            self._make_input_proj(channels, hidden_dim, enforce_input_project)
            for channels in in_channels
        )

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

    @staticmethod
    def _make_input_proj(
        in_channels: int,
        hidden_dim: int,
        enforce: bool,
    ) -> nn.Module:
        if in_channels != hidden_dim or enforce:
            projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(projection.weight, a=1)
            if projection.bias is not None:
                nn.init.constant_(projection.bias, 0)
            return projection
        return nn.Sequential()

    def _get_src_and_pos(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        sources: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for index, feature in enumerate(features):
            positions.append(self.pe_layer(feature).flatten(2).permute(2, 0, 1))
            projected = self.input_proj[index](feature).flatten(2)
            projected = projected + self.level_embed.weight[index][None, :, None]
            sources.append(projected.permute(2, 0, 1))
        return sources, positions

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} DDColor feature levels, got {len(features)}."
            )
        sources, positions = self._get_src_and_pos(features)
        batch_size = sources[0].shape[1]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)

        for index in range(self.num_layers):
            level = index % self.num_feature_levels
            output = self.transformer_cross_attention_layers[index](
                output,
                sources[level],
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=positions[level],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[index](
                output,
                target_mask=None,
                target_key_padding_mask=None,
                query_pos=query_embed,
            )
            output = self.transformer_ffn_layers[index](output)

        decoder_output = self.decoder_norm(output).transpose(0, 1)
        color_embed = self.color_embed(decoder_output)
        return torch.einsum("bqc,bchw->bqhw", color_embed, image_features)


__all__ = ["DDColor", "DuelDecoder", "ImageEncoder", "MultiScaleColorDecoder"]
