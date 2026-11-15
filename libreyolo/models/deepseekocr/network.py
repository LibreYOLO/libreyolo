"""The DeepSeek-OCR network: DeepEncoder features scattered into the MoE decoder.

Ported from ``DeepseekOCRModel`` / ``DeepseekOCRForCausalLM`` of the
DeepSeek-OCR Hugging Face release (deepseek-ai/DeepSeek-OCR @
9f30c71f441d010e5429c532364a86705536c53a, MIT) and their DeepSeek-OCR-2
counterparts (deepseek-ai/DeepSeek-OCR-2 @
aaa02f3811945a91062062994c5c4a3f4c0af2b0, Apache-2.0). The image-feature layout
(local crops row-major with a newline column, then the global view with a
newline column, then a view separator) and the greedy decode with
``no_repeat_ngram_size`` follow the release; device placement no longer
assumes CUDA.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional

import torch
import torch.nn as nn

from .deepencoder import MlpProjector, build_clip_l, build_sam_vit_b
from .encoder2 import Qwen2Decoder2Encoder
from .language import DeepSeekOCRTextConfig, DeepSeekV2TextModel, NoRepeatNGram


class DeepSeekOCRBackbone(DeepSeekV2TextModel):
    """Decoder plus the vision tower, under the checkpoint's ``model.`` prefix.

    ``version`` 1 is DeepSeek-OCR (SAM into CLIP-L, newline tokens); version 2
    is DeepSeek-OCR-2 (SAM into a Qwen2 encoder with learned queries, no
    newline tokens).
    """

    def __init__(self, config: DeepSeekOCRTextConfig, version: int = 1):
        super().__init__(config)
        self.version = int(version)
        n_embed = config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.version == 1:
            self.sam_model = build_sam_vit_b()
            self.vision_model = build_clip_l()
            self.projector = MlpProjector(input_dim=2048, n_embed=n_embed)
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        elif self.version == 2:
            self.sam_model = build_sam_vit_b(final_chans=896)
            self.qwen2_model = Qwen2Decoder2Encoder(hidden_size=896)
            self.projector = MlpProjector(input_dim=896, n_embed=n_embed)
        else:
            raise ValueError(f"Unknown DeepSeek-OCR version {version!r}")
        # Spelling kept from the checkpoint key.
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

    def _features(self, pixels: torch.Tensor) -> torch.Tensor:
        sam = self.sam_model(pixels)
        if self.version == 2:
            return self.projector(self.qwen2_model(sam))
        clip = self.vision_model(pixels, sam)
        feats = torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.projector(feats)

    def _with_newlines(self, features: torch.Tensor, rows: int) -> torch.Tensor:
        n_dim = features.shape[-1]
        return torch.cat(
            [features, self.image_newline[None, None, :].expand(rows, 1, n_dim)], dim=1
        ).view(-1, n_dim)

    def encode_image(
        self,
        global_view: torch.Tensor,
        crops: Optional[torch.Tensor],
        grid: tuple,
    ) -> torch.Tensor:
        """Visual token embeddings for one image, in placeholder order.

        Local crops (row-major, a newline column per grid row) come first, then
        the global view (same newline layout), then the view separator. The
        release encodes the crops before the global view; so does this.
        """
        if self.version == 2:
            # No newline tokens: crops in batch order, then the global view.
            parts = []
            if crops is not None:
                local = self._features(crops)
                parts.append(local.reshape(-1, local.shape[-1]))
            glob = self._features(global_view)
            parts += [glob.reshape(-1, glob.shape[-1]), self.view_seperator[None, :]]
            return torch.cat(parts, dim=0)
        local_features = None
        if crops is not None:
            local = self._features(crops)
            _, hw2, n_dim2 = local.shape
            h2 = w2 = int(hw2**0.5)
            cols, rows = grid
            local = (
                local.view(rows, cols, h2, w2, n_dim2)
                .permute(0, 2, 1, 3, 4)
                .reshape(rows * h2, cols * w2, n_dim2)
            )
            local_features = self._with_newlines(local, rows * h2)
        global_features = self._features(global_view)
        _, hw, n_dim = global_features.shape
        h = w = int(hw**0.5)
        global_features = self._with_newlines(global_features.view(h, w, n_dim), h)
        parts = [global_features, self.view_seperator[None, :]]
        if local_features is not None:
            parts.insert(0, local_features)
        return torch.cat(parts, dim=0)


class DeepSeekOCRNetwork(nn.Module):
    """Checkpoint-shaped root module: ``model`` plus ``lm_head``."""

    def __init__(self, config: DeepSeekOCRTextConfig, version: int = 1):
        super().__init__()
        self.config = config
        self.model = DeepSeekOCRBackbone(config, version)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states).float()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        image_mask: torch.Tensor,
        global_view: torch.Tensor,
        crops: Optional[torch.Tensor],
        grid: tuple,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        no_repeat_ngram_size: int = 0,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> List[int]:
        """Greedy decode for one image; returns the new token ids (EOS included).

        ``autocast_dtype`` reproduces the release's ``torch.autocast`` context
        (CUDA bf16); pass ``None`` to run in the weights' dtype.
        """
        device = input_ids.device
        ctx = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with ctx:
            embeds = self.model.embed_tokens(input_ids[None, :])
            features = self.model.encode_image(global_view, crops, grid)
            embeds[0].masked_scatter_(
                image_mask.unsqueeze(-1), features.to(embeds.dtype)
            )
            hidden, past = self.model(embeds)
            logits = self._logits(hidden)[:, -1, :]
            ngram = NoRepeatNGram(no_repeat_ngram_size, input_ids.tolist())
            generated: List[int] = []
            for _ in range(int(max_new_tokens)):
                scores = logits
                banned = ngram.banned()
                if banned:
                    scores = scores.clone()
                    scores[0, banned] = -float("inf")
                next_id = int(torch.argmax(scores, dim=-1)[0])
                generated.append(next_id)
                ngram.append(next_id)
                if next_id == eos_token_id:
                    break
                step = self.model.embed_tokens(
                    torch.tensor([[next_id]], dtype=torch.long, device=device)
                )
                hidden, past = self.model(step, past)
                logits = self._logits(hidden)[:, -1, :]
        return generated
