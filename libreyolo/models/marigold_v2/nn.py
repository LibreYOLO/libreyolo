"""Marigold V2's one-step Qwen graph using the Apache-2.0 Diffusers models.

Adapted from huawei-bayerlab/marigold-v2 at
cc6a7031abcd59fd9e1ceff7fdd0d9687d389bc5 (Apache-2.0).
Copyright 2026 Huawei Technologies Co., Ltd.
LibreYOLO changes: remove the global registry, enforce adapter completeness,
use tensor-only prompt loading, and isolate inference random-number state.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

BASE_REPO = "Qwen/Qwen-Image-Edit-2509"
BASE_REVISION = "d3968ef930e841f4c73640fb8afa3b306a78167e"
LORA_TARGETS = (
    "img_in",
    "txt_in",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
    "norm.linear",
    "a_to_out",
    "b_to_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "norm_out.linear",
)


def build_components(device, *, base_path=None, quantization="4bit"):
    """Load the pinned frozen base with the same NF4 recipe as upstream."""
    try:
        from diffusers import (
            AutoencoderKLQwenImage,
            BitsAndBytesConfig,
            QwenImageTransformer2DModel,
        )
        from peft import LoraConfig, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "Install Marigold V2 dependencies: pip install 'libreyolo[marigold]'"
        ) from exc

    device = torch.device(device)
    if quantization not in {"4bit", "none"}:
        raise ValueError("Marigold V2 quantization must be '4bit' or 'none'.")
    if quantization == "4bit" and device.type != "cuda":
        raise ValueError("Marigold V2 NF4 inference requires CUDA.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Marigold V2 currently supports CPU or CUDA, not MPS.")
    source = str(base_path) if base_path is not None else BASE_REPO
    common = {"torch_dtype": torch.bfloat16, "use_safetensors": True}
    if base_path is None:
        common["revision"] = BASE_REVISION
    vae = AutoencoderKLQwenImage.from_pretrained(source, subfolder="vae", **common)
    vae.requires_grad_(False).to(device)
    quant_cfg = None
    if quantization == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        source,
        subfolder="transformer",
        quantization_config=quant_cfg,
        device_map={"": device},
        **common,
    )
    if quant_cfg is not None:
        transformer = prepare_model_for_kbit_training(
            transformer,
            use_gradient_checkpointing=False,
        )
    transformer.requires_grad_(False)
    if quant_cfg is not None:
        from bitsandbytes.functional import dequantize_4bit

        layer = transformer.proj_out
        replacement = nn.Linear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            replacement.weight.copy_(
                dequantize_4bit(layer.weight.data, layer.weight.quant_state)
            )
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias)
        replacement.requires_grad_(False)
        transformer.proj_out = replacement
    transformer.add_adapter(
        LoraConfig(
            r=128,
            lora_alpha=128,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=list(LORA_TARGETS),
        )
    )
    for name, parameter in transformer.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(torch.bfloat16)
    return vae, transformer


class MarigoldV2Net(nn.Module):
    """RGB in [-1,1] to the three-channel native VAE prediction."""

    def __init__(self, vae, transformer, prompt_embeds, prompt_mask, *, seed=2025):
        super().__init__()
        self.VAE = vae
        self.Diffuser = transformer
        if prompt_embeds.ndim != 3 or prompt_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError("Prompt embeddings and attention mask are misaligned.")
        if prompt_embeds.shape[0] == 0:
            raise ValueError("Marigold V2 requires at least one prompt embedding.")
        self.register_buffer(
            "prompt_embeds", prompt_embeds.contiguous(), persistent=False
        )
        self.register_buffer(
            "prompt_mask", prompt_mask.bool().contiguous(), persistent=False
        )
        self.seed = int(seed)
        self._loaded_keys = frozenset()
        self.requires_grad_(False)

    def load_trainables(self, state):
        """Require every LoRA tensor and, if present, the complete VAE decoder."""
        expected = {
            "Diffuser." + name
            for name, _ in self.Diffuser.named_parameters()
            if "lora_" in name
        }
        if not expected:
            raise RuntimeError("The base transformer has no LoRA adapters.")
        if any(key.startswith("VAE.") for key in state):
            expected.update(
                "VAE." + name
                for name, _ in self.VAE.named_parameters()
                if name.startswith(("decoder.", "post_quant_conv."))
            )
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise ValueError(
                f"Incomplete Marigold V2 checkpoint: missing={sorted(missing)[:5]}, "
                f"unexpected={sorted(unexpected)[:5]}"
            )
        parameters = dict(self.named_parameters())
        for key, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameters[key].shape
            ):
                raise ValueError(f"Invalid Marigold V2 tensor shape: {key}")
        # Check all shapes before changing any parameter.
        with torch.no_grad():
            for key, value in state.items():
                parameters[key].copy_(value)
        self._loaded_keys = frozenset(expected)

    def trainable_state_dict(self):
        keys = self._loaded_keys | {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in self.named_parameters()
            if name in keys
        }

    def enable_finetuning(self, *, train_decoder=True):
        self.requires_grad_(False)
        for name, parameter in self.Diffuser.named_parameters():
            parameter.requires_grad_("lora_" in name)
        if train_decoder:
            for name, parameter in self.VAE.named_parameters():
                parameter.requires_grad_(
                    name.startswith(("decoder.", "post_quant_conv."))
                )
        self.Diffuser.enable_gradient_checkpointing()
        self.VAE.enable_gradient_checkpointing()
        self.train()

    @staticmethod
    def _stats(vae, latent):
        shape = (1, vae.config.z_dim, 1, 1, 1)
        mean = latent.new_tensor(vae.config.latents_mean).view(shape)
        inverse_std = (1.0 / latent.new_tensor(vae.config.latents_std)).view(shape)
        return mean, inverse_std

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or any(d % 16 for d in rgb.shape[-2:]):
            raise ValueError(
                "Marigold V2 expects BCHW RGB with dimensions divisible by 16."
            )
        if not self.training:
            devices = (
                [
                    rgb.device.index
                    if rgb.device.index is not None
                    else torch.cuda.current_device()
                ]
                if rgb.is_cuda
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                if rgb.is_cuda:
                    with torch.cuda.device(rgb.device):
                        torch.cuda.manual_seed(self.seed)
                return self._run_graph(rgb)
        return self._run_graph(rgb)

    def _run_graph(self, rgb):
        # The upstream RunInference wrapper supplies BF16 autocast. k-bit
        # preparation leaves selected linears in FP32, so this is required.
        context = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if rgb.is_cuda
            else nullcontext()
        )
        with context:
            return self._forward_graph(rgb)

    def _forward_graph(self, rgb):
        from diffusers import QwenImageEditPipeline

        with torch.no_grad():
            latent = self.VAE.encode(
                rgb.to(dtype=self.VAE.dtype).unsqueeze(2)
            ).latent_dist.sample()
            mean, inv_std = self._stats(self.VAE, latent)
            latent = (latent - mean) * inv_std
        batch, channels, _, height, width = latent.shape
        packed = QwenImageEditPipeline._pack_latents(
            latent[:, :, 0],
            batch_size=batch,
            num_channels_latents=channels,
            height=height,
            width=width,
        ).to(torch.bfloat16)
        # The depth bundle contains eight contexts. Upstream takes the first B,
        # repeating them only when B exceeds the stored context count.
        repeats = (batch + self.prompt_embeds.shape[0] - 1) // self.prompt_embeds.shape[
            0
        ]
        embeds = self.prompt_embeds.repeat(repeats, 1, 1)[:batch].to(
            packed.device, torch.bfloat16
        )
        mask = self.prompt_mask.repeat(repeats, 1)[:batch].to(packed.device)
        timestep = (
            torch.full((batch,), 499.0, device=packed.device, dtype=torch.bfloat16)
            / 1000.0
        )
        context = (
            self.Diffuser.cache_context("cond")
            if hasattr(self.Diffuser, "cache_context")
            else nullcontext()
        )
        with context:
            velocity = self.Diffuser(
                hidden_states=packed,
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=mask,
                img_shapes=[[(1, height // 2, width // 2)]] * batch,
                txt_seq_lens=mask.sum(dim=1).tolist(),
                guidance=None,
                attention_kwargs=getattr(self.Diffuser, "attention_kwargs", None) or {},
                return_dict=False,
            )[0]
        temporal = self.VAE.config.get("temperal_downsample")
        scale = 2 ** len(temporal) if temporal is not None else 8
        velocity = QwenImageEditPipeline._unpack_latents(
            velocity,
            height=height * scale,
            width=width * scale,
            vae_scale_factor=scale,
        )
        predicted = latent.to(self.VAE.dtype) - velocity.to(self.VAE.dtype)
        mean, inv_std = self._stats(self.VAE, predicted)
        return self.VAE.decode(predicted / inv_std + mean).sample[:, :, 0]
