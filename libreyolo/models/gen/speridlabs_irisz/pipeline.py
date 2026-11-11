"""Iris-Z sampling: the RGB and depth grids of one sequence, the schedule, depth decoding.

Ported from the Iris-Z SDK (``iris_z/rgbd.py``, ``iris_z/depth.py``,
``iris_z/pipeline.py`` and the schedule in ``iris_z/model/sampling.py``, the
last from FLUX.2). See ``NOTICE``. Randomness follows upstream: a
``torch.Generator`` on the model's device, seeded per sample.

Iris-Z denoises a sequence of two latent grids on the same positions, RGB
first and depth second, each token tagged with its modality. Image -> depth
keeps the RGB grid clean (``GIVEN_RGB``, noise level 0) and denoises depth;
prompt -> image denoises RGB and leaves depth as pure noise (level 1).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

RGB = 0
DEPTH = 1
GIVEN_RGB = 2
GIVEN_DEPTH = 3
DEPTH_TIME_COORDINATE = 1
SKY_VALUE = 1.0 - 1e-6
# Depth runs at 512 for images under half a megapixel, else at 1024.
SMALL_IMAGE_PIXELS = 500_000
LATENT_CHANNELS = 128


# -- Schedule (FLUX.2) -------------------------------------------------------


def _time_snr_shift(t: Tensor, mu: float, sigma: float) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int) -> List[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    return _time_snr_shift(timesteps, mu, 1.0).tolist()


# -- Token layout ------------------------------------------------------------


def text_context(embedding: Tensor) -> Tuple[Tensor, Tensor]:
    """``[B, L, D]`` prompt embeddings and their ``(t, h, w, l)`` position ids."""
    batch, length, _ = embedding.shape
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)
    ).to(embedding.device)
    return embedding, ids.unsqueeze(0).expand(batch, -1, -1).contiguous()


def grid_tokens(latent: Tensor, time_coordinate: int) -> Tuple[Tensor, Tensor]:
    batch, channels, height, width = latent.shape
    tokens = latent.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    device = latent.device
    ids = torch.cartesian_prod(
        torch.tensor([time_coordinate], device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        torch.tensor([0], device=device),
    )
    return tokens, ids.unsqueeze(0).expand(batch, -1, -1)


def rgbd_sequence(rgb_latent: Tensor, depth_latent: Tensor):
    """``[RGB tokens, depth tokens]`` on the same (h, w) positions."""
    rgb_tokens, rgb_ids = grid_tokens(rgb_latent, 0)
    depth_tokens, depth_ids = grid_tokens(depth_latent, DEPTH_TIME_COORDINATE)
    tokens = torch.cat([rgb_tokens, depth_tokens], dim=1)
    ids = torch.cat([rgb_ids, depth_ids], dim=1)
    batch, rgb_length = rgb_tokens.shape[:2]
    modality = torch.cat(
        [
            torch.full((batch, rgb_length), RGB, device=tokens.device, dtype=torch.long),
            torch.full((batch, depth_tokens.shape[1]), DEPTH, device=tokens.device, dtype=torch.long),
        ],
        dim=1,
    )
    return tokens, ids, modality, rgb_length


def tokens_to_grid(tokens: Tensor, height: int, width: int) -> Tensor:
    """``[B, h*w, C]`` row-major tokens back to a ``[B, C, h, w]`` latent."""
    batch, _, channels = tokens.shape
    return tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


def initial_noise(shape, seed: int, device, dtype) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def guided_velocity(prediction: Tensor, guidance: float) -> Tensor:
    if prediction.shape[0] == 1:
        return prediction
    unconditional, conditional = prediction.chunk(2)
    return unconditional + guidance * (conditional - unconditional)


# -- Samplers ----------------------------------------------------------------


@torch.no_grad()
def sample_rgb(model, context, height: int, width: int, seed: int, guidance: float, num_steps: int) -> Tensor:
    """Prompt -> image latent. ``context`` holds [unconditional, conditional] rows.

    The depth grid stays in the sequence as pure noise and is never denoised.
    """
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    grid_h, grid_w = height // 16, width // 16
    shape = (1, LATENT_CHANNELS, grid_h, grid_w)
    rgb_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    depth_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    tokens, ids, modality, rgb_length = rgbd_sequence(rgb_noise, depth_noise)
    rows = ctx.shape[0]
    batch_ids = torch.cat([ids] * rows)
    batch_modality = torch.cat([modality] * rows)
    depth_level = torch.ones((rows,), dtype=dtype, device=device)
    timesteps = get_schedule(num_steps, rgb_length)
    for t_current, t_previous in zip(timesteps[:-1], timesteps[1:]):
        t_vector = torch.full((rows,), t_current, dtype=dtype, device=device)
        prediction = model(
            x=torch.cat([tokens] * rows),
            x_ids=batch_ids,
            timesteps=t_vector,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            modality_ids=batch_modality,
            rgb_timesteps=t_vector,
            depth_timesteps=depth_level,
            depth_start=rgb_length,
        )
        velocity = guided_velocity(prediction, guidance)
        tokens[:, :rgb_length] = tokens[:, :rgb_length] + (t_previous - t_current) * velocity[:, :rgb_length]
    return tokens_to_grid(tokens[:, :rgb_length], grid_h, grid_w)


@torch.no_grad()
def sample_depth(
    model,
    context,
    rgb_latent: Tensor,
    seeds: Sequence[int],
    num_steps: int,
    average_last: int = 0,
    decode: Optional[Callable[[Tensor], np.ndarray]] = None,
):
    """Image -> depth, one result per seed.

    ``average_last = k > 0`` returns, per seed, the median of the last ``k``
    clean-depth estimates as log depth (each affinely aligned to the last);
    ``0`` returns the final depth latents.
    """
    ctx, ctx_ids = context
    device = ctx.device
    dtype = next(model.parameters()).dtype
    batch = len(seeds)
    grid_h, grid_w = rgb_latent.shape[-2:]
    rgb_latent = rgb_latent.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)
    clean, ids, modality, rgb_length = rgbd_sequence(rgb_latent, torch.zeros_like(rgb_latent))
    tokens = torch.cat([initial_noise((1,) + tuple(clean.shape[1:]), seed, device, dtype) for seed in seeds])
    tokens[:, :rgb_length] = clean[:, :rgb_length]
    modality[:, :rgb_length] = GIVEN_RGB
    ctx = ctx.expand(batch, -1, -1)
    ctx_ids = ctx_ids.expand(batch, -1, -1)
    rgb_level = torch.zeros((batch,), dtype=dtype, device=device)
    timesteps = get_schedule(num_steps, rgb_length)
    first_averaged = len(timesteps) - 1 - average_last
    estimates = []
    for step, (t_current, t_previous) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vector = torch.full((batch,), t_current, dtype=dtype, device=device)
        velocity = model(
            x=tokens,
            x_ids=ids,
            timesteps=t_vector,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=None,
            modality_ids=modality,
            rgb_timesteps=rgb_level,
            depth_timesteps=t_vector,
            depth_start=rgb_length,
        )
        if average_last and step >= first_averaged:
            estimate = tokens[:, rgb_length:] - t_current * velocity[:, rgb_length:]
            estimate_grid = tokens_to_grid(estimate, grid_h, grid_w)
            estimates.append([decode(estimate_grid[i : i + 1]) for i in range(batch)])
        tokens[:, rgb_length:] = tokens[:, rgb_length:] + (t_previous - t_current) * velocity[:, rgb_length:]
    if not average_last:
        depth_grid = tokens_to_grid(tokens[:, rgb_length:], grid_h, grid_w)
        return [depth_grid[i : i + 1] for i in range(batch)]
    averaged = []
    for i in range(batch):
        members = [step_estimates[i] for step_estimates in estimates]
        aligned = [align_affine(member, members[-1]) for member in members]
        averaged.append(keep_sky(np.median(np.stack(aligned), axis=0), members))
    return averaged


# -- Depth values ------------------------------------------------------------


def align_affine(member: np.ndarray, reference: np.ndarray) -> np.ndarray:
    design = np.stack([member.ravel(), np.ones(member.size)], axis=1)
    (scale, shift), _, _, _ = np.linalg.lstsq(design, reference.ravel(), rcond=None)
    return scale * member + shift


def keep_sky(averaged: np.ndarray, members: Sequence[np.ndarray]) -> np.ndarray:
    """Pixels most members mark as sky (+1) stay at +1; alignment moves them slightly."""
    sky_votes = np.mean([member >= SKY_VALUE for member in members], axis=0)
    return np.where(sky_votes > 0.5, 1.0, averaged)


def ensemble(log_depths: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of several depths, each affinely aligned to the first."""
    reference = log_depths[0]
    aligned = [reference] + [align_affine(member, reference) for member in log_depths[1:]]
    return keep_sky(np.median(np.stack(aligned), axis=0), log_depths)


def unsqueeze_from_ceiling(encoded: np.ndarray, sky_ceiling: float) -> np.ndarray:
    """Real depth back from ``[-1, sky_ceiling]`` to ``[-1, 1]``; sky to +1."""
    encoded = np.clip(np.asarray(encoded, dtype=np.float64), -1.0, 1.0)
    if sky_ceiling >= 1.0:
        return encoded
    real = np.clip(-1.0 + (encoded + 1.0) * 2.0 / (sky_ceiling + 1.0), -1.0, 1.0)
    return np.where(encoded > (sky_ceiling + 1.0) / 2.0, 1.0, real)


def choose_resolution(
    source_width: int, source_height: int, target_side: int, multiple: int = 16, area_tolerance: float = 0.2
) -> Tuple[int, int]:
    """About ``target_side**2`` pixels, same aspect ratio, multiples of 16, never larger than the source."""
    source_aspect = source_width / source_height
    target_area = min(target_side * target_side, source_width * source_height)
    best = None
    for height in range(multiple, source_height + 1, multiple):
        for width in range(multiple, source_width + 1, multiple):
            area = width * height
            if abs(area - target_area) > area_tolerance * target_area:
                continue
            aspect_error = abs(math.log((width / height) / source_aspect))
            key = (round(aspect_error, 6), abs(area - target_area))
            if best is None or key < best[0]:
                best = (key, width, height)
    if best is None:
        raise ValueError(f"no size near {target_side}^2 fits {source_width}x{source_height}")
    return best[1], best[2]


# -- FLUX.2 VAE --------------------------------------------------------------


class Flux2Latents:
    """FLUX.2 VAE as Iris-Z uses it: 2x2-patchified, batch-normalized 128-channel latents.

    Wraps diffusers' ``AutoencoderKLFlux2`` (Apache-2.0). Encoding takes the
    posterior mean, as upstream's ``AutoEncoder.encode`` does.
    """

    def __init__(self, vae):
        self.vae = vae

    @property
    def dtype(self):
        return next(self.vae.parameters()).dtype

    def _stats(self, like: Tensor):
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(like.device, like.dtype)
        std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            like.device, like.dtype
        )
        return mean, std

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        """``[B, 3, H, W]`` in ``[-1, 1]`` -> ``[B, 128, H/16, W/16]``."""
        moments = self.vae.encode(pixels.to(self.dtype)).latent_dist.mean
        batch, channels, height, width = moments.shape
        latent = moments.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latent = latent.reshape(batch, channels * 4, height // 2, width // 2)
        mean, std = self._stats(latent)
        return (latent - mean) / std

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """``[B, 128, h, w]`` -> ``[B, 3, 16h, 16w]``, roughly in ``[-1, 1]``."""
        latent = latent.to(self.dtype)
        mean, std = self._stats(latent)
        latent = latent * std + mean
        batch, channels, height, width = latent.shape
        latent = latent.view(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3)
        latent = latent.reshape(batch, channels // 4, height * 2, width * 2)
        return self.vae.decode(latent).sample

    def to(self, device) -> "Flux2Latents":
        self.vae.to(device)
        return self
