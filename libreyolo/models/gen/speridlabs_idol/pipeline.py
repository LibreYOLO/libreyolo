# Ported from Speridlabs Idol (https://github.com/speridlabs/idol-sdk, commit
# 4702a7f00850e34b6e83a2170cc123c3e9570744), Apache-2.0, Copyright Speridlabs.
"""Speridlabs Idol around the networks: the crop margins and the flow sampler.

Preprocessing, the image encoders and the camera decoding are shared with
TripoSplat (``libreyolo/models/gen/image3d.py``) and re-exported here.

Each step keeps upstream's arithmetic so a seed reproduces upstream's object.
Randomness is drawn from an explicit generator instead of the global one;
seeded the way upstream seeds the global generator, it yields the same
numbers in the same order.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from ..image3d import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageEncoder,
    decode_camera,
    has_real_alpha,
    preprocess_image,
    remove_background,
)

# The shared steps are re-exported under their upstream names.
__all__ = [
    "CROP_MARGIN",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageEncoder",
    "decode_camera",
    "has_real_alpha",
    "preprocess_image",
    "remove_background",
    "sample_flow",
]

# How much room the square crop leaves around the object: crop side = object
# bounding box x margin, so the object spans 1 / margin of the image. Renders
# of single objects work best tight; photos with a looser crop.
CROP_MARGIN = {"synthetic": 1.2, "real": 1.7}


# -- Sampler -----------------------------------------------------------------


@torch.no_grad()
def sample_flow(
    denoiser,
    latent: torch.Tensor,
    camera: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    steps: int = 50,
    shift: float = 3.0,
    guidance: float = 3.0,
    batch_guidance: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate the (latent, camera) pair from noise (t = 1) to data (t = 0) in ``steps`` Euler steps.

    shift moves the time grid towards t = 1 (more steps at high noise). guidance: guided velocity =
    (1 + w) * v(cond) - w * v(zeroed cond). batch_guidance runs both passes as one batch-2 forward."""
    neg_cond = {name: torch.zeros_like(value) for name, value in cond.items()}

    def velocity(sample, t, condition):
        z, cam = sample
        times = torch.tensor(
            [1000 * t] * z.shape[0], device=z.device, dtype=torch.float32
        )
        return denoiser(z, cam, times, condition)

    sample = (latent, camera)
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = shift * t_seq / (1 + (shift - 1) * t_seq)
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        if guidance > 0 and batch_guidance:
            n = sample[0].shape[0]
            both = velocity(
                tuple(torch.cat([part, part]) for part in sample),
                t,
                {name: torch.cat([cond[name], neg_cond[name]]) for name in cond},
            )
            v = tuple((1 + guidance) * part[:n] - guidance * part[n:] for part in both)
        else:
            v = velocity(sample, t, cond)
            if guidance > 0:
                negative = velocity(sample, t, neg_cond)
                v = tuple(
                    (1 + guidance) * part - guidance * neg
                    for part, neg in zip(v, negative)
                )
        sample = tuple(part - step * (t - t_prev) for part, step in zip(sample, v))
    return sample
