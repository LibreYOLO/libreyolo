# Ported from TripoSplat (https://github.com/VAST-AI-Research/TripoSplat,
# commit d8db9e018b413dd9c4a9fe22463781bf98e8e68d), MIT, Copyright (c) 2026 VAST.
"""TripoSplat around the networks: the flow sampler.

Preprocessing, the image encoders and the camera decoding are shared with
Speridlabs Idol (``libreyolo/models/gen/image3d.py``). The sampler keeps
upstream's ``FlowEulerCfgSampler`` arithmetic and its diffusers-style
guidance convention: ``guidance <= 1`` runs the conditional pass only,
``guidance > 1`` blends ``s * v(cond) - (s - 1) * v(zeroed cond)``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Upstream's crop: the square is 1.2 x the object's bounding box.
CROP_MARGIN = 1.2


@torch.no_grad()
def sample_flow(
    denoiser,
    latent: torch.Tensor,
    camera: Optional[torch.Tensor],
    cond: Dict[str, torch.Tensor],
    steps: int = 20,
    shift: float = 3.0,
    guidance: float = 3.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Integrate the (latent, camera) pair from noise (t = 1) to data (t = 0) in ``steps`` Euler steps.

    shift moves the time grid towards t = 1 (more steps at high noise)."""
    neg_cond = {name: torch.zeros_like(value) for name, value in cond.items()}
    with_camera = camera is not None

    def velocity(sample, t, condition):
        z, cam = sample
        times = torch.tensor(
            [1000 * t] * z.shape[0], device=z.device, dtype=torch.float32
        )
        v_z, v_cam = denoiser(z, cam, times, condition)
        return (v_z, v_cam) if with_camera else (v_z,)

    sample = (latent, camera) if with_camera else (latent,)
    grid = np.linspace(1, 0, steps + 1)
    t_seq = shift * grid / (1 + (shift - 1) * grid)
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        v = velocity((sample[0], sample[1] if with_camera else None), t, cond)
        if guidance is not None and guidance > 1:
            negative = velocity(
                (sample[0], sample[1] if with_camera else None), t, neg_cond
            )
            v = tuple(
                guidance * part - (guidance - 1) * neg for part, neg in zip(v, negative)
            )
        dt = t - t_prev
        sample = tuple(part - step * dt for part, step in zip(sample, v))
    return (sample[0], sample[1]) if with_camera else (sample[0], None)
