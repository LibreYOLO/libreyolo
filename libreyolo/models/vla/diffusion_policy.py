"""Original adapter for LeRobot's diffusion action policy public API."""

from collections import deque
from typing import ClassVar

from .lerobot_family import LeRobotPolicyFamily


class LibreDiffusionPolicy(LeRobotPolicyFamily):
    """Diffusion Policy with a rolling history of camera/state observations."""

    FAMILY = "diffusion_policy"
    FILENAME_PREFIX = "LibreDiffusionPolicy"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"base": 224}
    PRETRAINED_BASE = False
    REQUIRES_INSTRUCTION = False
    UPSTREAM_TYPE = "diffusion"
    CONFIG_MODULE = "lerobot.policies.diffusion.configuration_diffusion"
    CONFIG_CLASS = "DiffusionConfig"
    POLICY_MODULE = "lerobot.policies.diffusion.modeling_diffusion"
    POLICY_CLASS = "DiffusionPolicy"
    LEROBOT_EXTRA = "diffusion"

    @property
    def chunk_size(self):
        return int(self.config.n_action_steps)

    def reset(self):
        super().reset()
        self._observation_history = None

    def _raw_observation(self, observation):
        import torch

        raw = super()._raw_observation(observation)
        count = int(self.config.n_obs_steps)
        history = getattr(self, "_observation_history", None)
        if history is None:
            history = deque(maxlen=count)
            self._observation_history = history
        if history and set(raw) != set(history[-1]):
            raise ValueError(
                "Diffusion Policy camera keys changed; call reset() between episodes."
            )
        history.append(raw)
        while len(history) < count:
            history.appendleft(raw)
        return {
            key: torch.stack([frame[key] for frame in history]).unsqueeze(0)
            if key.startswith("observation.")
            else value
            for key, value in raw.items()
        }

    def _validation_targets(self, target, pad, steps):
        start = int(self.config.n_obs_steps) - 1
        stop = start + steps
        if stop > target.shape[1]:
            raise ValueError(
                "Recorded diffusion action horizon is shorter than the prediction."
            )
        return target[:, start:stop], pad[:, start:stop] if pad is not None else None
