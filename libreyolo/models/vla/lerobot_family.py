"""Shared adapter over LeRobot's public policy and processor APIs.

Original LibreYOLO adapter code; no upstream implementation is ported.
Families declare their config registration module and policy class and may
 override any hook for differences in observations, loading or prediction.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, ClassVar

from .base import _INSTALL_HINT, LibreVLAModel
from .observation import Observation, frame_to_tensor

_IMAGE_PREFIX = "observation.images."


class LeRobotPolicyFamily(LibreVLAModel):
    """Common lazy loading and observation contract for LeRobot families."""

    UPSTREAM_TYPE: ClassVar[str] = ""
    CONFIG_MODULE: ClassVar[str] = ""
    CONFIG_CLASS: ClassVar[str] = ""
    POLICY_MODULE: ClassVar[str] = ""
    POLICY_CLASS: ClassVar[str] = ""
    LEROBOT_EXTRA: ClassVar[str] = ""

    @classmethod
    def _require_lerobot(cls):
        """Register the family before deserializing its policy config."""
        try:
            configuration = import_module(cls.CONFIG_MODULE)
            config_class = getattr(configuration, cls.CONFIG_CLASS)
            policy_class = getattr(import_module(cls.POLICY_MODULE), cls.POLICY_CLASS)
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"{_INSTALL_HINT}\n{cls.FAMILY} also requires "
                f"pip install 'lerobot[{cls.LEROBOT_EXTRA}]>=0.6.1'."
            ) from exc
        return PreTrainedConfig, config_class, policy_class, make_pre_post_processors

    def _scratch_config(self, meta):
        """Create a default policy config; the factory fills dataset features."""
        _, config_class, *_ = self._require_lerobot()
        return config_class()

    def __init__(self, size: str = "base", **kwargs):
        super().__init__(size, **kwargs)
        self._config = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_policy(self, snapshot_dir: str) -> None:
        _PreTrainedConfig, _Config, Policy, make_pre_post_processors = (
            self._require_lerobot()
        )
        config = self._pretrained_config(snapshot_dir)
        config.device = str(self.device)
        policy = Policy.from_pretrained(snapshot_dir, config=config)
        policy.eval()
        policy.to(self.device)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=snapshot_dir,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self._config = config
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self.reset()

    # ------------------------------------------------------------------
    # Contract properties
    # ------------------------------------------------------------------

    def _pretrained_config(self, snapshot_dir: str):
        PreTrainedConfig, *_rest = self._require_lerobot()
        config = PreTrainedConfig.from_pretrained(snapshot_dir)
        if getattr(config, "type", None) != self.UPSTREAM_TYPE:
            raise ValueError(
                f"{snapshot_dir} holds a {getattr(config, 'type', None)!r} policy, "
                f"not {self.UPSTREAM_TYPE!r}."
            )
        return config

    @property
    def config(self):
        """The upstream policy config (loads the policy on first access)."""
        if self._config is None:
            self._ensure_loaded()
        return self._config

    @property
    def camera_slots(self) -> list[str]:
        cameras = self.contract.get("cameras") if self.contract else None
        if cameras:
            return [str(c) for c in cameras]
        keys = list(self.config.image_features)
        return [
            k.removeprefix(_IMAGE_PREFIX) for k in keys
        ]

    @property
    def state_dim(self) -> int:
        feature = self.config.robot_state_feature
        return int(feature.shape[0]) if feature is not None else 0

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    @property
    def chunk_size(self) -> int:
        return int(self.config.chunk_size)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _raw_observation(self, observation: Observation) -> dict[str, Any]:
        """The upstream observation dict before the preprocessor pipeline."""
        import torch

        raw: dict[str, Any] = {
            "observation.state": torch.as_tensor(
                observation.state, dtype=torch.float32
            ),
            "task": observation.instruction,
        }
        for slot, image in observation.frames.items():
            raw[f"{_IMAGE_PREFIX}{slot}"] = frame_to_tensor(image)
        raw.update(observation.extras)
        return raw

    def _predict_chunk(self, observation: Observation) -> Any:
        import torch

        batch = self._preprocessor(self._raw_observation(observation))
        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(batch)
        chunk = self._postprocessor(chunk)
        return chunk[:, :, : self.action_dim]
