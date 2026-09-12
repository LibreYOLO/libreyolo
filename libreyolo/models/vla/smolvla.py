"""LibreYOLO adapter for SmolVLA (Hugging Face / lerobot).

SmolVLA is a 450M-parameter vision-language-action policy: a SmolVLM2
backbone with a flow-matching action expert that emits a 50-step chunk.
The upstream policy, its pre/post processor pipelines and its weights load
through the Apache-2.0 ``lerobot`` package at a pinned Hub revision. This
module is LibreYOLO's own adapter code and calls only the upstream public
API; no upstream source is ported.

Base checkpoint (``lerobot/smolvla_base``): three camera slots (``camera1``,
``camera2``, ``camera3``), a 6-dimensional state, a 6-dimensional action,
50-step chunks. Fine-tunes recorded by ``train()`` carry the dataset's own
camera names, action names and control rate in the checkpoint contract.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Optional

from .base import _INSTALL_HINT, LibreVLAModel
from .observation import Observation, frame_to_tensor

logger = logging.getLogger(__name__)

_IMAGE_PREFIX = "observation.images."


def _require_lerobot():
    """Import the upstream pieces, registering the SmolVLA config choice."""
    try:
        # Importing the configuration module registers the "smolvla" policy
        # type with the upstream config registry; PreTrainedConfig.from_pretrained
        # refuses unknown types otherwise.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return PreTrainedConfig, SmolVLAConfig, SmolVLAPolicy, make_pre_post_processors


class LibreSmolVLA(LibreVLAModel):
    """SmolVLA behind the LibreVLA predict / train / val surface."""

    FAMILY = "smolvla"
    FILENAME_PREFIX = "LibreSmolVLA"
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "base": "lerobot/smolvla_base",
    }
    HF_REVISIONS: ClassVar[Dict[str, str]] = {
        "base": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    }
    # SmolVLA resizes with padding to a 512 square; nominal, for inventory.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"base": 512}
    UPSTREAM_TYPE: ClassVar[str] = "smolvla"

    def __init__(self, size: str = "base", **kwargs):
        super().__init__(size, **kwargs)
        self._config = None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Pinned ``lerobot/smolvla_base`` snapshot tree for ``LibreSmolVLAbase``."""
        return super().get_download_url(filename)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_policy(self, snapshot_dir: str) -> None:
        _PreTrainedConfig, _SmolVLAConfig, SmolVLAPolicy, make_pre_post_processors = (
            _require_lerobot()
        )
        config = self._pretrained_config(snapshot_dir)
        config.device = str(self.device)
        policy = SmolVLAPolicy.from_pretrained(snapshot_dir, config=config)
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
        PreTrainedConfig, *_rest = _require_lerobot()
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
    def camera_slots(self) -> List[str]:
        cameras = self.contract.get("cameras") if self.contract else None
        if cameras:
            return [str(c) for c in cameras]
        keys = list(self.config.image_features)
        return [
            k[len(_IMAGE_PREFIX) :] if k.startswith(_IMAGE_PREFIX) else k for k in keys
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

    def _raw_observation(self, observation: Observation) -> Dict[str, Any]:
        """The upstream observation dict before the preprocessor pipeline."""
        import torch

        raw: Dict[str, Any] = {
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
