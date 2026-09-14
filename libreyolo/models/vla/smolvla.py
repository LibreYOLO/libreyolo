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

from typing import ClassVar

from .lerobot_family import LeRobotPolicyFamily


class LibreSmolVLA(LeRobotPolicyFamily):
    """SmolVLA behind the LibreVLA predict / train / val surface."""

    FAMILY = "smolvla"
    FILENAME_PREFIX = "LibreSmolVLA"
    HF_REPOS: ClassVar[dict[str, str]] = {"base": "lerobot/smolvla_base"}
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "base": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"base": 512}
    UPSTREAM_TYPE = "smolvla"
    CONFIG_MODULE = "lerobot.policies.smolvla.configuration_smolvla"
    CONFIG_CLASS = "SmolVLAConfig"
    POLICY_MODULE = "lerobot.policies.smolvla.modeling_smolvla"
    POLICY_CLASS = "SmolVLAPolicy"
    LEROBOT_EXTRA = "smolvla"
