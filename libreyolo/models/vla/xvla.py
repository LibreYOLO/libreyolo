"""Original LibreYOLO adapter calling X-VLA through LeRobot's public API.

Weights: lerobot/xvla-base, Apache-2.0, downloaded at a pinned revision.
The base predicts 30 timesteps in the upstream 20-dimensional ee6d action
space (two arms, each with xyz, a 6D rotation and gripper). Values are not
converted to robot joint commands by this adapter.
"""

from typing import ClassVar

from .lerobot_family import LeRobotPolicyFamily


class LibreXVLA(LeRobotPolicyFamily):
    """X-VLA camera/state/instruction policy."""

    FAMILY = "xvla"
    FILENAME_PREFIX = "LibreXVLA"
    HF_REPOS: ClassVar[dict[str, str]] = {"base": "lerobot/xvla-base"}
    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "base": "cdb7964e4fe842935d671bfab5a5ebe00a96648c",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {"base": 224}
    UPSTREAM_TYPE = "xvla"
    CONFIG_MODULE = "lerobot.policies.xvla.configuration_xvla"
    CONFIG_CLASS = "XVLAConfig"
    POLICY_MODULE = "lerobot.policies.xvla.modeling_xvla"
    POLICY_CLASS = "XVLAPolicy"
    LEROBOT_EXTRA = "xvla"

    def _prepare_training_config(self, config):
        # The base's dual-arm ee6d representation cannot stand in for a new
        # dataset's joint targets. LeRobot's auto mode retains dataset action
        # dimensions and handles the policy's internal padding itself.
        config.action_mode = "auto"
        return config
