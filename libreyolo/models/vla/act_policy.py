"""Original LibreYOLO adapter for ACT through LeRobot's public API.

ACT is an action policy without language input. Policy weights start from
scratch; LeRobot's default ResNet18 image backbone uses torchvision weights.
"""

from typing import ClassVar

from .lerobot_family import LeRobotPolicyFamily


class LibreACT(LeRobotPolicyFamily):
    """Train ACT on a LeRobot dataset, then predict action chunks."""

    FAMILY = "act_policy"
    FILENAME_PREFIX = "LibreACT"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"base": 224}
    PRETRAINED_BASE = False
    REQUIRES_INSTRUCTION = False
    UPSTREAM_TYPE = "act"
    CONFIG_MODULE = "lerobot.policies.act.configuration_act"
    CONFIG_CLASS = "ACTConfig"
    POLICY_MODULE = "lerobot.policies.act.modeling_act"
    POLICY_CLASS = "ACTPolicy"
    LEROBOT_EXTRA = "dataset"
