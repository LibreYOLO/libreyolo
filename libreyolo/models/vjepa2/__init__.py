"""LibreVJEPA2 family: V-JEPA 2.0 video encoder and attentive-probe classifier.

See ``NOTICE`` in this directory for code and weight provenance. The encoder
port is adapted from Apache-2.0 Hugging Face Transformers and is not
relicensed as MIT.
"""

from .model import LibreVJEPA2
from .nn import (
    VJEPA2_CONFIGS,
    LibreVJEPA2Classifier,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)

__all__ = [
    "VJEPA2_CONFIGS",
    "LibreVJEPA2",
    "LibreVJEPA2Classifier",
    "LibreVJEPA2Encoder",
    "VJEPA2Config",
]
