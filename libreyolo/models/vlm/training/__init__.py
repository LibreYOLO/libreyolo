"""VLM detection fine-tuning: dataset rendering, collation, and the trainer.

The public entry point is ``LibreVLM(...).train(data=...)``; this package is
its implementation. See ``docs/adr/0002-librevlm-contract.md`` for the tier
contract and ``docs/vlm_training.md`` for the training design.

Heavy imports (torch, peft) stay inside the modules so importing the ``vlm``
package without the ``vlm-train`` extra keeps working.
"""

from .checkpoint import CONTRACT_FILENAME, is_vlm_checkpoint, read_contract
from .targets import FamilyFormat, serialize_detections

__all__ = [
    "CONTRACT_FILENAME",
    "FamilyFormat",
    "is_vlm_checkpoint",
    "read_contract",
    "serialize_detections",
]
