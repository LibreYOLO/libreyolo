"""Training safeguards for quantization-aware fine-tuning."""

from typing import Any, Tuple


def apply_qat_training_guards(config: Any) -> Tuple[str, ...]:
    """Disable training features that can interfere with fake-quant state."""
    changed = []
    for option, safe_value in (
        ("ema", False),
        ("sync_bn", False),
        ("average_best", 0),
    ):
        if getattr(config, option, safe_value) != safe_value:
            setattr(config, option, safe_value)
            changed.append(option)
    return tuple(changed)
