"""LibreVLA imitation fine-tuning and offline validation.

The public entry points are ``LibreVLA(...).train(data=...)`` and
``LibreVLA(...).val(data=...)``; this package is their implementation. See
``docs/adr/0028-librevla-contract.md`` and ``docs/librevla.md``.

Heavy imports (torch, lerobot) stay inside the modules so importing the
``vla`` package without the ``vla`` extra keeps working.
"""

from .data import resolve_data_source, split_episodes

__all__ = ["resolve_data_source", "split_episodes"]
