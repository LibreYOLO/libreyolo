"""DEIM postprocessing.

DEIM's decode is code-identical to D-FINE's (the historical copy in
``models/deim/utils.py`` differed only in docstrings — verified by diff),
so this module re-exports the single implementation. DEIMv2 consumes the
same function via ``models/deimv2/utils.py``.
"""

from .dfine import postprocess  # noqa: F401

__all__ = ["postprocess"]
