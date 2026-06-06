"""Local drag-and-drop inference UI for LibreYOLO.

A zero-dependency web UI served from Python's standard-library HTTP server.
Launched via ``libreyolo ui``. See :mod:`libreyolo.ui.server`.
"""

from .server import serve

__all__ = ["serve"]
