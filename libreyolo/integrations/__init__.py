"""Optional third-party integrations.

Nothing here is imported by ``import libreyolo``. Each submodule imports its
third-party package lazily and raises a clear install hint when it is absent,
so an integration never adds an import-time cost or a hard dependency.
"""

__all__: list[str] = []
