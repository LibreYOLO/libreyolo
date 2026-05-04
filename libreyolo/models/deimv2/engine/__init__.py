"""Vendored DEIMv2 engine modules used by the LibreYOLO wrapper.

Keep this package initializer intentionally small. The upstream initializer
eagerly imports training/data modules that are not needed for inference and can
pull optional dependencies into a plain model load.
"""
