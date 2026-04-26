"""Tests for default Python API logging behavior."""

import logging

import pytest

from libreyolo.utils.logging import ensure_default_logging

pytestmark = pytest.mark.unit


def test_ensure_default_logging_installs_handler_when_unconfigured():
    logger = logging.getLogger("libreyolo")
    root_logger = logging.getLogger()

    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_root_handlers = list(root_logger.handlers)

    try:
        logger.handlers.clear()
        root_logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

        ensure_default_logging()

        assert logger.handlers
        assert logger.level == logging.INFO
        assert logger.propagate is False
    finally:
        logger.handlers.clear()
        logger.handlers.extend(old_handlers)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        root_logger.handlers.clear()
        root_logger.handlers.extend(old_root_handlers)


def test_ensure_default_logging_respects_existing_root_handlers():
    logger = logging.getLogger("libreyolo")
    root_logger = logging.getLogger()

    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_root_handlers = list(root_logger.handlers)
    sentinel = logging.StreamHandler()

    try:
        logger.handlers.clear()
        root_logger.handlers.clear()
        root_logger.addHandler(sentinel)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

        ensure_default_logging()

        assert logger.handlers == []
        assert logger.propagate is True
    finally:
        logger.handlers.clear()
        logger.handlers.extend(old_handlers)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        root_logger.handlers.clear()
        root_logger.handlers.extend(old_root_handlers)
