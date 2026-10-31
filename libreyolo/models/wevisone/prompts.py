# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/Tencent/WeVisOne-VL (wevisone_vl/prompts.py,
# wevisone_vl/engine.py) at commit 9a6bcd85404505df0d05cfb5d97512ccacf35dee.
# Copyright (C) 2026 Tencent. All rights reserved.
# Modified for LibreYOLO: only the instructions this adapter issues are kept,
# each as a function over a LibreYOLO class-name list.
"""Instruction strings the released WeVisOne-VL checkpoints were tuned on."""

from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = "You are a helpful assistant."


def _phrase(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".?!" else text + "."


def _labels(labels: Sequence[str]) -> list[str]:
    return [str(label).strip() for label in labels if str(label).strip()]


def detection_prompt(labels: Sequence[str]) -> str:
    """Open-vocabulary detection of every listed category in one generation."""
    return (
        "Locate all the instances that match the following description: "
        + ". ".join(_labels(labels))
        + "."
    )


def referring_prompt(expression: str) -> str:
    """Referring-expression localization; the model answers ``None`` if absent."""
    return "Locate the instances that match the following description: " + _phrase(
        expression
    )


def point_prompt(labels: Sequence[str]) -> str:
    return "point to: " + ". ".join(_labels(labels))


def matting_prompt(expression: str) -> str:
    """Boxes followed by ``<OBJ>`` tokens that the pixel decoder turns into
    alpha mattes. Upstream uses the same instruction for segmentation."""
    return (
        "Locate the instances that match the following description and "
        "provide the alpha mattes: " + _phrase(expression)
    )


OCR_PROMPT = "detect all the word in the box format"
