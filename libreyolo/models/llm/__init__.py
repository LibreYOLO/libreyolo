"""LibreLLM: OpenAI-compatible language and vision chat client.

This is not a detector. It forwards text (and optional images) to an
OpenAI-compatible Responses or Chat Completions endpoint and returns the
SDK's native response object. Detection stays on ``LibreVLM`` / ``LibreYOLO``.

    from libreyolo import LibreLLM

    llm = LibreLLM("gpt-5.6-luna")
    print(llm("What is YOLO?").output_text)

See ``docs/librellm.md`` and ``docs/adr/0019-librellm-contract.md``.
"""

from .client import LibreLLM

__all__ = ["LibreLLM"]
