"""Unit tests for LibreLLM (offline; the OpenAI SDK is never called)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from libreyolo.models.llm.client import LibreLLM, _load_openai

pytestmark = [pytest.mark.unit, pytest.mark.llm]


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="Hello",
                    )
                ]
            )
        return SimpleNamespace(output_text="ok")


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="chat-ok"))]
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeAsyncResponses:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="async-ok")


class FakeAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="async-chat"))]
        )


class FakeAsyncClient:
    def __init__(self):
        self.responses = FakeAsyncResponses()
        self.chat = SimpleNamespace(completions=FakeAsyncCompletions())


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(LibreLLM, "_make_sync_client", lambda self: client)
    return client


@pytest.fixture
def rgb_jpeg(tmp_path):
    path = tmp_path / "red.jpg"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path, format="JPEG")
    return path


def test_default_model_and_api(fake_client):
    llm = LibreLLM()
    response = llm("What is YOLO?")
    assert response.output_text == "ok"
    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["input"] == "What is YOLO?"


def test_chat_completions_text(fake_client):
    llm = LibreLLM("gpt-5.6-luna", api="chat.completions")
    response = llm("What is non-maximum suppression?")
    assert response.choices[0].message.content == "chat-ok"
    call = fake_client.chat.completions.calls[0]
    assert call["messages"] == [
        {"role": "user", "content": "What is non-maximum suppression?"}
    ]
    assert fake_client.responses.calls == []


def test_string_source_is_text_not_image(fake_client):
    llm = LibreLLM()
    llm("bus.jpg")
    assert fake_client.responses.calls[0]["input"] == "bus.jpg"


def test_constructor_prompt_prepended(fake_client):
    llm = LibreLLM(prompt="Answer in one sentence.")
    llm("Describe this image.")
    assert fake_client.responses.calls[0]["input"] == (
        "Answer in one sentence.\nDescribe this image."
    )


def test_prompt_only_call(fake_client):
    llm = LibreLLM(prompt="Answer in one sentence.")
    llm()
    assert fake_client.responses.calls[0]["input"] == "Answer in one sentence."


def test_empty_call_without_prompt_raises(fake_client):
    llm = LibreLLM()
    with pytest.raises(ValueError, match="source, image, or a constructor prompt"):
        llm()


def test_unknown_api_raises():
    with pytest.raises(ValueError, match="api must be"):
        LibreLLM(api="legacy")


def test_empty_model_raises():
    with pytest.raises(ValueError, match="non-empty"):
        LibreLLM("   ")


def test_http_image_is_passed_through(fake_client):
    url = "https://example.com/bus.jpg"
    llm = LibreLLM()
    llm("What is happening in this image?", image=url)
    payload = fake_client.responses.calls[0]["input"]
    assert payload[0]["role"] == "user"
    content = payload[0]["content"]
    assert content[0] == {
        "type": "input_text",
        "text": "What is happening in this image?",
    }
    assert content[1] == {"type": "input_image", "image_url": url}


def test_data_uri_image_is_passed_through(fake_client):
    uri = "data:image/jpeg;base64,AAAA"
    llm = LibreLLM()
    llm("look", image=uri)
    content = fake_client.responses.calls[0]["input"][0]["content"]
    assert content[1]["image_url"] == uri


def test_local_file_becomes_jpeg_data_uri(fake_client, rgb_jpeg):
    llm = LibreLLM()
    llm("Describe this image.", image=str(rgb_jpeg))
    content = fake_client.responses.calls[0]["input"][0]["content"]
    image_url = content[1]["image_url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    raw = base64.standard_b64decode(image_url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"


def test_pil_image_as_source(fake_client):
    llm = LibreLLM(prompt="Caption.")
    llm(Image.new("RGB", (2, 2), color=(0, 255, 0)))
    content = fake_client.responses.calls[0]["input"][0]["content"]
    assert content[0]["text"] == "Caption."
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_numpy_bgr_encodes_as_jpeg(fake_client):
    # OpenCV order: blue square.
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    arr[..., 0] = 255
    llm = LibreLLM()
    llm("color?", image=arr)
    image_url = fake_client.responses.calls[0]["input"][0]["content"][1]["image_url"]
    assert image_url.startswith("data:image/jpeg;base64,")


def test_chat_completions_image_shape(fake_client, rgb_jpeg):
    llm = LibreLLM(api="chat.completions")
    llm("Describe this image.", image=str(rgb_jpeg))
    messages = fake_client.chat.completions.calls[0]["messages"]
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_native_messages_pass_through(fake_client):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What tasks does YOLO support?"},
    ]
    llm = LibreLLM()
    llm(messages)
    assert fake_client.responses.calls[0]["input"] is messages


def test_native_messages_reject_image_kwarg(fake_client):
    llm = LibreLLM()
    with pytest.raises(ValueError, match="native message list"):
        llm([{"role": "user", "content": "hi"}], image="bus.jpg")


def test_kwargs_merge_per_call_wins(fake_client):
    llm = LibreLLM(temperature=0.1, max_output_tokens=16)
    llm("hi", temperature=0.7)
    call = fake_client.responses.calls[0]
    assert call["temperature"] == 0.7
    assert call["max_output_tokens"] == 16
    assert call["model"] == "gpt-5.6-luna"


def test_stream_passthrough(fake_client):
    llm = LibreLLM()
    events = list(llm("Explain object detection.", stream=True))
    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "Hello"
    assert fake_client.responses.calls[0]["stream"] is True


def test_async_call(monkeypatch):
    client = FakeAsyncClient()
    monkeypatch.setattr(LibreLLM, "_make_async_client", lambda self: client)
    llm = LibreLLM()

    async def run():
        return await llm.async_call("What is YOLO?")

    response = asyncio.run(run())
    assert response.output_text == "async-ok"
    assert client.responses.calls[0]["input"] == "What is YOLO?"


def test_async_chat_completions(monkeypatch):
    client = FakeAsyncClient()
    monkeypatch.setattr(LibreLLM, "_make_async_client", lambda self: client)
    llm = LibreLLM(api="chat.completions")

    async def run():
        return await llm.async_call("hi")

    response = asyncio.run(run())
    assert response.choices[0].message.content == "async-chat"


def test_client_kwargs_include_base_url_and_key(monkeypatch):
    seen = {}

    class Recording:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(
        "libreyolo.models.llm.client._load_openai",
        lambda: SimpleNamespace(OpenAI=Recording, AsyncOpenAI=Recording),
    )
    llm = LibreLLM(
        "provider-model",
        api="chat.completions",
        base_url="https://provider.example/v1",
        api_key="secret",
    )
    llm("What tasks does YOLO support?")
    assert seen == {
        "api_key": "secret",
        "base_url": "https://provider.example/v1",
    }


def test_missing_extra_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="libreyolo\\[llm\\]"):
        _load_openai()


@pytest.mark.parametrize(
    "method",
    ["train", "val", "export", "track", "benchmark"],
)
def test_cv_verbs_raise(method):
    llm = LibreLLM()
    with pytest.raises(NotImplementedError, match="inference-only"):
        getattr(llm, method)()


def test_public_import_does_not_need_openai():
    from libreyolo import LibreLLM as Exported

    assert Exported is LibreLLM
    instance = Exported("gpt-5.6-luna")
    assert instance.api == "responses"


def test_vlm_alias_denylist_raises():
    with pytest.raises(ValueError, match="LibreVLM local alias"):
        LibreLLM("qwen3-vl-4b")


def test_vlm_alias_allowed_with_base_url(fake_client):
    llm = LibreLLM("qwen3-vl-4b", base_url="http://localhost:8000/v1")
    assert llm.model == "qwen3-vl-4b"
    assert llm.base_url == "http://localhost:8000/v1"


def test_openai_prefix_strips_to_bare_model(fake_client):
    llm = LibreLLM("openai/gpt-5.6-luna")
    llm("hi")
    assert fake_client.responses.calls[0]["model"] == "gpt-5.6-luna"
    assert llm.base_url is None


def test_openrouter_prefix_sets_host_and_env_key(monkeypatch, fake_client):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm = LibreLLM("openrouter/qwen/qwen3-max")
    assert llm.model == "qwen/qwen3-max"
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.api_key == "or-key"


def test_openrouter_prefix_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LibreLLM("openrouter/qwen/qwen3-max")


def test_unknown_prefix_stays_whole_model_id(fake_client):
    llm = LibreLLM("qwen/qwen3-max", base_url="https://openrouter.ai/api/v1")
    llm("hi")
    assert fake_client.responses.calls[0]["model"] == "qwen/qwen3-max"


def test_explicit_base_url_wins_over_prefix_default(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    llm = LibreLLM("openrouter/qwen/qwen3-max", base_url="http://proxy.local/v1")
    assert llm.base_url == "http://proxy.local/v1"


def test_prefix_model_id_case_preserved(fake_client):
    llm = LibreLLM("openai/GPT-5.6-Luna")
    assert llm.model == "GPT-5.6-Luna"


def test_missing_local_image_raises(fake_client, tmp_path):
    llm = LibreLLM()
    missing = tmp_path / "nope.jpg"
    with pytest.raises(FileNotFoundError):
        llm("look", image=str(missing))


def test_source_and_image_objects_conflict(fake_client, rgb_jpeg):
    llm = LibreLLM()
    with pytest.raises(ValueError, match="not both"):
        llm(Path(rgb_jpeg), image=str(rgb_jpeg))
