# LibreLLM

`LibreLLM` is a small client for language and vision models behind an
OpenAI-compatible API. It prepares text or image input, forwards request
arguments to the official OpenAI Python SDK, and returns the SDK's native
response object.

It is not a detector. For open-vocabulary boxes use `LibreVLM`. For a
trained detector use `LibreYOLO`.

The default API is `responses`. Use `api="chat.completions"` for providers
that implement only Chat Completions.

## Install

```bash
pip install "libreyolo[llm]"
export OPENAI_API_KEY="your-api-key"
```

## Responses API

```python
from libreyolo import LibreLLM

llm = LibreLLM("gpt-5.6-luna")
response = llm("What is YOLO?")
print(response.output_text)
```

### Multimodal input

Pass a path, HTTP URL, data URI, NumPy array, or PIL image through `image`:

```python
from libreyolo import LibreLLM

llm = LibreLLM("gpt-5.6-luna")
response = llm("What is happening in this image?", image="bus.jpg")
print(response.output_text)
```

A string passed as `source`, such as `llm("bus.jpg")`, is treated as text.
Use the `image` argument to send an image.

Local files and image objects are encoded as JPEG data URIs. NumPy arrays
use OpenCV's BGR channel order. HTTP URLs and data URIs are forwarded as-is.

Use `prompt` for an instruction shared by every request:

```python
llm = LibreLLM("gpt-5.6-luna", prompt="Answer in one sentence.")
response = llm("Describe this image.", image="bus.jpg")
```

## Chat Completions

```python
from libreyolo import LibreLLM

llm = LibreLLM("gpt-5.6-luna", api="chat.completions")
response = llm("What is non-maximum suppression?")
print(response.choices[0].message.content)
```

Pass native message objects when you need conversation history. They are
forwarded unchanged.

## Streaming and async

```python
llm = LibreLLM("gpt-5.6-luna")
for event in llm("Explain object detection.", stream=True):
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

```python
import asyncio
from libreyolo import LibreLLM

llm = LibreLLM("gpt-5.6-luna")

async def main():
    response = await llm.async_call("What is YOLO?")
    print(response.output_text)

asyncio.run(main())
```

## Other OpenAI-compatible hosts

```python
from libreyolo import LibreLLM

llm = LibreLLM(
    "provider-model",
    api="chat.completions",
    base_url="https://provider.example/v1",
    api_key="your-api-key",
)
response = llm("What tasks does YOLO support?")
print(response.choices[0].message.content)
```

The same configuration works with compatible local servers.

## Provider prefixes and model-name routing

Bare model names are remote model ids, sent to the configured (or default
OpenAI) host. That namespace stays open on purpose: a model that ships
tomorrow works today, with no libreyolo release in between.

Two refinements on top of the bare form:

- An optional `provider/` prefix supplies that provider's default host and
  env key, for symmetry with `LibreVLM`. Known prefixes: `openai/` (strip,
  default host, `OPENAI_API_KEY`) and `openrouter/` (strip,
  `https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`). An unknown prefix
  is not an error; the whole string stays the model id (OpenRouter slugs
  like `"qwen/qwen3-max"` keep working against an explicit `base_url=`).

```python
LibreLLM("gpt-5.6-luna")                 # bare form, primary
LibreLLM("openai/gpt-5.6-luna")          # synonym for the line above
LibreLLM("openrouter/qwen/qwen3-max")    # host + key from the prefix
```

- A bare name that matches a `LibreVLM` local alias (e.g.
  `"qwen3-vl-4b"`) raises instead of silently billing the default host for
  a 404: that string names a local detector, not a hosted chat model. Pass
  `base_url=` if you really run a hosted model under that exact id. This is
  a denylist against the (small, disjoint) VLM alias table, not an
  allowlist of valid provider ids.

## Combine a detector and an LLM

```python
from libreyolo import LibreLLM, LibreYOLO

yolo = LibreYOLO("LibreYOLO9t.pt")
llm = LibreLLM("gpt-5.6-luna")
image = "bus.jpg"

result = yolo(image)
if any(result.names[int(cls)] == "person" for cls in result.boxes.cls):
    response = llm("Describe the scene.", image=image)
    print(response.output_text)
```

## Constructor

| Argument | Default | Description |
|---|---|---|
| `model` | `"gpt-5.6-luna"` | Model identifier sent to the selected endpoint |
| `api` | `"responses"` | `"responses"` or `"chat.completions"` |
| `base_url` | `None` | Optional OpenAI-compatible endpoint |
| `api_key` | `None` | API key; otherwise the SDK reads `OPENAI_API_KEY` |
| `prompt` | `None` | Instruction prepended to plain text and image requests |
| `**kwargs` | | Default SDK request arguments; per-call arguments override matching constructor values |

`LibreLLM` is inference-only and is not exposed through the `libreyolo` CLI.
`train`, `val`, `export`, `track`, and `benchmark` raise.

See [`adr/0019-librellm-contract.md`](adr/0019-librellm-contract.md).
