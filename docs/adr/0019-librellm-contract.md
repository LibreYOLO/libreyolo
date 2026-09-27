# ADR 0019: LibreLLM Contract For OpenAI-Compatible Chat

- Status: Accepted
- Date: 2026-08-15
- Scope: New sibling client (hosted language and vision chat over
  OpenAI-compatible HTTP APIs)

## Context

LibreYOLO already has `LibreVLM` (ADR 0002) for generative models used as
open-vocabulary detectors. That contract is `set_classes` plus `predict` /
`track` returning `Results`.

Hosted chat models do not fit it. They take text, optionally an image, and
return a provider response object. There are no boxes and no confidence.
Forcing them through `LibreVLM` would make `predict()` sometimes mean
"detect" and sometimes mean "chat".

The de-facto ecosystem already published a small LLM interface for this job.
The public call shape (constructor arguments, `__call__(source, image=)`,
`async_call`, native SDK response) is the contract users will expect. This
ADR records that shape. Implementation talks to the official OpenAI Python
SDK (Apache-2.0) and to that SDK's documented Responses / Chat Completions
APIs. No third-party CV library source was read.

## Decision

Add `LibreLLM` as a sibling factory, not a `BaseModel` and not a detector.

- `LibreVLM` stays the detector. Image in, `Results` out.
- `LibreLLM` is the chat client. Text in, optional image, native SDK
  response out. Vision is an optional `image=` field on the same object
  because that is how the wire protocol works.

```python
from libreyolo import LibreLLM, LibreYOLO

llm = LibreLLM("gpt-5.6-luna")
print(llm("What is YOLO?").output_text)

response = llm("What is happening in this image?", image="bus.jpg")

llm = LibreLLM("gpt-5.6-luna", api="chat.completions")
print(llm("What is NMS?").choices[0].message.content)

llm = LibreLLM(
    "provider-model",
    api="chat.completions",
    base_url="https://provider.example/v1",
    api_key="...",
)
```

## Public API

| Argument | Default | Meaning |
|---|---|---|
| `model` | `"gpt-5.6-luna"` | Identifier sent to the endpoint |
| `api` | `"responses"` | `"responses"` or `"chat.completions"` |
| `base_url` | `None` | OpenAI-compatible host |
| `api_key` | `None` | Else the SDK reads `OPENAI_API_KEY` |
| `prompt` | `None` | Sticky instruction prepended to plain text and image requests |
| `**kwargs` | | Default SDK request arguments; per-call values win |

Call rules:

- `source` is text, a native message list, or an image object.
- A string such as `"bus.jpg"` is text, not an image.
- `image=` accepts a path, HTTP URL, data URI, NumPy array (OpenCV BGR), or
  PIL image. Local files and image objects become JPEG data URIs. HTTP URLs
  and data URIs pass through (the client does not download them).
- Native message lists are forwarded unchanged and cannot be combined with
  `image=`.
- Calling the instance with only a constructor `prompt` sends that prompt.
- `stream=True` and other SDK request arguments pass through.
- `async_call` is the async counterpart.
- Return value is the SDK object (`output_text` on Responses,
  `choices[0].message.content` on Chat Completions).

Inference only. `train`, `val`, `export`, `track`, and `benchmark` raise
`NotImplementedError`. There is no CLI verb.

Install extra: `pip install "libreyolo[llm]"`.

## Out of scope

- Remote detection (`set_classes` + boxes). That remains a `LibreVLM`
  transport if it lands later.
- Provider-native Gemini / Anthropic transports.
- Cost estimation, batch APIs, auto-label.

## Consequences

- Users get one OpenAI-compatible client that reaches any host speaking
  Responses or Chat Completions, including localhost.
- The detector contract stays honest.
- The `openai` package is an optional extra, not a core dependency.

## Amendment (2026-08-15): name routing locked

Locked alongside the remote `LibreVLM` transport (ADR 0020):

- Bare model names stay remote model ids forever. There is no closed
  allowlist of valid provider ids: it could never be maintained, and it
  would break every newly shipped remote model until a release.
- The billing-accident protection is a denylist: a bare name matching a
  `LibreVLM` local alias raises with a pointer to `LibreVLM`. `base_url=`
  is the escape hatch (the name becomes opaque and is sent there).
- Optional known `provider/` prefixes (`openai/`, `openrouter/`) strip and
  supply the provider's default host and env key. Unknown prefixes leave
  the whole string as the model id (unchanged behavior). The bare form
  stays primary.
