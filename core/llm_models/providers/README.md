# `core/llm_models/providers/`

One module per LLM backend. Every file defines a single class extending `LLMProvider`
(see `../base_provider.py`) and implements the same four methods, so the rest of the
codebase can swap backends without knowing which SDK is underneath.

**You should not import these classes directly.** `ModelRouter` builds them lazily from
a `provider/model` name. Importing a provider imports its SDK, so instantiating them
by hand reintroduces the "every SDK must be installed" problem the router avoids.

## Contents

| File | Class | SDK wrapped |
|---|---|---|
| `gemini.py` | `GeminiProvider` | `google-genai` |
| `openai.py` | `OpenAIProvider` | `openai` |
| `anthropic.py` | `AnthropicProvider` | `anthropic` |
| `perplexity.py` | `PerplexityProvider` | `openai` pointed at `api.perplexity.ai` |
| `ollama.py` | `OllamaProvider` | `ollama` |
| `vllm.py` | `VLLMProvider` | `openai` pointed at a self-hosted vLLM server |

`__init__.py` is a one-line placeholder; there are no re-exports.

## The contract

Every provider implements these four methods. The abstract signatures live in
`../base_provider.py`.

```python
model_response(module, uploaded_file=None, **kwargs) -> Any
upload_media(file_bytes, mime_type) -> Any
embed_content(text, **kwargs) -> list[float] | list[list[float]]
evaluate_response(input_prompt, generated_output, rubric=None) -> JudgeResult
```

### Shared conventions

**Parameter resolution.** Every `model_response` resolves each setting in the same
order: an explicit `kwargs` value, then the attribute on the passed `module`, then the
provider default captured from the config at construction time.

```python
temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
```

**Retries.** Each provider retries `max_retries` times (default `3`, overridable via
`kwargs['max_retries']`) with a fixed `time.sleep(2)` between attempts. Exhausting the
loop raises `RuntimeError` chained to the last exception via `from last_exception`.
This is *per provider*; the router's fallback chain sits above it, so a three-model
chain can make up to nine attempts.

**Streaming.** When `stream=True` the method returns a generator, not a value. The
generator wraps the SDK's stream and records the cost transaction when the final chunk
carrying usage data arrives — so **cost is only recorded if the caller actually drains
the generator.** Abandoning a stream halfway means the call never appears in the
summary.

**Cost recording.** Providers call `cost_tracker.calculate_cost(...)` then
`cost_tracker.record_transaction(...)` with the bare model id. They never see the
`provider/` prefix; `ModelRouter` strips it before the call.

**Media.** Non-Gemini providers convert media locally through
`../utils/media_utils.py` rather than uploading it, because only Gemini has a Files API.

**`return_reasoning`.** When the flag is on, every provider returns
`[response, reasoning]` (or yields `[content_delta, reasoning_delta]` pairs while
streaming). `reasoning` is `None` when the model produced none. Shared helpers live in
`../reasoning.py`. How each provider obtains the chain of thought:

| Provider | Source |
|---|---|
| Gemini | Parts flagged `thought=True`. The flag alone forces `include_thoughts=True`. |
| OpenAI | Responses API reasoning summaries (`summary="auto"`), or `reasoning_content` / `<think>` tags on Chat Completions. |
| Anthropic | `thinking` / `redacted_thinking` content blocks. |
| Ollama | The message's `thinking` field, falling back to `<think>` tags in the content. |
| vLLM | The message's `reasoning_content` field, falling back to `<think>` tags. |
| Perplexity | `<think>` tags inline in the content (sonar-reasoning models). |

Inline `<think>` tags are stripped from the answer even when the flag is off, so they
never pollute a structured-output parse. With streaming on and the flag off, the raw
SDK chunks are still yielded unchanged.

---

## `gemini.py` — `GeminiProvider`

The only provider with two authentication paths, chosen by `GEMINI_KEY_TYPE` in `.env`.

| `GEMINI_KEY_TYPE` | Backend | Credential |
|---|---|---|
| `GEMINI_KEY` (default) | Gemini Developer API | `GEMINI_KEY` |
| `SERVICE_ACC_JSON` | Vertex AI | `GEMINI_SERVICE_ACCOUNT_FILE` or `GEMINI_SERVICE_ACCOUNT_JSON`, plus `GEMINI_PROJECT` / `GEMINI_LOCATION` |

`_build_client` constructs either `genai.Client(api_key=...)` or
`genai.Client(vertexai=True, project=..., location=..., credentials=...)`. The instance
exposes `self.uses_vertex` and `self.key_type` so other methods can branch.

- **Structured output** is native: the Pydantic class is passed straight through as
  `response_schema` on `GenerateContentConfig`, and `response.parsed` is returned.
  This is the cleanest structured-output path of any provider here.
- **Reasoning** maps a string `reasoning_budget` onto `ThinkingLevel` and sets
  `include_thoughts=True`.
- **`upload_media`** branches on auth mode. Under `GEMINI_KEY` it uploads to the Files
  API and polls while `state.name == "PROCESSING"`, raising if the state becomes
  `FAILED`. Under `SERVICE_ACC_JSON` it returns `types.Part.from_bytes(...)` instead,
  because **Vertex AI rejects the Developer Files API entirely**.
- **`embed_content`** defaults to `model="gemini-embedding-001"`, `dimensions=1536`,
  `task_type="RETRIEVAL_DOCUMENT"`. If the response carries no usage metadata it makes
  a second `count_tokens` call to price the request.
- **Judge model:** `gemini-2.0-flash`.

## `openai.py` — `OpenAIProvider`

The most complex provider, because OpenAI has two incompatible endpoints.

**Endpoint routing.** `_requires_responses_api` hard-routes `gpt-5-pro` and
`gpt-5.5-pro` to the Responses API. Any other model that rejects Chat Completions
triggers a transparent mid-attempt fallback: `_is_chat_completion_endpoint_error`
matches on `"not a chat model"` or
`"not supported in the v1/chat/completions endpoint"` and re-issues the request through
`_responses_api_generate`. `_build_responses_input` rewrites chat-style messages into
the Responses `input_text` / `input_image` shape.

**Reasoning models.** `_is_reasoning_model` matches any model *containing* `o1`, `o3`,
`o4` or `gpt-5`. For those, `temperature`, `top_p`, `presence_penalty` and
`frequency_penalty` are silently dropped and `max_tokens` is renamed to
`max_completion_tokens`, because the API rejects them. Worth knowing: your sampling
settings are ignored rather than erroring.

**Structured output** uses `beta.chat.completions.parse` with the Pydantic class as
`response_format`, returning `message.parsed`. A non-Pydantic truthy `structure` falls
back to `{"type": "json_object"}`.

- Streaming is disabled automatically when `structure` is set.
- Reads `cached_tokens` from `usage.prompt_tokens_details`.
- **`embed_content`** defaults to `model="text-embedding-3-small"`.
- **Judge model:** `gpt-4o-mini`.

## `anthropic.py` — `AnthropicProvider`

Key: `ANTHROPIC_KEY`.

- **`DEFAULT_MAX_TOKENS = 4096`** — the Messages API rejects requests without an
  explicit output cap, so one is always supplied.
- **Structured output** has no native JSON mode. It is emulated by forcing a single
  tool call: `_build_structured_tool` turns the Pydantic schema into a tool named
  `emit_structured_response` and pins `tool_choice` to it, then reads the arguments back
  and validates them.
- **System prompt** is a top-level `system` argument, not a message.
- **Reasoning:** a string `reasoning_budget` maps through `EFFORT_TOKEN_BUDGETS` onto
  explicit token budgets (`minimal`: 1024, `low`: 2048, `medium`: 4096, `high`: 8192, `xhigh`: 16384).
  Extended thinking uses `thinking={"type": "enabled", "budget_tokens": ...}`, forces
  `temperature=1`, drops `top_p`/`top_k`, and raises `max_tokens` by
  `THINKING_OUTPUT_HEADROOM = 1024`.
- **`_to_content_blocks`** converts the OpenAI-shaped payloads from `media_utils` into
  Anthropic image blocks.
- **Usage:** cached reads come from `cache_read_input_tokens`, separate from
  `input_tokens`.
- **`embed_content` raises `NotImplementedError`** — Anthropic has no embeddings API.
  Use Gemini or OpenAI.
- **Judge model:** `claude-haiku-4-5`.

## `perplexity.py` — `PerplexityProvider`

Key: `PERPLEXITY_KEY`. Uses the `openai` SDK with
`base_url="https://api.perplexity.ai"`, so no extra dependency.

Supports the search-specific settings from the module config: `search_domain_filter`,
`return_citations`, `search_recency_filter`.

Features:

- **`evaluate_response`** requests a JSON-structured `JudgeResult` response from the model,
  parses it via `json.loads`, and validates it using `JudgeResult.model_validate()`.
- **`upload_media` only handles `text/plain`.** Anything else returns the placeholder
  string `"[Media of type {mime_type} attached]"`.
- **`embed_content` raises `NotImplementedError`.**
- **Judge model:** `sonar-pro`.

## `ollama.py` — `OllamaProvider`

Local inference. Reads `OLLAMA_URL` (default `http://localhost:11434`) and `OLLAMA_KEY`
(default `"local-key"`; Ollama does not usually check it).

- **Auto-pull.** If a call fails with `ollama.ResponseError` where `status_code == 404`
  and the message contains `not found`, the provider pulls the model from Ollama Hub and
  retries. This happens **once per call** (guarded by `pull_attempted`) and without the
  usual 2-second sleep. A first call against an unpulled model can therefore block for a
  long time while several gigabytes download.
- **Structured output is schema-less.** It can only set `format='json'`; the schema is
  not enforced server-side. The response is parsed with `json.loads` and validated
  against the Pydantic class. **If parsing fails on the final attempt the raw string is
  returned rather than a model instance** — callers using `structure` should be ready
  for a `str`. The same trapdoor exists in `vllm.py`.
- Streaming is disabled automatically when `structure` is set.
- **Images** are passed as raw bytes on the message's `images` key; base64 data URLs
  from `media_utils` are decoded back to bytes first.
- **`upload_media`** returns raw bytes for images (not the base64 dict other providers use).
- **`embed_content`** loops one call per text (no batching) and records a
  zero-token transaction, since Ollama reports no usage for embeddings.
- **Judge model:** whatever `self.model_name` is.

## `vllm.py` — `VLLMProvider`

Client for a self-hosted vLLM OpenAI-compatible server. Reads `VLLM_URL`
(`DEFAULT_VLLM_URL = "http://localhost:8000/v1"`) and `VLLM_KEY`
(`PLACEHOLDER_API_KEY = "EMPTY"`, since vLLM only checks the key when started with
`--api-key`).

**The heavyweight `vllm` package is never imported** — this is a client only, so no GPU
runtime is required to use it.

- **Structured output** uses the standard `response_format` json_schema field.
  The legacy `guided_json` extra was removed in vLLM v0.12.0.
- vLLM-only sampling knobs (e.g. `repetition_penalty`) are forwarded through
  `extra_body`.
- Same JSON-parse trapdoor as Ollama: a parse failure on the last attempt returns the
  raw string.
- Because a vLLM server exposes arbitrary model names, models **must** carry the
  explicit `vllm/` prefix. Note that `vllm/meta-llama/Llama-3.1-8B` correctly yields
  `meta-llama/Llama-3.1-8B` — only the first path segment is consumed.
- **Judge model:** `self.model_name` (a vLLM server hosts one model, so it judges itself).

---

## Comparison

| | Structured output | Embeddings | Streaming | Native file upload |
|---|---|---|---|---|
| Gemini | native `response_schema` | yes | yes | yes (API-key mode only) |
| OpenAI | `beta...parse` | yes | yes | no (local extraction) |
| Anthropic | forced tool call | **no** | yes | no (local extraction) |
| Perplexity | **no** | **no** | yes | **no** (placeholder) |
| Ollama | `format='json'`, unenforced | yes | yes | no (local extraction) |
| vLLM | `response_format` json_schema | yes (if server hosts one) | yes | no (local extraction) |

## Adding a provider

1. Create `newprovider.py` with a class extending `LLMProvider` and implementing all
   four abstract methods.
2. Resolve credentials with `get_secret("NEWPROVIDER_KEY")` from `utils.env_ops`, never
   `os.getenv` — that is what makes `KEY_LOCATION=AWS_SM` work.
3. Record usage through `cost_tracker` on both the streaming and non-streaming paths.
4. Add the prefix to `PROVIDER_ALIASES` in `../model_names.py`.
5. Add a lazy `if provider == 'newprovider':` branch to `ModelRouter._build_provider`.
   Keep the import inside the branch so an uninstalled SDK never breaks other providers.
6. Add pricing rows to `assets/model_pricing.csv` (optional — an unpriced model still
   runs, it just reports `$0.00`).
