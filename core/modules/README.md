# `core/modules/`

Prompt configuration. A "module" is a Pydantic model that bundles a prompt with every
setting the call needs — model, fallbacks, sampling, output schema — into one object you
hand to `ModelRouter`.

This is the **only** place model names and fallback lists should be written. Providers
never hardcode them.

## Contents

- **`base.py`** — the `Base` class every module inherits from.
- **`test_module.py`** — `FileSummaryPrompt`, a worked example.
- **`__init__.py`** — empty.

## Usage

```python
from core.modules.base import Base
from core.llm_models.router import ModelRouter

class SummarisePrompt(Base):
    prompt: str = "Summarise the following document."
    model: str = "gemini/gemini-2.5-flash"
    fallback_models: list[str] = ["openai/gpt-4o-mini"]
    temperature: float = 0.0

module = SummarisePrompt()
response = ModelRouter(module).model_response(module)
```

Note the module is passed **twice** — once to build the router and once to the call.
The first determines the model chain, the second supplies the prompt and per-call
settings.

## Model naming

`model` and every entry in `fallback_models` must use the canonical `provider/model`
form:

```python
model: str = "anthropic/claude-opus-5"
fallback_models: list[str] = ["gemini/gemini-2.5-pro", "openai/gpt-5.5-pro"]
```

Valid prefixes are defined by `PROVIDER_ALIASES` in `core/llm_models/model_names.py`:
`gemini`/`google`, `openai`, `anthropic`/`claude`, `perplexity`/`sonar`, `ollama`,
`vllm`.

A bare name still works but logs a warning on every call, and only resolves if the model
family is one the legacy heuristic recognises. Prefixing also means **a model that does
not exist yet routes correctly**, with no change to the router and no pricing row.

Fallbacks may cross providers freely — the router builds a fresh provider per attempt.

## `base.py` — the `Base` schema

A plain `pydantic.BaseModel`. There are no validators, so field values are not checked
beyond their type hints.

### Core

| Field | Type | Default |
|---|---|---|
| `prompt` | `str \| None` | `None` |
| `system_prompt` | `str \| None` | `None` |
| `structure` | `Any \| None` | `None` — a Pydantic class to enforce as output schema |

### Model selection

| Field | Type | Default |
|---|---|---|
| `model` | `str` | `"gemini/gemini-2.5-pro"` |
| `fallback_models` | `list[str]` | `["gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"]` |

### Generation

| Field | Type | Default |
|---|---|---|
| `temperature` | `float` | `0.2` |
| `top_p` | `float` | `0.8` |
| `top_k` | `int` | `40` |
| `max_tokens` | `int \| None` | `None` |
| `reasoning_budget` | `int \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| None` | `None` |
| `return_reasoning` | `bool` | `False` — when `True`, `model_response` returns `[response, reasoning]` instead of the bare response. `reasoning` is `None` when the model did not produce a chain of thought. |

### Penalties and sampling

| Field | Type | Default |
|---|---|---|
| `presence_penalty` | `float` | `0.0` |
| `frequency_penalty` | `float` | `0.0` |
| `seed` | `int \| None` | `None` |
| `stop_sequences` | `list[str] \| None` | `None` |

### Response

| Field | Type | Default |
|---|---|---|
| `response_mime_type` | `str` | `"application/json"` |
| `stream` | `bool` | `False` |

### Logging, provider features, search

| Field | Type | Default |
|---|---|---|
| `logprobs` | `bool` | `False` |
| `top_logprobs` | `int \| None` | `None` |
| `service_tier` | `"auto" \| "default" \| None` | `None` |
| `candidate_count` | `int` | `1` |
| `safety_settings` | `Any \| None` | `None` |
| `tools` | `Any \| None` | `None` |
| `return_citations` | `bool` | `True` |
| `search_domain_filter` | `list[str] \| None` | `None` |
| `search_recency_filter` | `str \| None` | `None` |

### Portability

Not every field applies to every provider, and unsupported ones are **silently dropped
rather than erroring**:

- OpenAI reasoning models (`o1`/`o3`/`o4`/`gpt-5`) discard `temperature`, `top_p` and
  both penalties.
- `search_*` and `return_citations` only do anything on Perplexity.
- `safety_settings` and `candidate_count` are Gemini-only.
- `structure` is ignored entirely by Perplexity, and unenforced (JSON mode only) on
  Ollama.
- Setting both `stream` and `structure` disables streaming on most providers, with a
  warning.
- `return_reasoning` is supported on every provider. With streaming on it yields
  `[content_delta, reasoning_delta]` pairs instead of the raw SDK chunks; with
  streaming off it returns a two-element list. How each provider surfaces the
  chain of thought is documented in `core/llm_models/providers/README.md`.

## `test_module.py` — `FileSummaryPrompt`

```python
class FileSummaryPrompt(Base):
    prompt: str = read_prompt('test_prompt')
    model: str = 'openai/o3-mini-2025-01-31'
    stream: bool = False
```

Two things to notice:

1. `read_prompt('test_prompt')` runs at **class-definition time**, i.e. on import. A
   missing `core/prompts/test_prompt.txt` breaks the import, not the request.
2. It does not override `fallback_models`, so an OpenAI primary falls back to the
   inherited **Gemini** models. That cross-provider fallback is intentional and works,
   but it means a failure here needs `GEMINI_KEY` configured too.

## Conventions

- One class per logical task; name it after the job (`FileSummaryPrompt`,
  `InvoiceExtractPrompt`).
- Prefer `read_prompt('name')` over inline strings for anything longer than a line.
  See `core/prompts/README.md`.
- Set `structure` to a Pydantic class when you need parseable output, and check
  `providers/README.md` first — support varies sharply by provider.
- Keep business logic out of these classes. They are configuration.
