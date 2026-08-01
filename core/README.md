# `core/`

The application layer: everything that defines *what* to ask a model and *how* the
request reaches a provider. Generic infrastructure with no LLM knowledge lives in
`utils/` instead.

## Layout

| Directory | Purpose |
|---|---|
| `llm_models/` | The LLM engine — routing, providers, cost tracking. |
| `modules/` | Prompt configuration classes. The only place model names belong. |
| `prompts/` | Plain-text prompt bodies loaded by `read_prompt`. |
| `scripts/` | Runnable entry points and manual harnesses. |

Each has its own README with full detail.

## The mental model

Three layers, each ignorant of the one above it:

```
core/modules/     WHAT to ask     - prompt, model choice, fallbacks, sampling, schema
core/llm_models/  HOW to send it  - provider selection, retries, fallback, cost
core/prompts/     the prompt text
```

A module is a Pydantic object. You hand it to `ModelRouter`, which resolves the provider
from the model name, constructs the right SDK client, and calls it — falling back down a
chain if anything fails.

## Minimal example

```python
from core.modules.base import Base
from core.llm_models.router import ModelRouter

class SummarisePrompt(Base):
    prompt: str = "Summarise the following document."
    model: str = "gemini/gemini-2.5-flash"
    models: list[str] = ["openai/gpt-4o-mini"]

module = SummarisePrompt()
router = ModelRouter(module)
print(router.model_response(module))
```

With a file attached:

```python
from utils.file_ops import get_file

media = router.upload_media(get_file("report.pdf"), "application/pdf")
print(router.model_response(module, uploaded_file=media))
```

## Two rules

**1. Always go through `ModelRouter`.** Never instantiate a provider directly. The
router is what gives you the fallback chain, and it imports SDKs lazily — so you do not
need every provider's package installed to use one of them.

**2. Model names live in `core/modules/`, never in provider code.** They use the
canonical `provider/model` form, e.g. `gemini/gemini-2.5-flash`. Because the provider is
declared rather than guessed, a model released after this code was written works with no
code change and no pricing-CSV row.

## Adding a feature

| Goal | Where |
|---|---|
| New prompt/task | New class in `core/modules/`, text in `core/prompts/` |
| New LLM backend | New file in `core/llm_models/providers/`, plus an alias in `model_names.py` and a branch in `router.py` |
| New model on an existing backend | Nothing — just use `provider/new-model-name`. Add a pricing row if you want cost tracked. |
| New media type | `core/llm_models/utils/media_utils.py` and each provider's `upload_media` |
| Generic helper with no LLM knowledge | `utils/`, not here |

## Note on `core/workflows/`

This directory is empty. It held a LangGraph judge graph that was removed when
`langchain`/`langgraph` were dropped from the project. Git does not track empty
directories, so it exists only in this working copy and can be deleted.

## Packaging

`core/`, `core/llm_models/`, `core/llm_models/providers/`, `core/llm_models/utils/`,
`core/modules/` and `core/scripts/` all carry an `__init__.py`. They are empty or
one-line placeholders with no re-exports, so always import from the concrete module:

```python
from core.llm_models.router import ModelRouter   # yes
from core.llm_models import ModelRouter          # no
```

`core/prompts/` has no `__init__.py` — it is data, not a package.
