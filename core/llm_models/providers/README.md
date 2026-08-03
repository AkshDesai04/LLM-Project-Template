# `core/llm_models/providers/`

Modular, provider-specific subpackages for each supported LLM backend (`gemini`, `openai`, `anthropic`, `perplexity`, `ollama`, `vllm`).

Every provider subpackage contains:
- `__init__.py`: Exposes the main `<ProviderName>Provider` class.
- `provider.py`: Implements the `LLMProvider` abstract base class (< 150 lines).
- `media_handler.py`: Handles media upload and payload preprocessing.
- `response_handler.py`: Handles API generation loops, streaming, thought/reasoning extraction, and cost tracking.

**Note:** You should not import these provider classes directly. `ModelRouter` builds them dynamically using absolute imports from `provider/model` names.

## Provider Subpackage Contents

| Subpackage Directory | Class | SDK Wrapped | Key Files |
|---|---|---|---|
| `gemini/` | `GeminiProvider` | `google-genai` | `provider.py`, `media_handler.py`, `response_handler.py` |
| `openai/` | `OpenAIProvider` | `openai` | `provider.py`, `media_handler.py`, `response_handler.py`, `responses_api.py`, `chat_completions.py` |
| `anthropic/` | `AnthropicProvider` | `anthropic` | `provider.py`, `media_handler.py`, `response_handler.py` |
| `perplexity/` | `PerplexityProvider` | `openai` (base URL) | `provider.py`, `media_handler.py`, `response_handler.py` |
| `ollama/` | `OllamaProvider` | `ollama` | `provider.py`, `media_handler.py`, `response_handler.py` |
| `vllm/` | `VLLMProvider` | `openai` (base URL) | `provider.py`, `media_handler.py`, `response_handler.py` |

The top-level `providers/__init__.py` re-exports all provider classes for backward compatibility.

## The Contract

Every provider class implements these four core methods defined in `core/llm_models/base_provider.py`:

```python
model_response(module, uploaded_file=None, **kwargs) -> Any
upload_media(file_bytes, mime_type) -> Any
embed_content(text, **kwargs) -> list[float] | list[list[float]]
evaluate_response(input_prompt, generated_output, rubric=None) -> JudgeResult
```

### Shared Conventions

1. **Parameter Resolution:** Parameters resolve from explicit `kwargs` first, then module attributes, then provider defaults.
2. **Retries:** Configurable max retries (default 3) with exponential / sleep backoffs.
3. **Streaming & Cost Tracking:** Usage token metadata is recorded via `cost_tracker.record_transaction()`.
4. **Absolute Imports:** All internal module dependencies enforce absolute import paths.
