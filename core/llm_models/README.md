# `core/llm_models/`

The LLM engine. Everything about talking to a model provider, choosing between providers, falling back when one fails, and accounting for execution cost lives here.

## Contents

| Path | Purpose |
|---|---|
| `router.py` | `ModelRouter` — the entry point. Picks a provider, walks the fallback chain. |
| `model_names.py` | Owns the `provider/model` naming convention. |
| `reasoning.py` | Shared helpers for the `return_reasoning` flag: tag stripping, stream splitting, and the `[response, reasoning]` return shape. |
| `base_provider.py` | `LLMProvider` abstract base + the `JudgeResult` schema. |
| `cost_tracker.py` | Singleton recording tokens and cost metrics. |
| `cost_summary.py` | Formats and outputs transaction summaries upon program completion. |
| `providers/` | Modular provider subpackages (`gemini`, `openai`, `anthropic`, `perplexity`, `ollama`, `vllm`). |
| `utils/` | Media decoding and processing utilities. |

## Structural Rules & Guidelines

1. **200 Lines Limit:** No Python file in this module or its subpackages exceeds 200 lines.
2. **Absolute Imports:** Relative imports (`from .` or `from ..`) are strictly forbidden. All imports must be absolute.
3. **Uniform Provider Architecture:** All model provider implementations under `providers/` follow the exact same modular directory layout (`__init__.py`, `provider.py`, `media_handler.py`, `response_handler.py`).
