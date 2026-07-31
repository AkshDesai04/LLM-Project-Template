# `core/llm_models/`

The LLM engine. Everything about talking to a model provider, choosing between
providers, falling back when one fails, and accounting for what it all cost lives here.

## Contents

| Path | Purpose |
|---|---|
| `router.py` | `ModelRouter` — the entry point. Picks a provider, walks the fallback chain. |
| `model_names.py` | Owns the `provider/model` naming convention. |
| `base_provider.py` | `LLMProvider` abstract base + the `JudgeResult` schema. |
| `cost_tracker.py` | Singleton recording tokens and cost; prints a summary at exit. |
| `providers/` | One module per backend. See `providers/README.md`. |
| `utils/` | Offline media decoding. See `utils/README.md`. |

## How a call flows

```
YourModule(Base)                    model = "gemini/gemini-2.5-flash"
      |                             fallback_models = ["openai/gpt-4o", ...]
      v
ModelRouter(module)                 builds the ordered, de-duplicated chain
      |
      v
split_model_name()                  "gemini/gemini-2.5-flash" -> ("google", "gemini-2.5-flash")
      |
      v
_build_provider()                   lazily imports + constructs GeminiProvider
      |
      v
provider.model_response()           retries up to 3x internally
      |
      +-- success --> cost_tracker.record_transaction()   -> return
      |
      +-- failure --> cost_tracker.record_failed_attempt() -> next model in chain
```

If every model in the chain fails, the router raises `RuntimeError` chained to the last
exception.

## `router.py` — `ModelRouter`

**Always call models through this class.** Direct provider instantiation skips the
fallback chain and forces you to import an SDK you may not have installed.

### `__init__(module: BaseModule, fallback_index: int = 0)`

Builds `self._model_chain` as `[module.model] + module.fallback_models`, preserving
order and dropping duplicates, then slices from `fallback_index`. Raises `ValueError`
if the index is out of range.

Constructing the primary provider is **best-effort**: it is wrapped in `try/except` and
only logs a warning on failure. This matters because a missing `GEMINI_KEY` should not
stop an OpenAI fallback from running. `self.model_instance` stays `None` until something
succeeds, and `_require_model_instance()` builds it on demand for the proxy methods.

### Key methods

- **`model_response(module, uploaded_file=None, **kwargs)`** — walks the chain. Builds a
  *fresh* provider per entry, which is what makes cross-provider fallback work. On
  failure it records a zero-token failed attempt and continues.
  Passing an explicit `kwargs['model']` **bypasses the chain entirely** and calls that
  one model.
- **`get_provider_by_model_name(name)`** *(static)* — reads the provider off the prefix.
  Falls through to `_infer_provider_from_bare_name` for unprefixed names.
- **`_infer_provider_from_bare_name(name)`** *(static)* — legacy compatibility. Guesses
  from the model family (`gpt`/`o1`/`o3`/`o4` → openai, `gemini` → google, and so on) and
  logs a warning naming the explicit form to use. Raises `ValueError` with a suggestion
  if it cannot guess.
- **`strip_routing_prefix(name)`** *(static)* — delegates to `split_model_name`.
- **`_build_provider(name, module)`** — resolves the provider, strips the prefix, and
  copies the module with `model_copy(update={'model': stripped})` so the provider never
  sees the prefix. **Every SDK import is inside its own branch**, so an uninstalled
  provider never breaks the others.
- **`upload_media` / `embed_content` / `evaluate_response`** — thin proxies onto the
  current provider. Note these do **not** walk the fallback chain.

## `model_names.py`

Owns the naming convention, in one place, because both the router and the cost tracker
need it and the cost tracker is imported *by* the router (so it cannot import back).

```python
PROVIDER_ALIASES = {
    "gemini": "google",  "google": "google",
    "openai": "openai",
    "anthropic": "anthropic", "claude": "anthropic",
    "perplexity": "perplexity", "sonar": "perplexity",
    "ollama": "ollama",
    "vllm": "vllm",
}
```

**`split_model_name(name) -> (provider | None, model)`** splits on the *first* slash, but
only when the prefix is a known alias. Two consequences worth internalising:

- `meta-llama/Llama-3.1-8B` is left completely intact — `meta-llama` is not an alias, so
  it is treated as part of the model id.
- `vllm/meta-llama/Llama-3.1-8B` yields `("vllm", "meta-llama/Llama-3.1-8B")`.

**`strip_provider_prefix(name) -> str`** returns just the model id.

### Why the prefix exists

Routing used to guess the provider from the model family, so a model whose name did not
match a known pattern could not be called at all. Declaring the provider means **a model
released after this code was written routes correctly with no code change and no pricing
row.** An unpriced model is reported at `$0.00` with a warning; the call itself is
unaffected.

## `base_provider.py`

### `JudgeResult`

The schema for LLM-as-a-judge evaluation.

| Field | Type | Default |
|---|---|---|
| `score` | `int` | required — 1 to 10 |
| `reasoning` | `str` | required |
| `improvements` | `Optional[str]` | `None` |

### `LLMProvider(ABC)`

`__init__(api_key, base_config)` flattens the module config onto `self`. Five attributes
are read directly and will raise `AttributeError` if missing (`model`, `temperature`,
`top_p`, `top_k`, plus `api_key`); the rest use `getattr` with a fallback, e.g.
`self.max_tokens = getattr(base_config, 'max_tokens', None)`.

It deliberately does **not** capture `prompt`, `fallback_models`, `reasoning_budget` or
`return_reasoning`. Fallback selection is the router's job, and prompts arrive per-call.

**`prepare_module(module)`** *(static)* normalises input and lazily loads prompts:
accepts either a class (which it instantiates) or an instance; raises `TypeError`
otherwise. If `prompt` is empty but the module defines a `prompt_path`, it reads that
file into `prompt`. `prompt_path` is opt-in — `Base` does not declare it.

**Abstract methods** every provider must implement: `model_response`, `upload_media`,
`embed_content`, `evaluate_response`.

## `cost_tracker.py`

A process-wide singleton, exported as `cost_tracker`. State lives on `_instance` and is
initialised in `__new__`, so re-constructing `CostTracker()` never resets totals.

### Pricing

`_load_pricing()` reads `assets/model_pricing.csv` (path resolved from `__file__`, so it
is CWD-independent) into `Dict[str, Dict[str, float]]` keyed by the bare `model_id`.
Rows are skipped unless `sr` is a digit and `model_id` is non-empty. Empty cells and the
literal `N/A` become `0.0`. The whole load is wrapped in `try/except`, so a missing CSV
degrades to an empty dict rather than breaking import.

### `calculate_cost(model_name, prompt_tokens, output_tokens, cached_tokens=0) -> dict`

Lookup order:

1. Strip any known `provider/` prefix.
2. Exact match on the pricing key.
3. **Longest** substring match — deliberately longest, so a short id like `gpt-5` cannot
   shadow `gpt-5.5-pro-2026-01-15`. Logs which row it substituted.
4. No match: warn and return all zeros. The call is never blocked.

Tiered pricing kicks in above a **200,000-token threshold**, applied independently to
input and output, and only when an above-tier rate is present. The first 200k always
bills at the base rate. Cached tokens have no tier.

Returns `{"input_cost", "output_cost", "cached_cost", "total_cost"}`.

### Recording

- **`record_transaction(module_name, model_name, costs, duration, input_tokens=0, output_tokens=0, cached_tokens=0, status="success")`**
  appends to history and updates the running totals **only when `status == "success"`**.
- **`record_failed_attempt(module_name, model_name, duration, error=None)`** appends a
  row with zeroed tokens and costs, `status="failed"`, and the stringified error. Totals
  are untouched.

### `print_final_summary()`

Registered with `atexit` during initialisation, so it fires automatically at interpreter
exit. Idempotent, and a no-op when no calls were made.

Columns: `SR.`, `MODULE`, `MODEL`, `WALL TIME`, `IN TOK`, `OUT TOK`, `CACHE TOK`,
`INPUT`, `OUTPUT`, `CACHED`, `TOTAL`. Failed rows are labelled `model-name (failed)`.

One subtlety in the `AVERAGE` row: wall time is divided by **all** calls, while tokens
and costs are divided by **successful** calls only. Failed attempts therefore drag the
average latency up without diluting the average cost.

It writes no file of its own — it emits two `logger.info` records (`session_history` and
`session_totals`), which land wherever `utils/logger.py` is configured to send them.

## Gotchas

- **Streaming defers cost recording** to the moment the final usage chunk is consumed.
  Abandon the generator and the call never appears in the summary.
- **Retries multiply.** Each provider retries 3× internally *and* the router walks the
  chain, so a 3-model chain can make 9 attempts before raising.
- **Proxy methods skip fallback.** `embed_content` and friends use whichever provider is
  currently loaded.
- **`record_failed_attempt` gets the stripped name**, so failed and successful rows for
  the same model share an id in the summary table.
