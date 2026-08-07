# `assets/`

Static data files loaded at runtime. Not code, not generated output.

## Contents

- **`model_pricing.csv`** — per-model token pricing, the lookup table behind all cost
  reporting.

## `model_pricing.csv`

Read once at process start by the `CostTracker` singleton
(`core/llm_models/cost_tracker.py`) and turned into an in-memory dict. The path is
resolved from `__file__`, so it works regardless of the current working directory.

### Schema

```
sr,model_id,model_provider,model_type,release_date,deprication_date,
input_cost_per_million,output_cost_per_million,
input_cost_per_million_above_200k,output_cost_per_million_above_200k,
context_caching_cost_per_million
```

| Column | Used by the loader | Meaning |
|---|---|---|
| `sr` | yes | Row number. **A row is skipped unless this is a digit.** |
| `model_id` | yes | Lookup key. Must be non-empty. |
| `model_provider` | no | Documentation only. |
| `model_type` | no | Documentation only (`llm`, `embedding`, ...). |
| `release_date` | no | Documentation only. |
| `deprication_date` | no | Documentation only. The misspelling is in the header. |
| `input_cost_per_million` | yes | USD per 1M input tokens. |
| `output_cost_per_million` | yes | USD per 1M output tokens. |
| `input_cost_per_million_above_200k` | yes | Tiered rate past 200k input tokens. |
| `output_cost_per_million_above_200k` | yes | Tiered rate past 200k output tokens. |
| `context_caching_cost_per_million` | yes | USD per 1M cached-read tokens. |

All prices are **USD per million tokens**. Empty cells and the literal `N/A` are parsed
as `0.0`, which for the two `above_200k` columns is what disables tiered pricing for
that row.

### Keys are bare model ids

`model_id` holds the bare name — `gemini-2.5-flash`, not `gemini/gemini-2.5-flash`.
The `provider/` prefix used in module configs is stripped by `ModelRouter` before the
cost tracker ever sees it, and `calculate_cost` strips it again defensively.

### How lookup works

1. Exact match on `model_id`.
2. **Longest** substring match. Longest is deliberate: a short id like `gpt-5` would
   otherwise shadow every longer id sharing its prefix, and `gpt-5.5-pro-2026-01-15`
   would bill at `gpt-5` rates. The substituted row is logged.
3. No match: a warning, and the call is reported at `$0.00`.

**A missing row never blocks a call.** The request runs normally; only its reported cost
is zero. This is what lets a newly released model be used before anyone updates this
file.

### Tiered pricing

When an `above_200k` rate is non-zero and the token count exceeds **200,000**, the first
200k bills at the base rate and the remainder at the tiered rate. Input and output are
evaluated independently. Cached tokens have no tier.

## Editing

- Append rows; keep `sr` sequential and numeric, or the row is silently skipped.
- Use the bare model id, with no provider prefix.
- Use `N/A` for rates that do not apply.
- Prices change and introductory rates expire. Treat this file as a snapshot and
  re-check against provider pricing pages before trusting a cost report.
- Beware near-identical ids. Because of substring matching, adding `foo-mini` when
  `foo-mini-preview` already exists is fine, but adding a *shorter* id that is a
  substring of existing ones can change how unlisted variants resolve.

## Verifying a change

There is no test suite for this file. To sanity-check a row:

```python
from core.llm_models.cost_tracker import cost_tracker
print(cost_tracker.calculate_cost("your-model-id", 1_000_000, 1_000_000))
```

Use a token count **below 200,000** if you want to see base rates rather than tiered
ones.
