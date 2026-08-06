# Integration Testing Suite

This directory contains integration test suites for LLM providers integrated into the repository architecture (`OpenAI` and `Google Gemini`).

---

## 🚀 Running Tests in Parallel (Concurrency = 8)

To execute all OpenAI integration test cases concurrently across worker threads with high throughput and immediate financial metric logging, run the following command from the workspace root:

### PowerShell / Windows Terminal
```powershell
python -m tests.run_parallel_openai_tests
```

### Bash / Linux / macOS
```bash
python -m tests.run_parallel_openai_tests
```

---

## 🎯 OpenAI Test Suite Architecture

### 1. Cost & Model Selection Constraints
To prevent unnecessary API expenses, models with high output costs (**> $40 / 1M tokens**) or `pro` tier pricing are **strictly excluded**:
- ❌ **Excluded Models**: `gpt-5.4-pro` ($180/M output), `gpt-5.2-pro` ($168/M output), `gpt-5-pro` ($120/M output), `o1-pro` ($600/M output), `o1` ($60/M output), `gpt-4-0613` ($60/M output).
- ✅ **Included Models**: `gpt-5.4` ($15/M output), `gpt-4o` ($10/M output), `gpt-4o-mini` ($0.60/M output), `o3-mini` ($4.40/M output), `o4-mini` ($4.40/M output), `gpt-5-mini` ($2.00/M output).

---

### 2. Test Modules Breakdown

| Test Module File | Test Suite Name | Description & Coverage |
| :--- | :--- | :--- |
| `tests/test_live_openai_models.py` | `Models: OpenAI Standard` | Validates `gpt-5.4`, `gpt-4o`, `gpt-4o-mini`, and `gpt-5-mini` across short (~15t) & large (~2,500t) scale inputs and Pydantic schema validation (`CapitalInfo`). |
| | `Models: OpenAI Reasoning` | Validates reasoning budget (`low`, `medium`) and output format on reasoning models `o3-mini` and `o4-mini`. |
| | `Models: Streaming & Judge` | Generator streaming chunks on `gpt-4o-mini` and `evaluate_response` LLM-as-a-Judge validation with `gpt-5.4`. |
| | `Models: Router Fallback` | Multi-model fallback execution (`non-existent-model-xyz` &rarr; `gpt-4o-mini`). |
| | `Models: Tool Calling` | Tool calling execution (`calculate_multiply` & `safe_weather_lookup`) with invocation assertions and error handling safeguards. |
| `tests/test_live_openai_embeddings.py` | `Embeddings: Short Inputs` | Tests `text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002` across `.txt`, `.md`, `.json`, `.csv`, `.py` formats (~15t each). |
| | `Embeddings: Large Inputs` | Tests embedding generation across 5 document formats with large inputs (~2,000t each). |
| | `Embeddings: Batch & Options` | Tests array batch inputs and reduced dimensionality (`output_dimensionality=768`). |
| `tests/test_live_openai_multimodal.py` | `Multimodal: Single Uploads` | Media uploads for PNG images, JPEG images, WAV audio files, and PDF documents. |
| | `Multimodal: Dual Combos` | Dual media inputs (PNG + WAV, JPEG + PDF). |
| | `Multimodal: Quad Combo` | Quad multi-media inputs (PNG + WAV + PDF + Text prompt). |

---

## 🛠️ Environment Prerequisites

Ensure your `.env` file or environment contains valid OpenAI API keys:

```env
KEY_LOCATION=LOCAL
OPENAI_API_KEY=sk-proj-...
```

---

## 📊 Summary & Cost Reporting

Upon completion of test runs, `CostTracker` automatically logs itemized metric tables detailing prompt tokens, output tokens, wall clock durations, and transaction costs for each call. Total suite cost is estimated at **~$0.006 (less than 1 cent)** per complete run.
