# Project Overview

This repository is a modular, scalable Python template designed for building robust applications that integrate with Large Language Models (LLMs) such as Google Gemini, OpenAI, Anthropic, Ollama, vLLM, and Perplexity. It abstracts away provider-specific implementations, offering unified interfaces for prompt configuration, model routing, media uploading (PDF, Video, Images), cost calculation, parallel execution, and complex file conversions.

## Core Architectural Principles
1. **File Readability & Maintainability:** No Python file in the codebase exceeds **200 lines**. Large modules are split into logical subpackages.
2. **Absolute Imports:** All internal module imports use absolute import paths (e.g. `from utils.logging import get_logger`, `from core.llm_models.providers.gemini import GeminiProvider`). Relative imports (`from .` or `from ..`) are strictly forbidden.
3. **Provider Consistency:** All model provider implementations under `core/llm_models/providers/` follow an identical subpackage directory architecture (`__init__.py`, `provider.py`, `media_handler.py`, `response_handler.py`).

---

### `assets/model_pricing.csv`
This file is the central source of truth for all supported models within the framework. It acts as a local database containing crucial metadata for dozens of models across different providers.
*   **Structure:** It maps `model_id` (e.g., `gpt-4o`, `gemini-2.5-pro`) and details exact costs per million tokens for input, output, cached context, and tiered high-context pricing (e.g., input > 200k). Rows are keyed by the **bare** model id, without the `provider/` prefix used in module configs.
*   **Not required for routing:** A model absent from this file still runs; it is simply reported at `$0.00` with a warning. Add a row when you want its cost tracked.
*   **Dependency in `core/llm_models/cost_tracker.py`:** The singleton tracking class uses this CSV to dynamically resolve transaction costs in real-time, assigning exact financial metrics on each executed LLM completion.

---

## 1. Core Module (`core/`)

The `core/` directory contains the foundational logic for interacting with AI models, defining prompts, and routing requests dynamically across varying LLM providers.

### `core/modules/base.py`
Defines the `Base` class, which extends Pydantic's `BaseModel`. This is the standard configuration schema for defining LLM prompts and their runtime parameters.
*   **Key Attributes:**
    *   `prompt` (str): The specific text instruction for the LLM.
    *   `system_prompt` (str): Broad system guidelines injected dynamically.
    *   `structure` (Any): Optional Pydantic model enforcing a structured JSON output.
    *   `model` (str | None): Optional primary model identifier in the canonical `provider/model` form (e.g., `"gemini/gemini-2.5-pro"`). Has no default value in `base.py`.
    *   `models` (list[str]): List of model identifiers in canonical `provider/model` form. Defaults to `["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite"]`. If both `model` and `models` exist, `model` is primary and `models` contains ordered fallbacks; if only `models` is specified, the first entry is primary and the remainder are ordered fallbacks.
    *   `temperature`, `top_p`, `top_k`, `max_tokens`: Standard LLM generation parameters.
    *   `reasoning_budget` (int | str): Controls deep thinking parameters native to advanced reasoning models (`o1`, `o3`, `gpt-5`, Gemini Thinking modes).
    *   `stream` (bool): Configures the model to dispatch chunked iterative generators instead of static single completions.
    *   `response_mime_type` (str): The requested return format (default is `"application/json"`).
    *   `tools`, `logprobs`, `search_domain_filter`, `candidate_count`: Feature-specific metrics natively bridged to supported models seamlessly.
*   **Use Case / REQUIREMENT:** When a new module is to be created, the new module **must** inherit from this `Base` class and set its target variables prior to interacting with the model pipeline.

### `core/llm_models/base_provider.py`
Contains the abstract base class `LLMProvider` that all provider-specific implementations inherit from. 
*   **Key Components & Functions:**
    *   **`JudgeResult`:** A predefined Pydantic BaseModel defining structures for LLM-as-a-judge outputs, requiring scores, reasoning, and granular improvement tracking.
    *   **`prepare_module(module: Any) -> BaseModule` (Static):** Standardizes instantiation of `Base` prompt classes. Resolves textual file dependencies mapping absolute prompt paths safely.
*   **Abstract Methods (Must be overridden by all subsequent providers):**
    *   `model_response(module, uploaded_file)`: Executes the main generation call.
    *   `upload_media(file_bytes, mime_type)`: Handles provider-specific visual/document preprocessing.
    *   `embed_content(input_content, **kwargs)`: Executes textual vector clustering mapping.
    *   `evaluate_response(input_prompt, generated_output, rubric)`: Validates the model leveraging LLM validation pipelines natively.

### `core/llm_models/cost_tracker.py` & `cost_summary.py`
Singleton class defining the `CostTracker` and table reporting utility `cost_summary.py`. Calculates and stores multi-call metrics seamlessly.
*   *Flow:* Upon completion generation, models transmit usage tokens to the `calculate_cost` logic mapping tier thresholds natively. Subsequent valid queries map to the global dictionary via `record_transaction` (including token counts). Failed primary/fallback attempts are recorded via `record_failed_attempt` as zero-token, zero-cost rows with `status="failed"`.
*   **`print_final_summary()`:** Hooks to the `atexit` garbage collector natively executing and logging tabular CLI metric layouts detailing every processed transaction (tokens + costs, with failed models labeled `model (failed)`) upon code termination.

### `core/llm_models/router.py`
The unified factory entry point (`ModelRouter`). It abstracts the provider selection process. **IMPORTANT: When calling an LLM, you must default to using `ModelRouter`. Direct instantiation of provider-specific files should be avoided.**
*   **Key Functions & Flow:**
    *   **`get_provider_by_model_name()`:** Reads the provider directly off the canonical `provider/model` prefix (`gemini/`, `openai/`, `anthropic/`, `perplexity/`, `ollama/`, `vllm/`).
    *   **Initialization & Lazy Loading:** Builds the model chain from `module.model` and/or `module.models`. Only initializes the specific requisite module avoiding bloat and missing API key errors.
    *   **Fallback Chain:** `model_response` walks the chain across providers. Each failed model is recorded as a zero-token `(failed)` row before the next fallback is attempted.

### `core/llm_models/providers/`
This directory contains modular provider subpackages integrating directly with backend Python SDK architectures natively. Each provider (`gemini`, `openai`, `anthropic`, `ollama`, `vllm`, `perplexity`) is structured into `__init__.py`, `provider.py`, `media_handler.py`, and `response_handler.py`.
*   **`gemini/` (`GeminiProvider`)**: Operates the `google-genai` SDK executing standard loops, managing `ThinkingConfig` generation loops, and routing media either through the Developer File API (`GEMINI_KEY_TYPE=GEMINI_KEY`) or as inlined `Part`s under Vertex AI (`GEMINI_KEY_TYPE=SERVICE_ACC_JSON`).
*   **`openai/` (`OpenAIProvider`)**: Implements `beta.chat.completions.parse` routines and native responses API integration (`responses_api.py` & `chat_completions.py`).
*   **`anthropic/` (`AnthropicProvider`)**: Drives the Claude Messages API, supplying mandatory `max_tokens` cap, hoisting system prompt, and using schema-derived tools for structured JSON outputs.
*   **`ollama/` (`OllamaProvider`)**: Seamlessly connects to dynamic `localhost` Ollama deployments mapping images dynamically.
*   **`vllm/` (`VLLMProvider`)**: Client for a self-hosted vLLM OpenAI-compatible server resolved from `VLLM_URL` (default `http://localhost:8000/v1`).
*   **`perplexity/` (`PerplexityProvider`)**: Hooks across OpenAI base class structures mapping web index references dynamically.

### `core/llm_models/utils/media_utils.py`
Abstracts robust native file decoding algorithms resolving visual and textual structures entirely offline. 
*   Leverages `cv2` mapping robust frame extractions structuring internal loops yielding high-resolution visual dict arrays supporting seamless video integrations.
*   Resolves native `PyPDF2` unstructured texts preventing token bloat seamlessly mapping simple string iterations natively.

---

## 2. Utilities Module (`utils/`)

Helper scripts for infrastructure, conversion, and system-level operations divided into domain subpackages:

### `utils/env/` (`env_ops.py`, `constants.py`, `aws_secrets.py`, `gemini_credentials.py`)
A hybrid local/cloud secrets manager mapping local overrides contextually scaling complex cloud infrastructures robustly via `boto3`.
*   **`KEY_LOCATION` switch:** `get_secret()` is the single entry point providers call for credentials. `LOCAL` resolves through `.env`; `AWS_SM` resolves against AWS Secrets Manager bundle named by `SECRET_NAME`.

### `utils/io/` (`file_ops.py`)
Standardized I/O operations ensuring utf-8 encodings and proper error handling for text, prompt, and CSV files.

### `utils/logging/` (`logger.py`, `lambda_logger.py`)
Provides a robust dual-output logging system driven by `LOGGING_MODE` (`NORMAL` vs `LAMBDA`) and `LOGGING_LEVEL`.

### `utils/db/` (`base.py`, `helpers.py`, `router.py`, `sqlite.py`, `postgres.py`, `mysql.py`)
Interoperable database connectors and factory router supporting SQLite, PostgreSQL, and MySQL. All connectors stay under 180 lines.

### `utils/document/` (`markitdown.py`)
Microsoft `MarkItDown` integration converting documents, URLs, and media into Markdown.

### `utils/concurrency/` (`parallel_executor.py`)
Multi-threading execution utility with rate limiting and thread safety.

---

## 3. Testing and Usage (`core/scripts/` & `core/modules/`)

Standard interaction methodologies and end-to-end integration test scripts.
*   `core/modules/test_module.py`: Example prompt module derivation.
*   `core/scripts/test_script.py`: Unified end-to-end execution script.
