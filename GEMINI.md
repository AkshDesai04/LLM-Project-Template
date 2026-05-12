# Project Overview

This repository is a modular, scalable Python template designed for building robust applications that integrate with Large Language Models (LLMs) such as Google Gemini, OpenAI, Ollama, and Perplexity. It abstracts away provider-specific implementations, offering unified interfaces for prompt configuration, model routing, media uploading (PDF, Video, Images), cost calculation, parallel execution, and complex file conversions.

### `assets/model_pricing.csv`
This file is the central source of truth for all supported models within the framework. It acts as a local database containing crucial metadata for dozens of models across different providers.
*   **Structure:** It maps `model_id` (e.g., `gpt-4o`, `gemini-2.5-pro`) and details exact costs per million tokens for input, output, cached context, and tiered high-context pricing (e.g., input > 200k).
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
    *   `model` (str): The primary model identifier (e.g., `"gemini-2.5-pro"`).
    *   `fallback_models` (list[str]): Alternative models to systematically attempt if the primary fails.
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

### `core/llm_models/cost_tracker.py`
Singleton class defining the `CostTracker`. Calculates and stores multi-call metrics seamlessly.
*   *Flow:* Upon completion generation, models transmit usage tokens to the `calculate_cost` logic mapping tier thresholds natively. Subsequent valid queries map to the global dictionary via `record_transaction`.
*   **`print_final_summary()`:** Hooks to the `atexit` garbage collector natively executing and logging deep tabular CLI metric layouts detailing every processed transaction upon code termination. Captures overarching application performance cleanly to JSON telemetry files dynamically.

### `core/llm_models/router.py`
The unified factory entry point (`ModelRouter`). It abstracts the provider selection process. **IMPORTANT: When calling an LLM, you must default to using `ModelRouter`. Direct instantiation of provider-specific files should be avoided.**
*   **Key Functions & Flow:**
    *   **`get_provider_by_model_name()`:** Scans model name prefixes dynamically categorizing routing constraints (`ollama/`, `gemini`, `gpt`, `sonar`) targeting local instances, Google, OpenAI, or Perplexity natively.
    *   **Initialization & Lazy Loading:** Only initializes the specific requisite module avoiding bloat and missing API key errors. Retrieves keys contextually through local `.env` definitions mapping them into the abstracted `LLMProvider`.
    *   **Proxy Methods:** Exposes core tasks cleanly bridging execution transparently without altering API interactions.

### `core/llm_models/providers/`
This folder maps the abstracted provider classes integrating directly with backend Python SDK architectures natively.
*   **`gemini.py` (`GeminiProvider`)**: Operates the `google-genai` SDK executing standard loops, managing `ThinkingConfig` generation loops, routing structured files natively into the active File API utilizing robust HTTP polling loops preventing generation overlap on unready internal metrics.
*   **`openai.py` (`OpenAIProvider`)**: Implements strict `beta.chat.completions.parse` routines natively handling extensive visual data parsing via `media_utils` logic avoiding missing native media API boundaries. Triggers the specialized `responses` API robustly handling unaccepted chat logic configurations automatically seamlessly retrying context errors securely.
*   **`ollama.py` (`OllamaProvider`)**: Seamlessly connects to dynamic `localhost` Ollama deployments mapping images dynamically avoiding cloud storage, securely capturing locally run metrics gracefully mapping structural returns safely.
*   **`perplexity.py` (`PerplexityProvider`)**: Hooks across OpenAI base class structures mapping web index references dynamically defining native citations formatting without additional client SDK integrations.

### `core/llm_models/utils/media_utils.py`
Abstracts robust native file decoding algorithms resolving visual and textual structures entirely offline. 
*   Leverages `cv2` mapping robust frame extractions structuring internal loops yielding high-resolution visual dict arrays supporting seamless OpenAI video integrations.
*   Resolves native `PyPDF2` unstructured texts preventing token bloat seamlessly mapping simple string iterations natively.

---

## 2. Utilities Module (`utils/`)

Helper scripts for infrastructure, conversion, and system-level operations.

### `utils/env_ops.py`
A hybrid local/cloud secrets manager mapping local overrides contextually scaling complex cloud infrastructures robustly via `boto3`.
*   **Key Functions:** Integrates `get_aws_secret` utilizing dynamic `SECRET_NAME` references mapping deep structural API key dicts gracefully resolving memory locks caching local execution metrics resolving latency.

### `utils/file_ops.py`
Standardized IO operations ensuring utf-8 encodings and proper error handling. Includes text/binary data ingestion mapping local text layouts securely natively mapping `.csv` metrics contextually.

### `utils/logger.py`
Provides a robust, dual-output logging system.
*   **Key Features:** Outputs standard readable logs to the terminal (`sys.stdout`) natively yielding structured, contextually mapped **JSON** architectures yielding native telemetry parsing routines straight to `logs/` timestamping seamlessly.

### `utils/markitdown_utils.py`
A comprehensive Microsoft `MarkItDown` integration mapping heavy multi-modal documents securely via a centralized class pipeline resolving disparate inputs securely.
*   **Key Features:** Generates text streams mapping URLs natively rendering YouTube transcripts natively. Yields multi-modal text structures routing local `.pptx`, `.docx` or active Excel worksheets efficiently parsing strings yielding high-fidelity textual strings native to AI consumption loops effectively.

### `utils/parallel_executor.py`
A highly resilient, multi-threading utility explicitly built for external API routing integrations managing HTTP blocks.
*   **`ThreadSafeRateLimiter`**: Maps dynamic `Threading.Lock` layouts preserving total API throughput limits executing requests evenly securely mapping global bounds safely natively mapping bounds efficiently.
*   **`parallel_execute(...)`**: The active multi-threaded engine orchestrating list iteration loops natively decoupling iteration failures mapping Exception catches specifically back to the exact list index avoiding total pipeline crashes seamlessly bridging heavy async loads confidently reliably returning structured layouts seamlessly mapping native lists contextually.

---

## 3. Testing and Usage (`core/scripts/` & `core/modules/`)

The testing and scripts directories cleanly define standard interaction methodologies targeting the underlying integrations efficiently securely.

### `core/modules/test_module.py`
Highlights an architecture instantiation mapping `FileSummaryPrompt` implementing a direct class derivation mapping local text `.txt` references into standard `base` config objects confidently initializing configurations contextually natively mapping local `.md` iterations efficiently.

### `core/scripts/test_script.py`
A unified end-to-end integration mapping complete `ModelRouter` definitions interacting broadly executing generation requests utilizing custom logger formats confirming complete modular interactions executing effectively across backend logic structures properly integrating logging, latency scaling, and error bridging securely comprehensively wrapping test validations reliably globally mapping configurations.
