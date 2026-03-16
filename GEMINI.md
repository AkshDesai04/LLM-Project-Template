# Project Overview

This repository is a modular, scalable Python template designed for building robust applications that integrate with Large Language Models (LLMs) such as Google Gemini and OpenAI. It abstract away provider-specific implementations, offering unified interfaces for prompt configuration, model routing, media uploading (PDF, Video, Images), cost calculation, and parallel execution.

### `assets/model_pricing.csv`
This file is the central source of truth for all supported models within the framework. It acts as a local database containing crucial metadata for dozens of models across different providers (OpenAI, Google).
*   **Structure:** It maps `model_id` (e.g., `gpt-4o`, `gemini-2.5-pro`) to its `model_provider`, and details the exact costs per million tokens for input, output, cached context, and tiered pricing (e.g., tokens > 200k).
*   **Dependency in `core/llm_models/model_router.py`:** The router uses this CSV to dynamically resolve which provider a requested model belongs to. When you pass a model name to the router, it scans the CSV (using `_load_provider_mapping()`) to find the matching `model_provider`. If a match is found, it instantiates the correct client (e.g., `GeminiModel` for Google, `OpenAIModel` for OpenAI). If an exact match isn't found, it attempts a substring fallback search.
*   **Dependency in `core/llm_models/llm_model_base.py`:** The base class loads this pricing data globally upon initialization. Whenever a model completes a generation, the `_calculate_cost()` function looks up the exact model string in the loaded CSV data to compute the precise financial cost of the transaction based on the returned token usage metadata.

---

## 1. Core Module (`core/`)

The `core/` directory contains the foundational logic for interacting with AI models, defining prompts, and routing requests.

### `core/modules/base.py`
Defines the `Base` class, which extends Pydantic's `BaseModel`. This is the standard configuration schema for defining LLM prompts and their runtime parameters.
*   **Key Attributes:**
    *   `prompt` (str): The specific text instruction for the LLM.
    *   `structure` (Any): Optional Pydantic model enforcing a structured JSON output.
    *   `model` (str): The primary model identifier (e.g., `"gemini-2.5-pro"`).
    *   `fall_back_models` (list[str]): Alternative models to try if the primary fails.
    *   `temperature`, `top_p`, `top_k`: Standard LLM generation parameters.
    *   `reasoning_budget` (int | str): Used for reasoning models (like `o1`, `gpt-5`, or Gemini thinking modes).
    *   `response_mime_type` (str): The requested return format (default is `"application/json"`).
*   **Use Case / REQUIREMENT:** When a new module is to be created, the new module **must** inherit from this `Base` module and set its configurations (prompt, model, parameters, etc.) within its `__init__` method.

### `core/llm_models/llm_model_base.py`
Contains the abstract base class `LLMModels` that all provider-specific implementations inherit from. It handles critical lifecycle events such as global cost tracking and pricing estimation.
*   **Key Components & Functions:**
    *   **`JudgeResult` (Pydantic BaseModel):** Defines the structure for LLM-as-a-judge outputs, requiring a `score` (int 1-10), `reasoning` (str), and optional `improvements` (str).
    *   **`load_pricing(csv_path: str) -> dict`:** A standalone utility that reads `assets/model_pricing.csv`. It parses costs into a dictionary mapping `model_id` to nested floats for `input`, `output`, `cached`, `input_above_200k`, and `output_above_200k`. It safely handles 'N/A' strings by converting them to `0.0`.
    *   **`__init__(api_key: str, base_config: BaseModule)`:** Base constructor. Maps parameters like `model_name`, `temperature`, `top_p`, `top_k`, `fall_back_models`, `structure`, and `response_mime_type` from the passed module. **Crucially, it registers an `atexit` handler (`print_final_summary`) the first time it is called**, ensuring cost summaries are always printed when the python script terminates.
    *   **`_calculate_cost(model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict`:** Estimates the exact financial cost.
        *   *Flow:* It searches for the `model_name` in the loaded `PRICING` dictionary (falling back to substring matches if exact match fails). It then applies a tiered pricing threshold (`200,000` tokens) if the model defines different rates for high-context windows. Returns a dictionary containing `input_cost`, `output_cost`, `cached_cost`, and `total_cost`.
    *   **`record_transaction(cls, module_name: str, model_name: str, costs: dict, duration: float)`:** A class method that appends the current API call details to a global `_call_history` list and cumulatively adds to `_total_overall_cost`, `_total_input_cost`, etc.
    *   **`print_final_summary()` (Static Method):** The `atexit` hook. It prints a highly formatted, tabular summary to `sys.stdout` detailing every single transaction (Module, Model, Wall Time, Input/Output/Cached/Total Costs). It also calculates averages and final totals. Finally, it logs this entire dataset as a structured JSON object to the logger for telemetry.
    *   **`prepare_module(module: Any) -> BaseModule` (Static Method):** Standardizes the instantiation of `Base` prompt classes. If passed a Class, it instantiates it. If the `prompt` attribute is missing but `prompt_path` is provided, it automatically reads the text file from the path.
*   **Abstract Methods (Must be overridden):**
    *   `model_response(module, uploaded_file)`: Executes the main generation call.
    *   `upload_media(file_bytes, mime_type)`: Handles provider-specific file processing.
    *   `embed_content(input_content, **kwargs)`: Handles text-to-vector embeddings.
    *   `evaluate_response(input_prompt, generated_output, rubric)`: Executes the `JudgeResult` validation.

    ### `core/llm_models/gemini_model.py`
    This file implements the `LLMModels` abstract base class specifically for Google's Gemini API via the `google-genai` SDK. It handles the nuances of the Gemini API, from native file uploads to specialized reasoning budgets.
    *   **Interactions:** It inherits state and utility methods (like cost calculation) from `LLMModels` and is exclusively instantiated by `ModelRouter` using the `initialize_gemini` wrapper.
    *   **Key Functions & Flow:**
    *   **`__init__(api_key, base)`:** Initializes the native `genai.Client` and pre-compiles generation configs (temperature, top_k, top_p).
    *   **`model_response(module, uploaded_file) -> Any`:** The core generation engine.
        *   *Flow:* It attempts to generate content using the primary model. If the primary fails or throws an exception, it enters a `max_retries=3` loop, systematically falling back to models defined in `fall_back_models`.
        *   *Reasoning:* It dynamically checks for a `reasoning_budget`. If a string (e.g., "high", "low") is passed, it converts it into a `ThinkingLevel` and attaches a `ThinkingConfig` to the request, capturing the model's internal thought process.
        *   *Metadata & Cost:* Upon success, it extracts exhaustive usage metadata (prompt tokens, candidate tokens, cached tokens) directly from the Gemini response object. It passes these values to the parent `_calculate_cost()` method and logs a comprehensive "GEMINI METADATA REPORT" to the terminal.
        *   *Structured Output:* If a Pydantic `structure` was provided in the module, it returns `response.parsed`; otherwise, it returns raw `response.text`.
    *   **`upload_media(file_bytes: bytes, mime_type: str) -> types.File`:** Handles Gemini's native File API. It uploads binary data via `BytesIO`. Crucially, it implements a polling loop: if the file state is `"PROCESSING"`, it sleeps and checks again until the state becomes `"ACTIVE"`, preventing premature generation calls on unready media.
    *   **`embed_content(text, task_type, model, dimensions)`:** Calls the `gemini-embedding-001` model, calculates cost, and returns the vector list.
    *   **`evaluate_response(input_prompt, generated_output, rubric)`:** Implements LLM-as-a-judge by spinning up a temporary `BaseModule` targeting `gemini-2.0-flash` to grade the provided output.

    ### `core/llm_models/openai_model.py`
    This file implements the `LLMModels` abstract base class for the official `openai` Python client. It manages OpenAI's specific structural requirements, especially concerning multimodal inputs and reasoning models.
    *   **Interactions:** Like Gemini, it inherits from `LLMModels` and is instantiated solely via `ModelRouter` using `initialize_openai`. It standardizes OpenAI's differing API paradigms so the `ModelRouter` interface remains unified.
    *   **Key Functions & Flow:**
    *   **`model_response(module, uploaded_file) -> Any`:**
        *   *Message Construction:* It complexly handles formatting. If an image or video frame array is provided, it structures the prompt as a list of dicts (`{"type": "image_url"...}`). If standard text files are attached, it appends them as string context to the main prompt.
        *   *Model Overrides:* It detects if the chosen model is a reasoning model (e.g., `o1-`, `o3-`, `gpt-5`). If so, it dynamically strips standard parameters (`temperature`, `top_p`) which are incompatible with these models, and injects `reasoning_effort` instead.
        *   *Structured Output:* If a Pydantic structure is provided, it uses the specialized `beta.chat.completions.parse` endpoint to guarantee JSON schema compliance. Otherwise, it uses the standard completions endpoint.
        *   *Metadata & Cost:* Extracts `usage` from the response, passes tokens to the parent `_calculate_cost()`, logs the "OPENAI METADATA REPORT", and returns the content.
    *   **`upload_media(file_bytes: bytes, mime_type: str) -> Any`:** Because OpenAI handles media differently than Gemini (no persistent file API for standard chat), this method processes media locally *before* the API call:
        *   *PDFs:* Uses `PyPDF2` to extract text strings locally.
        *   *Images:* Encodes bytes to a `base64` string formatted for the Vision API.
        *   *Videos (`_process_video_for_openai`):* Saves the video to a temporary file, uses OpenCV (`cv2`) to capture frames at a specific interval (FPS), encodes each frame to `base64`, and returns a list of image dictionary objects.
    *   **`embed_content(text, model)`:** Calls `text-embedding-3-small`, logs costs, and returns the vector.
    *   **`evaluate_response(...)`:** Spins up a temporary prompt targeting `gpt-4o-mini` to grade outputs against a rubric.

    ### `core/llm_models/model_router.py`
This file implements the `LLMModels` abstract base class specifically for Google's Gemini API via the `google-genai` SDK. It handles the nuances of the Gemini API, from native file uploads to specialized reasoning budgets.
*   **Interactions:** It inherits state and utility methods (like cost calculation) from `LLMModels` and is exclusively instantiated by `ModelRouter` using the `initialize_gemini` wrapper.
*   **Key Functions & Flow:**
    *   **`__init__(api_key, base)`:** Initializes the native `genai.Client` and pre-compiles generation configs (temperature, top_k, top_p).
    *   **`model_response(module, uploaded_file) -> Any`:** The core generation engine.
        *   *Flow:* It attempts to generate content using the primary model. If the primary fails or throws an exception, it enters a `max_retries=3` loop, systematically falling back to models defined in `fall_back_models`.
        *   *Reasoning:* It dynamically checks for a `reasoning_budget`. If a string (e.g., "high", "low") is passed, it converts it into a `ThinkingLevel` and attaches a `ThinkingConfig` to the request, capturing the model's internal thought process.
        *   *Metadata & Cost:* Upon success, it extracts exhaustive usage metadata (prompt tokens, candidate tokens, cached tokens) directly from the Gemini response object. It passes these values to the parent `_calculate_cost()` method and logs a comprehensive "GEMINI METADATA REPORT" to the terminal.
        *   *Structured Output:* If a Pydantic `structure` was provided in the module, it returns `response.parsed`; otherwise, it returns raw `response.text`.
    *   **`upload_media(file_bytes: bytes, mime_type: str) -> types.File`:** Handles Gemini's native File API. It uploads binary data via `BytesIO`. Crucially, it implements a polling loop: if the file state is `"PROCESSING"`, it sleeps and checks again until the state becomes `"ACTIVE"`, preventing premature generation calls on unready media.
    *   **`embed_content(text, task_type, model, dimensions)`:** Calls the `gemini-embedding-001` model, calculates cost, and returns the vector list.
    *   **`evaluate_response(input_prompt, generated_output, rubric)`:** Implements LLM-as-a-judge by spinning up a temporary `BaseModule` targeting `gemini-2.0-flash` to grade the provided output.

### `core/llm_models/openai_model.py`
This file implements the `LLMModels` abstract base class for the official `openai` Python client. It manages OpenAI's specific structural requirements, especially concerning multimodal inputs and reasoning models.
*   **Interactions:** Like Gemini, it inherits from `LLMModels` and is instantiated solely via `ModelRouter` using `initialize_openai`. It standardizes OpenAI's differing API paradigms so the `ModelRouter` interface remains unified.
*   **Key Functions & Flow:**
    *   **`model_response(module, uploaded_file) -> Any`:**
        *   *Message Construction:* It complexly handles formatting. If an image or video frame array is provided, it structures the prompt as a list of dicts (`{"type": "image_url"...}`). If standard text files are attached, it appends them as string context to the main prompt.
        *   *Model Overrides:* It detects if the chosen model is a reasoning model (e.g., `o1-`, `o3-`, `gpt-5`). If so, it dynamically strips standard parameters (`temperature`, `top_p`) which are incompatible with these models, and injects `reasoning_effort` instead.
        *   *Structured Output:* If a Pydantic structure is provided, it uses the specialized `beta.chat.completions.parse` endpoint to guarantee JSON schema compliance. Otherwise, it uses the standard completions endpoint.
        *   *Metadata & Cost:* Extracts `usage` from the response, passes tokens to the parent `_calculate_cost()`, logs the "OPENAI METADATA REPORT", and returns the content.
    *   **`upload_media(file_bytes: bytes, mime_type: str) -> Any`:** Because OpenAI handles media differently than Gemini (no persistent file API for standard chat), this method processes media locally *before* the API call:
        *   *PDFs:* Uses `PyPDF2` to extract text strings locally.
        *   *Images:* Encodes bytes to a `base64` string formatted for the Vision API.
        *   *Videos (`_process_video_for_openai`):* Saves the video to a temporary file, uses OpenCV (`cv2`) to capture frames at a specific interval (FPS), encodes each frame to `base64`, and returns a list of image dictionary objects.
    *   **`embed_content(text, model)`:** Calls `text-embedding-3-small`, logs costs, and returns the vector.
    *   **`evaluate_response(...)`:** Spins up a temporary prompt targeting `gpt-4o-mini` to grade outputs against a rubric.
The unified factory entry point. It abstracts the provider selection process. **IMPORTANT: When calling an LLM, you must default to using `ModelRouter`. Direct instantiation of provider-specific files (like `GeminiModel` or `OpenAIModel`) should only be done if it is absolutely impossible to use the router.**
*   **Key Functions & Flow:**
    *   **`_load_provider_mapping() -> dict`:** A cached helper function that reads `model_pricing.csv` specifically to extract a mapping of `model_id` -> `model_provider` (e.g., `'gpt-4o': 'openai'`, `'gemini-2.5-pro': 'google'`). It caches the result in a global variable `_PROVIDER_MAPPING` to avoid repeated disk reads.
    *   **`__init__(module: BaseModule, api_keys: dict, fallback_index: int = 0)`:** The routing constructor.
        *   *Flow:*
            1.  Determines the target model name. If `fallback_index` is 0, it uses `module.model`. If > 0, it looks into `module.fall_back_models[fallback_index - 1]`.
            2.  Looks up the target model in the provider mapping (using exact match, then substring fallback). Raises a `ValueError` if the provider cannot be determined.
            3.  Based on the resolved provider string (`'google'` or `'openai'`), it extracts the appropriate key from the `api_keys` dictionary (`GEMINI_KEY` or `OPEN_AI_KEY`).
            4.  Finally, it calls the specific initializer (e.g., `initialize_gemini` or `initialize_openai`) and stores the resulting object in `self.model_instance`.
    *   **Proxy Methods:** The router exposes `model_response`, `upload_media`, `embed_content`, and `evaluate_response`. These simply pass the arguments directly through to the underlying `self.model_instance`, ensuring the caller never needs to know which specific provider API is being executed under the hood.

---

## 2. Utilities Module (`utils/`)

Helper scripts for infrastructure and system-level operations.

### `utils/env_ops.py`
A hybrid local/cloud secrets manager.
*   **Key Functions:**
    *   `get_local_secret(key_name)`: Loads from `.env`.
    *   `_get_secrets_manager_client()`: Initializes `boto3` for AWS Secrets Manager.
    *   `get_secret_dict(secret_name)`: Fetches a JSON dictionary of secrets from AWS and caches it in memory to prevent duplicate API calls.
    *   `get_keys_dict()`: The main entry point. Retrieves `GEMINI_KEY` and `OPEN_AI_KEY` from the AWS secret defined by the local `SECRET_NAME` environment variable.

### `utils/file_ops.py`
Standardized IO operations ensuring utf-8 encodings and proper error handling.
*   **Functions:** `read_file` (text), `get_file` (binary - used heavily for media uploads), `read_prompt` (reads txt files from `core/prompts`), and `read_csv` (loads rows as dicts).

### `utils/logger.py`
Provides a robust, dual-output logging system.
*   **Key Features:**
    *   Outputs standard readable logs to the terminal (`sys.stdout`).
    *   Simultaneously writes **JSON formatted** logs to `logs/rfp_ai_{YYYYMMDD}.log`. This JSON structure is crucial for downstream analysis of metadata, exceptions, and token costs recorded by the LLM models.

### `utils/parallel_executor.py`
A highly resilient, multi-threading utility specifically designed for managing concurrent tasks, particularly network requests to LLM APIs where rate-limiting and intermittent failures are common.
*   **`ThreadSafeRateLimiter`**: A helper class that enforces a maximum number of requests per minute across all active threads using a threading lock. It calculates exact sleep durations to ensure the rate limit is never exceeded.
*   **`calculate_worker_count(max_threads: int, data_size: int) -> int`**: Intelligently determines the size of the thread pool based on user input:
    *   `0`: Unbounded. Creates exactly one thread per data item (or 1 if `data_size` is 0).
    *   `-1`: Creates one thread per available CPU core.
    *   `< -1`: Creates a multiple of CPU cores (e.g., `-2` creates `2 * CPU cores`).
    *   `> 0`: Uses the exact integer specified as the maximum number of worker threads.
*   **`parallel_execute(target_function, data, max_threads=0, max_req_per_min=None, max_retries=0, retry_timer=0)`**: The core function to orchestrate parallel execution.
    *   **Argument Unpacking:** The `data` parameter accepts a list of arguments. If an item in the list is a tuple or list, it is automatically unpacked (`*item`) when passed to the `target_function`. If it is any other type, it is passed as a single argument.
    *   **Retry Logic:** If a thread raises an exception, an internal wrapper catches it and retries the execution up to `max_retries` times, pausing for `retry_timer` seconds between attempts.
    *   **Deterministic Output:** It utilizes `concurrent.futures.ThreadPoolExecutor` and maps futures back to their original index. This guarantees that the returned list of results is in the **exact same order** as the input `data` list, regardless of which thread finished first.
    *   **Fault Tolerance:** If a specific task fails completely after exhausting all retries, the framework does not crash the entire execution batch. Instead, the resulting `Exception` object itself is placed at that specific index in the output list, allowing the main thread to handle individual failures gracefully without losing successful results.

---

## 3. Testing and Usage (`tests/`)

The testing directory provides a complete blueprint of how to integrate the components.

### `tests/TestModule.py`
Defines a concrete prompt instance `FileSummaryPrompt` that inherits from `Base`.
*   **Use Case:** Sets the specific goal: *"Summarize all the provided files and return a detailed summary..."* and injects the dynamic model target.

### `tests/TestFile.py`
An end-to-end integration test highlighting parallel execution, automatic file handling, and cross-provider routing.
*   **Workflow:**
    1.  **Keys:** Fetches credentials via `env_ops.get_keys_dict()`.
    2.  **Target Selection:** Scans `tests/test_files/` for media (PDFs, MP4s, JPGs).
    3.  **Parallel Setup:** Defines a list of models across different providers (`gemini-3.1-pro-preview`, `gpt-5.4-2026-03-05`, etc.). Prepares data for `parallel_execute`.
    4.  **Execution (`_summarize_with_model`):** For each model (running concurrently):
        *   A `FileSummaryPrompt` is created.
        *   A `ModelRouter` is initialized.
        *   Local files are loaded using `get_file()` and automatically processed/uploaded via `router.upload_media()`. The router handles the complex differences (e.g., waiting for Gemini active status vs. extracting frames for OpenAI).
        *   The prompt and processed media are submitted to `router.model_response()`.
    5.  **Output:** Results and errors are aggregated. Upon script exit, the `LLMModels` `atexit` hook automatically prints the comprehensive pricing and duration summary to the terminal.

---

## 4. Building and Running

### Prerequisites
*   Python 3.9+
*   Environment Variables: Set `SECRET_NAME` and `AWS_REGION` if using AWS, or define `GEMINI_KEY` and `OPEN_AI_KEY` directly.

### Installation
```bash
pip install -r requirements.txt
```

### Running the End-to-End Test
Run the test module to see the parallel execution and model routing in action. Drop test media (videos, images, PDFs) into the `tests/test_files/` directory first.
```bash
python -m tests.TestFile
```