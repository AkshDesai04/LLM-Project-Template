This project is a template.

# LLM-Project-Template

## Overview
This repository serves as a modular template for building applications that integrate with Large Language Models (LLMs) like Google Gemini, OpenAI, Anthropic, Ollama, vLLM, and Perplexity. It includes robust utilities for logging, parallel execution, environment management, and document processing via MarkItDown.

---

## Module: `core`
The `core` module contains the primary logic for interfacing with LLM providers and defining the base structure for application modules.

### File: `core/llm_models/base_provider.py`
This file defines the abstract base class and shared utilities for all LLM implementations.

#### Functions
- **`LLMProvider.prepare_module(module: Any)` (static)**
  - **Input:** `module` (Any) - A class or an instance of `BaseModule`.
  - **Output:** `BaseModule` - An initialized module instance.
  - **Process:** Ensures the module is instantiated and its prompt is loaded from the specified path if not already present.

#### Class: `JudgeResult` (BaseModel)
Defines the structure for LLM-as-a-judge outputs.
- `score` (int): 1-10 evaluating response quality.
- `reasoning` (str): Detailed explanation.
- `improvements` (Optional[str]): Suggestions.

#### Class: `LLMProvider` (ABC)
Abstract base class for LLM providers.
- **`__init__(self, api_key: str, base_config: BaseModule)`**
  - **Input:** `api_key` (str), `base_config` (BaseModule).
  - **Output:** `None`.
  - **Process:** Initializes API credentials and model configurations (e.g., temperature, max_tokens, reasoning budgets).
- **`model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any` (abstract)**
- **`upload_media(self, file_bytes: bytes, mime_type: str) -> Any` (abstract)**
- **`embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]` (abstract)**
- **`evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult` (abstract)**

---

### File: `core/llm_models/cost_tracker.py`
Singleton utility for centralized tracking and calculation of API usage costs.

#### Class: `CostTracker`
- **`_load_pricing(self)`**
  - **Process:** Loads dynamic model pricing from `assets/model_pricing.csv`.
- **`calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict`**
  - **Input:** `model_name` (str), `prompt_tokens` (int), `output_tokens` (int), `cached_tokens` (int, default=0).
  - **Output:** `dict` - Calculated costs for input, output, cached tokens, and the total.
  - **Process:** Matches the model name against pricing data and computes estimated costs based on token usage, including >200k token tier thresholds.
- **`record_transaction(self, module_name: str, model_name: str, costs: dict, duration: float, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0, status: str = "success")`**
  - **Input:** `module_name` (str), `model_name` (str), `costs` (dict), `duration` (float), token counts, and optional `status`.
  - **Process:** Appends transaction details (including token counts and status) to the global history and updates overall session metrics for successful calls.
- **`record_failed_attempt(self, module_name: str, model_name: str, duration: float, error: Optional[Exception] = None)`**
  - **Process:** Appends a zero-token, zero-cost row with `status="failed"` so failed primary/fallback models appear in the summary without inflating totals.
- **`print_final_summary(self)`**
  - **Process:** Registered via `atexit`. Prints a detailed, itemized tabular summary of all session costs, token counts, and execution times upon script exit (labeling failed models as `model (failed)`), and logs the full history as JSON.

---

### File: `core/llm_models/router.py`
Unified entry point for dynamic model selection and routing across multiple LLM backends.

#### Class: `ModelRouter`
- **`__init__(self, module: BaseModule, fallback_index: int = 0)`**
  - **Process:** Builds an order-preserving model chain from `module.model` and/or `module.models` (starting at `fallback_index`), identifies the necessary provider for the primary model, lazily imports the respective provider's class, and initializes the `LLMProvider` instance.
- **`model_response(...)`**
  - **Process:** Walks the model chain. On each failure, records a zero-token failed attempt via `CostTracker` and tries the next fallback (including cross-provider fallbacks). Raises if the entire chain fails.
- **`get_provider_by_model_name(model_name: str) -> str` (static)**
  - **Input:** `model_name` (str) in the canonical `provider/model` form.
  - **Output:** `str` - Provider name (e.g., "openai", "google", "anthropic", "ollama", "vllm", "perplexity").
  - **Process:** Reads the provider straight off the prefix. A name with no recognised prefix falls back to `_infer_provider_from_bare_name`, which guesses from the model family and logs a warning.
- **`strip_routing_prefix(model_name: str) -> str` (static)**
  - **Process:** Removes the provider prefix so the bare model id is what reaches the SDK and the pricing table.

---

### File: `core/llm_models/model_names.py`
Owns the `provider/model` naming convention, shared by the router and the cost tracker.

- **`PROVIDER_ALIASES`**: Accepted prefixes mapped onto internal provider keys — `gemini`/`google` → google, `openai`, `anthropic`/`claude`, `perplexity`/`sonar`, `ollama`, `vllm`.
- **`split_model_name(model_name) -> (provider | None, model)`**: Splits on the first slash, but only when the prefix is a known alias, so a repository-style id like `meta-llama/Llama-3.1-8B` survives intact. `vllm/meta-llama/Llama-3.1-8B` yields `("vllm", "meta-llama/Llama-3.1-8B")`.
- **`strip_provider_prefix(model_name) -> str`**: The bare model id.

**Why the prefix:** the provider is stated rather than guessed, so a model released after this code was written routes correctly with no change here and no row in `assets/model_pricing.csv`. An unpriced model is reported at `$0.00` with a warning; the call itself is unaffected.

---

### File: `core/llm_models/providers/gemini.py`
Implementation of the Google Gemini LLM provider.

#### Class: `GeminiProvider` (LLMProvider)
Credential type is selected by `GEMINI_KEY_TYPE` in `.env` (`GEMINI_KEY` or `SERVICE_ACC_JSON`). `GEMINI_KEY` talks to the Gemini Developer API. `SERVICE_ACC_JSON` talks to Vertex AI using a service-account JSON (`GEMINI_SERVICE_ACCOUNT_FILE` or `GEMINI_SERVICE_ACCOUNT_JSON`) plus `GEMINI_PROJECT` / `GEMINI_LOCATION`.
- **`model_response(...)`**
  - **Process:** Sends prompts/files to Gemini using the `google-genai` SDK. Supports advanced parameters, reasoning budgets via `ThinkingConfig`, structured JSON output, native file caching metadata, and streaming generators.
- **`upload_media(...)`**
  - **Process:** Under `GEMINI_KEY`, uploads bytes to Gemini's File API and polls until active. Under `SERVICE_ACC_JSON`, Vertex rejects that Files API, so the bytes are inlined as a `types.Part` instead.
- **`embed_content(...)`**
  - **Process:** Generates embeddings utilizing `gemini-embedding-001`.

---

### File: `core/llm_models/providers/openai.py`
Implementation of the OpenAI LLM provider.

#### Class: `OpenAIProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Manages chat completions endpoints. Gracefully handles `o1`, `o3`, and `gpt-5` reasoning logic parameters, structured JSON outputs via `beta.chat.completions.parse`, base64 visual inputs, and automatically retries utilizing the Responses API if a chat completion fails due to strict model restrictions.
- **`upload_media(...)`**
  - **Process:** Extracts text from PDFs locally, encodes image bytes to base64 dictionaries, and slices video frames for the Vision API since standard chat doesn't utilize robust persistent external files.
- **`embed_content(...)`**
  - **Process:** Generates embeddings utilizing OpenAI's `text-embedding-3-small`.

---

### File: `core/llm_models/providers/anthropic.py`
Implementation of the Anthropic (Claude) LLM provider.

#### Class: `AnthropicProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Calls the Messages API. Supplies the mandatory `max_tokens` (defaulting to 4096), passes the system prompt as a top-level `system` argument rather than a message, and converts the OpenAI-shaped media payloads from `media_utils` into Anthropic image blocks. Structured output is enforced by forcing a single tool call built from the Pydantic schema, since Anthropic has no JSON response format. A string `reasoning_budget` maps onto the `effort` scale, while an integer budget enables extended thinking with `temperature=1`.
- **`upload_media(...)`**
  - **Process:** Extracts PDF text locally, base64-encodes images, and slices video into frames, matching the OpenAI provider.
- **`embed_content(...)`**
  - **Process:** Not implemented; Anthropic exposes no embeddings API.

---

### File: `core/llm_models/providers/ollama.py`
Implementation of the local Ollama LLM provider.

#### Class: `OllamaProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Communicates with local open-source models using the Ollama SDK, gracefully injecting system prompts, maintaining structured formats via JSON parsing, and handling base64 visual files.

---

### File: `core/llm_models/providers/vllm.py`
Client for a self-hosted vLLM OpenAI-compatible server.

#### Class: `VLLMProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Points the OpenAI SDK at the vLLM server resolved from `VLLM_URL` (default `http://localhost:8000/v1`). Constrains structured output through the standard `response_format` with a `json_schema` derived from the Pydantic model, and passes vLLM-only sampling knobs (`top_k`, `repetition_penalty`) via `extra_body`. Supports streaming with usage accounting.
- **`upload_media(...)`**
  - **Process:** Extracts PDF text locally, base64-encodes images for vision models, and slices video into frames.
- **`embed_content(...)`**
  - **Process:** Calls `/v1/embeddings`, which requires the server to be running an embedding model.

Note: this is a client only. The `vllm` package is never imported, so no GPU runtime is needed to use it. Because vLLM serves arbitrary model names, routing requires the explicit `vllm/` prefix (for example `vllm/meta-llama/Llama-3.1-8B-Instruct`). Self-hosted models have no per-token price, so transactions are recorded at $0.00.

---

### File: `core/llm_models/providers/perplexity.py`
Implementation of the Perplexity LLM provider.

#### Class: `PerplexityProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Uses the OpenAI SDK mapped to Perplexity endpoints. Allows custom query routing featuring `search_recency_filter`, `search_domain_filter`, and robust citation returns.
- **`embed_content(...)`**
  - **Process:** Not implemented natively for this provider.

---

### File: `core/llm_models/utils/media_utils.py`
Tools for decomposing media locally before dispatch.
- **`extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str`**: Extracts textual context from PDF bytes leveraging PyPDF2.
- **`process_video_frames(video_bytes: bytes, frames_per_second: int = 1) -> List[dict]`**: Writes video streams, cycles through frames via OpenCV, and extracts/compresses discrete JPEG images to base64 dictionaries suitable for multi-modal ingestion.
- **`encode_image_base64(image_bytes: bytes, mime_type: str) -> dict`**: Formats raw images into valid base64 strings.

---

### File: `core/modules/base.py`
Defines the standard configuration schema for application modules using Pydantic.

#### Class: `Base` (BaseModel)
- **Core Parameters:** `prompt`, `system_prompt`, `structure`.
- **Model Parameters:** `model` (optional `str`, no default model value), `models` (`list[str]`, default list of Gemini models). Consumed by `ModelRouter` to determine primary and ordered fallback models.
- **Generation:** `temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning_budget`.
- **Sampling:** `presence_penalty`, `frequency_penalty`, `seed`, `stop_sequences`.
- **Provider Features:** `response_mime_type`, `stream`, `logprobs`, `service_tier`, `tools`, `candidate_count`.
- **Search:** `return_citations`, `search_domain_filter`, `search_recency_filter`.

---

## Module: `utils`
General utility functions for environment management, file handling, multi-threading execution, and complex file conversion.

### File: `utils/env_ops.py`
Handles secret management and environment variables natively or via AWS.
- **`get_secret(key_name: str)`**: The entry point providers should use. Routes to `.env` or AWS Secrets Manager based on `KEY_LOCATION`.
- **`get_key_location()`**: Reads `KEY_LOCATION` (`LOCAL` | `AWS_SM`), defaulting to `LOCAL`.
- **`get_local_secret(key_name: str)`**: Reads directly from `.env`.
- **`get_aws_secret(key_name: str, secret_name: str)`**: Reads a single key out of an AWS Secrets Manager bundle.
- **`get_database_url()`**: Convenience wrapper resolving `DATABASE_URL` through `KEY_LOCATION`.
- **`get_gemini_key_type()`**: Reads `GEMINI_KEY_TYPE` (`GEMINI_KEY` | `SERVICE_ACC_JSON`), defaulting to `GEMINI_KEY`.
- **`load_gemini_service_account_credentials()`**: Builds Vertex credentials from `GEMINI_SERVICE_ACCOUNT_FILE` or inline `GEMINI_SERVICE_ACCOUNT_JSON`.
- **`resolve_gemini_project()` / `resolve_gemini_location()`**: Resolve the Vertex project and region with fallbacks to the standard `GOOGLE_CLOUD_*` variables.
- **`get_secret_dict(secret_name: str)`**: Fetches a bulk dictionary of configuration from AWS Secrets Manager utilizing memory caching via `boto3`.
- **`get_keys_dict()`**: Orchestrates global keys seamlessly.

**Config vs secrets:** `KEY_LOCATION` only moves credentials (`DATABASE_URL`, provider keys, provider URLs). Mode switches, project ids and regions are deployment config and are always read from `.env`, since `SECRET_NAME` itself must be readable before Secrets Manager can be reached.

### File: `utils/file_ops.py`
Standardized file operations.
- **`read_file(file_path: str)`**: Reads `utf-8` text.
- **`get_file(file_path: str)`**: Grabs raw binary data.
- **`read_prompt(prompt_title: str)`**: Short-hand fetch for `.txt` files within `core/prompts`.
- **`read_csv(file_path: str)`**: Reads CSV rows utilizing the standard dictionary reader.

### File: `utils/logger.py`
- **`get_logger(name: str, level: Optional[int] = None)`**: Returns a logger honouring `LOGGING_MODE` and `LOGGING_LEVEL`. Under `NORMAL` it builds a dual-channel `logging.Logger` passing standard strings to the CLI and capturing the JSON log structure into `/logs` by date. Under `LAMBDA` it returns a `LambdaLogger` that prints one JSON record per line to stdout, since Lambda has a read-only filesystem and CloudWatch already captures stdout. An explicit `level` argument overrides `LOGGING_LEVEL`.
- **`get_logging_mode()` / `get_logging_level()`**: Parse and validate the two variables, accepting either level names (`DEBUG`, `INFO`, ...) or numeric levels.
- **`LambdaLogger`**: Print-based stand-in mirroring the `Logger` methods used across the project (`debug`/`info`/`warning`/`error`/`critical`/`exception`/`log`), including `%`-style lazy interpolation and traceback capture.

### File: `utils/markitdown_utils.py`
Integrates Microsoft's MarkItDown.
#### Class: `MarkItDownUtils`
- **`convert(source: str)`**: Universal conversion to Markdown supporting paths and URLs.
- **`convert_local(file_path: str)`**: Safe local parsing for varied files (PDF, DOCX, XLSX).
- **`convert_url(url: str)`**: Resolves remote web HTML or YouTube transcripts seamlessly.
- **`convert_image()` / `convert_audio()`**: Specialized methods utilizing LLMs for transcription or detailed multi-modal parsing.

### File: `utils/parallel_executor.py`
Resilient multi-threading utility with rate limiting and retry logic.
- **`calculate_worker_count(max_threads: int = 0, data_size: int = 0)`**: Resolves CPU-scaled or fixed integer bounds.
- **`parallel_execute(target_function: Callable, data: List[Any], max_threads: int = 0, max_req_per_min: Optional[int] = None, max_retries: int = 0, retry_timer: float = 0)`**: Orchestrates multi-threading execution processing individual iterations utilizing a robust rate limiter logic alongside failure catching strategies guaranteeing list continuity dynamically.
