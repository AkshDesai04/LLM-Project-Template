This project is a template.

# LLM-Project-Template

## Overview
This repository serves as a modular template for building applications that integrate with Large Language Models (LLMs) like Google Gemini, OpenAI, Ollama, and Perplexity. It includes robust utilities for logging, parallel execution, environment management, and document processing via MarkItDown.

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
- **`record_transaction(self, module_name: str, model_name: str, costs: dict, duration: float)`**
  - **Input:** `module_name` (str), `model_name` (str), `costs` (dict), `duration` (float).
  - **Process:** Appends transaction details to the global history and updates overall session metrics.
- **`print_final_summary(self)`**
  - **Process:** Registered via `atexit`. Prints a detailed, itemized tabular summary of all session costs and execution times upon script exit, and logs the full history as JSON.

---

### File: `core/llm_models/router.py`
Unified entry point for dynamic model selection and routing across multiple LLM backends.

#### Class: `ModelRouter`
- **`__init__(self, module: BaseModule, api_keys: Optional[dict] = None, fallback_index: int = 0)`**
  - **Process:** Resolves the intended model name (handling fallback arrays), identifies the necessary provider, lazily imports the respective provider's class, and initializes the `LLMProvider` instance.
- **`get_provider_by_model_name(model_name: str) -> str` (static)**
  - **Input:** `model_name` (str).
  - **Output:** `str` - Provider name (e.g., "openai", "google", "ollama", "perplexity").
  - **Process:** Routes requests by matching model string prefixes.

---

### File: `core/llm_models/providers/gemini.py`
Implementation of the Google Gemini LLM provider.

#### Class: `GeminiProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Sends prompts/files to Gemini using the `google-genai` SDK. Supports advanced parameters, reasoning budgets via `ThinkingConfig`, structured JSON output, native file caching metadata, and streaming generators.
- **`upload_media(...)`**
  - **Process:** Uploads bytes natively to Gemini's File API, utilizing a polling loop to verify the file is active before proceeding.
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

### File: `core/llm_models/providers/ollama.py`
Implementation of the local Ollama LLM provider.

#### Class: `OllamaProvider` (LLMProvider)
- **`model_response(...)`**
  - **Process:** Communicates with local open-source models using the Ollama SDK, gracefully injecting system prompts, maintaining structured formats via JSON parsing, and handling base64 visual files.

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
- **Model Parameters:** `model` (default `"gemini-2.5-pro"`), `fallback_models`.
- **Generation:** `temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning_budget`.
- **Sampling:** `presence_penalty`, `frequency_penalty`, `seed`, `stop_sequences`.
- **Provider Features:** `response_mime_type`, `stream`, `logprobs`, `service_tier`, `tools`, `candidate_count`.
- **Search:** `return_citations`, `search_domain_filter`, `search_recency_filter`.

---

## Module: `utils`
General utility functions for environment management, file handling, multi-threading execution, and complex file conversion.

### File: `utils/env_ops.py`
Handles secret management and environment variables natively or via AWS.
- **`get_local_secret(key_name: str)`**: Reads directly from `.env`.
- **`get_secret_dict(secret_name: str)`**: Fetches a bulk dictionary of configuration from AWS Secrets Manager utilizing memory caching via `boto3`.
- **`get_keys_dict()`**: Orchestrates global keys seamlessly.

### File: `utils/file_ops.py`
Standardized file operations.
- **`read_file(file_path: str)`**: Reads `utf-8` text.
- **`get_file(file_path: str)`**: Grabs raw binary data.
- **`read_prompt(prompt_tite: str)`**: Short-hand fetch for `.txt` files within `core/prompts`.
- **`read_csv(file_path: str)`**: Reads CSV rows utilizing the standard dictionary reader.

### File: `utils/logger.py`
- **`get_logger(name: str, level: int = logging.INFO)`**: Builds a dual-channel logger passing standard strings to the CLI, and capturing the robust JSON log structure internally to the `/logs` directory based on the active date timestamp.

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
