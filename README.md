This project is a template.

# Code-Template-Samples

## Overview
This repository serves as a modular template for building applications that integrate with Large Language Models (LLMs) like Google Gemini and OpenAI. It includes robust utilities for logging, parallel execution, environment management, and PDF processing.

---

## Module: `core`
The `core` module contains the primary logic for interfacing with LLM providers and defining the base structure for application modules.

### File: `core/llm_models/llm_model_base.py`
This file defines the abstract base class and shared utilities for all LLM implementations.

#### Functions
- **`load_pricing(csv_path: str)`**
  - **Input:** `csv_path` (str) - Path to the pricing CSV file.
  - **Output:** `dict` - A dictionary containing model pricing.
  - **Process:** Reads a CSV file containing model cost data and returns a structured dictionary for cost calculation.
- **`LLMModels.prepare_module(module: Any)` (static)**
  - **Input:** `module` (Any) - A class or an instance of `BaseModule`.
  - **Output:** `BaseModule` - An initialized module instance.
  - **Process:** Ensures the module is instantiated and its prompt is loaded from the specified path if not already present.

#### Class: `LLMModels` (ABC)
Abstract base class for LLM providers.
- **`__init__(self, api_key: str, base_config: BaseModule)`**
  - **Input:** `api_key` (str), `base_config` (BaseModule).
  - **Output:** `None`.
  - **Process:** Initializes API credentials and model configuration, and registers an exit handler to print a cost summary upon script termination.
- **`print_final_summary()` (static)**
  - **Input:** `None`.
  - **Output:** `None`.
  - **Process:** Prints a detailed, itemized table of all transactions, including token costs and execution time, and logs the session history in JSON format.
- **`record_transaction(cls, module_name: str, model_name: str, costs: dict, duration: float)` (classmethod)**
  - **Input:** `module_name` (str), `model_name` (str), `costs` (dict), `duration` (float).
  - **Output:** `None`.
  - **Process:** Appends transaction details to the global history and updates session-wide cost totals.
- **`_calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0)`**
  - **Input:** `model_name` (str), `prompt_tokens` (int), `output_tokens` (int), `cached_tokens` (int, default=0).
  - **Output:** `dict` - Calculated costs for input, output, cached tokens, and the total.
  - **Process:** Matches the model name against pricing data and computes estimated costs based on token usage.
- **`model_response(self, module: Any, uploaded_file: Optional[Any] = None)` (abstract)**
- **`upload_pdf(self, pdf_bytes: bytes)` (abstract)**
- **`embed_content(self, input_content: Union[str, List[str]], **kwargs)` (abstract)**
- **`_extract_text_from_pdf_bytes(self, pdf_bytes: bytes)`**
  - **Input:** `pdf_bytes` (bytes).
  - **Output:** `str` - Extracted text.
  - **Process:** A wrapper utility that uses internal PDF operations to extract text from binary PDF data.

---

### File: `core/llm_models/gemini_model.py`
Implementation of the Google Gemini LLM provider.

#### Functions
- **`initialize_gemini(key, module)`**
  - **Input:** `key` (str), `module` (Any).
  - **Output:** `GeminiModel` instance.
  - **Process:** Helper function to prepare the module and initialize the Gemini client.

#### Class: `GeminiModel` (LLMModels)
- **`__init__(self, api_key: str, base: BaseModule)`**
  - **Input:** `api_key` (str), `base` (BaseModule).
  - **Output:** `None`.
  - **Process:** Initializes the Google GenAI client with provided credentials and configuration.
- **`model_response(self, module: Any, uploaded_file: Optional[Any] = None)`**
  - **Input:** `module` (Any), `uploaded_file` (Optional[Any], default=None).
  - **Output:** `Any` - The model's response (either parsed object or raw text).
  - **Process:** Sends the prompt and optional file to Gemini, logs detailed metadata/costs, and returns the response.
- **`upload_pdf(self, pdf_bytes: bytes)`**
  - **Input:** `pdf_bytes` (bytes).
  - **Output:** `types.File` - The uploaded file object.
  - **Process:** Uploads PDF data directly to Gemini's file service for native processing.
- **`embed_content(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", model="gemini-embedding-001", dimensions=1536)`**
  - **Input:** `text` (str), `task_type` (str, default="RETRIEVAL_DOCUMENT"), `model` (str, default="gemini-embedding-001"), `dimensions` (int, default=1536).
  - **Output:** `List[float]` - Vector embedding.
  - **Process:** Generates embeddings for the provided text using Gemini's embedding models.

---

### File: `core/llm_models/openai_model.py`
Implementation of the OpenAI LLM provider.

#### Functions
- **`initialize_openai(key, module)`**
  - **Input:** `key` (str), `module` (Any).
  - **Output:** `OpenAIModel` instance.
  - **Process:** Helper function to prepare the module and initialize the OpenAI client.

#### Class: `OpenAIModel` (LLMModels)
- **`__init__(self, api_key: str, base: BaseModule)`**
  - **Input:** `api_key` (str), `base` (BaseModule).
  - **Output:** `None`.
  - **Process:** Initializes the official OpenAI Python client.
- **`model_response(self, module: Any, uploaded_file: Optional[Any] = None)`**
  - **Input:** `module` (Any), `uploaded_file` (Optional[Any], default=None).
  - **Output:** `Any` - The model's response.
  - **Process:** Manages chat completions, handles reasoning-specific parameters (e.g., for o1/o3 models), calculates costs, and returns the output.
- **`upload_pdf(self, pdf_bytes: bytes)`**
  - **Input:** `pdf_bytes` (bytes).
  - **Output:** `str` - Extracted text.
  - **Process:** Extracts text from the PDF locally to be included as context in the prompt, as OpenAI does not support native PDF uploads via the chat API in the same way as Gemini.
- **`embed_content(self, text: Union[str, List[str]], model="text-embedding-3-small", **kwargs)`**
  - **Input:** `text` (Union[str, List[str]]), `model` (str, default="text-embedding-3-small"), `**kwargs`.
  - **Output:** `Union[List[float], List[List[float]]]` - Vector embedding(s).
  - **Process:** Generates embeddings using OpenAI's embedding API.

---

### File: `core/modules/base.py`
Defines the default configuration for application modules.

#### Class: `Base` (BaseModel)
- **Attributes:**
  - `prompt`: `str` (default: Loaded via `read_prompt("prompt")`).
  - `structure`: `Optional[Any]` (default: `None`) - Expected JSON structure/Pydantic model.
  - `model`: `str` (default: `"gemini-2.5-pro"`) - Target LLM model name.
  - `top_p`: `float` (default: `0.8`).
  - `top_k`: `int` (default: `40`).
  - `temperature`: `float` (default: `0.2`).
  - `reasoning_budget`: `Optional[int]` (default: `None`).
  - `response_mime_type`: `str` (default: `"application/json"`).

---

## Module: `utils`
General utility functions for file handling, environment management, and execution control.

### File: `utils/env_ops.py`
Handles secret management and environment variables.

#### Functions
- **`get_local_secret(key_name: str)`**
  - **Input:** `key_name` (str).
  - **Output:** `str` - The value of the environment variable.
  - **Process:** Loads `.env` file and retrieves the requested key.
- **`get_aws_secret(key_name: str, secret_name: str = "RFP-New", region_name: str = "us-east-1")`**
  - **Input:** `key_name` (str), `secret_name` (str, default="RFP-New"), `region_name` (str, default="us-east-1").
  - **Output:** `Any` - The secret value.
  - **Process:** Retrieves secrets from AWS Secrets Manager using credentials found in the local environment.

---

### File: `utils/file_ops.py`
Standardized file and directory operations.

#### Functions
- **`read_file(file_path: str)`**
  - **Input:** `file_path` (str).
  - **Output:** `str` - Stripped file content.
  - **Process:** Reads a text file with UTF-8 encoding.
- **`get_file(file_path: str)`**
  - **Input:** `file_path` (str).
  - **Output:** `bytes` - Binary content.
  - **Process:** Reads a file in binary mode.
- **`read_prompt(prompt_tite: str)`**
  - **Input:** `prompt_tite` (str).
  - **Output:** `str` - Content of the prompt file.
  - **Process:** Helper to read text files from the `core/prompts` directory.
- **`read_csv(file_path: str)`**
  - **Input:** `file_path` (str).
  - **Output:** `List[Dict[str, str]]` - List of rows as dictionaries.
  - **Process:** Parses a CSV file using `csv.DictReader`.

---

### File: `utils/logger.py`
Logging configuration supporting both human-readable and machine-readable (JSON) formats.

#### Functions
- **`get_logger(name: str, level: int = logging.INFO)`**
  - **Input:** `name` (str), `level` (int, default=`logging.INFO`).
  - **Output:** `logging.Logger`.
  - **Process:** Creates a logger that outputs to the console and to a date-stamped JSON log file in the `logs/` directory.

---

### File: `utils/parallel_executor.py`
Robust multi-threading utility with rate limiting and retry logic.

#### Functions
- **`calculate_worker_count(max_threads: int = 0, data_size: int = 0)`**
  - **Input:** `max_threads` (int, default=0), `data_size` (int, default=0).
  - **Output:** `int` - Number of threads to use.
  - **Process:** Computes thread count based on rules: `0` for unbounded, `-1` for CPU count, `-x` for multiples of CPU count, or a fixed positive integer.
- **`parallel_execute(target_function: Callable, data: List[Any], max_threads: int = 0, max_req_per_min: Optional[int] = None, max_retries: int = 0, retry_timer: float = 0)`**
  - **Input:** `target_function` (Callable), `data` (List), `max_threads` (int, default=0), `max_req_per_min` (Optional[int]), `max_retries` (int, default=0), `retry_timer` (float, default=0).
  - **Output:** `List[Any]` - Ordered list of results or Exception objects.
  - **Process:** Executes the `target_function` across the `data` list in parallel, respecting rate limits and retrying on failure.

---

### File: `utils/pdf_ops.py`
Tools for manipulating PDF files.

#### Functions
- **`split_pages(input_stream: BinaryIO, start_index: Optional[int] = None, end_index: Optional[int] = None)`**
  - **Input:** `input_stream` (BinaryIO), `start_index` (Optional[int], default=None), `end_index` (Optional[int], default=None).
  - **Output:** `BytesIO` - A new PDF stream containing the subset of pages.
  - **Process:** Extracts a specific range of pages from a PDF.
- **`extract_text_from_bytes(pdf_bytes: bytes)`**
  - **Input:** `pdf_bytes` (bytes).
  - **Output:** `str` - Extracted text.
  - **Process:** Iterates through all pages of a PDF and concatenates extracted text.
