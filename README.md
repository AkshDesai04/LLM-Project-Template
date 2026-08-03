# LLM-Project-Template

## Overview
This repository serves as a modular template for building applications that integrate with Large Language Models (LLMs) like Google Gemini, OpenAI, Anthropic, Ollama, vLLM, and Perplexity. It includes robust utilities for logging, parallel execution, database connections, environment management, and document processing via MarkItDown.

## Key Architectural Principles
- **200 Lines Limit:** Every Python file in the repository is strictly under 200 lines long.
- **Absolute Imports:** All internal module imports use absolute import paths (e.g. `from utils.logging import get_logger`). Relative imports are forbidden.
- **Uniform Provider Subpackages:** Each model provider in `core/llm_models/providers/` is structured as a dedicated subpackage (`__init__.py`, `provider.py`, `media_handler.py`, `response_handler.py`).

---

## Module: `core`
The `core` module contains the primary logic for interfacing with LLM providers and defining the base structure for application modules.

### File: `core/llm_models/base_provider.py`
This file defines the abstract base class and shared utilities for all LLM implementations.

#### Class: `JudgeResult` (BaseModel)
Defines the structure for LLM-as-a-judge outputs.
- `score` (int): 1-10 evaluating response quality.
- `reasoning` (str): Detailed explanation.
- `improvements` (Optional[str]): Suggestions.

#### Class: `LLMProvider` (ABC)
Abstract base class for LLM providers.
- **`__init__(self, api_key: str, base_config: BaseModule)`**
- **`model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any` (abstract)**
- **`upload_media(self, file_bytes: bytes, mime_type: str) -> Any` (abstract)**
- **`embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]` (abstract)**
- **`evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult` (abstract)**

---

### File: `core/llm_models/cost_tracker.py` & `cost_summary.py`
Singleton utility for centralized tracking, calculation, and reporting of API usage costs.

---

### File: `core/llm_models/router.py`
Unified entry point for dynamic model selection and routing across multiple LLM backends. Always call models via `ModelRouter`.

---

### Directory: `core/llm_models/providers/`
Contains provider-specific modular implementations (`gemini`, `openai`, `anthropic`, `ollama`, `vllm`, `perplexity`).

---

## Module: `utils`
General utility functions for environment management, file handling, database connections, multi-threading execution, and complex file conversion.
- `utils/env/`: Secrets manager supporting `.env` (`LOCAL`) and AWS Secrets Manager (`AWS_SM`).
- `utils/logging/`: Logging framework supporting console/file output (`NORMAL`) and CloudWatch/stdout output (`LAMBDA`).
- `utils/db/`: Database connectors for SQLite, PostgreSQL, and MySQL.
- `utils/io/`: Text, prompt, and CSV readers.
- `utils/concurrency/`: Rate-limited multi-threaded parallel execution engine.
- `utils/document/`: Document conversion utility using MarkItDown.
