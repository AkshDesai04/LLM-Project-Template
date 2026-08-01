from typing import Any, Literal

from pydantic import BaseModel, Field

# Default Configuration Values
DEFAULT_MODELS: list[str] = [
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
]
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_P: float = 0.8
DEFAULT_TOP_K: int = 40
DEFAULT_PRESENCE_PENALTY: float = 0.0
DEFAULT_FREQUENCY_PENALTY: float = 0.0
DEFAULT_RESPONSE_MIME_TYPE: str = "application/json"
DEFAULT_STREAM: bool = False
DEFAULT_LOGPROBS: bool = False
DEFAULT_CANDIDATE_COUNT: int = 1
DEFAULT_RETURN_CITATIONS: bool = True


class Base(BaseModel):
    # Core
    prompt: str | None = None
    system_prompt: str | None = None
    structure: Any | None = None

    # Model Selection. Canonical form is "provider/model"; see PROVIDER_ALIASES
    # in core/llm_models/router.py for the accepted prefixes.
    model: str | None = None
    models: list[str] = Field(default_factory=lambda: list(DEFAULT_MODELS))

    # Generation
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_tokens: int | None = None
    reasoning_budget: int | Literal["minimal", "low", "medium", "high", "xhigh"] | None = None
    return_reasoning: bool = False

    # Penalties & Sampling
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    seed: int | None = None
    stop_sequences: list[str] | None = None

    # Response
    response_mime_type: str = DEFAULT_RESPONSE_MIME_TYPE
    stream: bool = DEFAULT_STREAM

    # Logging / Debugging
    logprobs: bool = DEFAULT_LOGPROBS
    top_logprobs: int | None = None

    # Provider Features
    service_tier: Literal["auto", "default"] | None = None
    candidate_count: int = DEFAULT_CANDIDATE_COUNT
    safety_settings: Any | None = None
    tools: Any | None = None

    # Search / Retrieval
    return_citations: bool = DEFAULT_RETURN_CITATIONS
    search_domain_filter: list[str] | None = None
    search_recency_filter: str | None = None
