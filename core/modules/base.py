from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class Base(BaseModel):
    # Core
    prompt: ClassVar[str | None] = None
    system_prompt: ClassVar[str | None] = None
    structure: ClassVar[Any | None] = None

    # Model Selection
    model: ClassVar[str] = "gemini-2.5-pro"

    fallback_models: ClassVar[list[str]] = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    # Generation
    temperature: ClassVar[float] = 0.2
    top_p: ClassVar[float] = 0.8
    top_k: ClassVar[int] = 40
    max_tokens: ClassVar[int | None] = None
    reasoning_budget: ClassVar[int | Literal["low", "medium", "high"] | None] = None

    # Penalties & Sampling
    presence_penalty: ClassVar[float] = 0.0
    frequency_penalty: ClassVar[float] = 0.0
    seed: ClassVar[int | None] = None
    stop_sequences: ClassVar[list[str] | None] = None

    # Response
    response_mime_type: ClassVar[str] = "application/json"
    stream: ClassVar[bool] = False

    # Logging / Debugging
    logprobs: ClassVar[bool] = False
    top_logprobs: ClassVar[int | None] = None

    # Provider Features
    service_tier: ClassVar[Literal["auto", "default"] | None] = None
    candidate_count: ClassVar[int] = 1
    safety_settings: ClassVar[Any | None] = None
    tools: ClassVar[Any | None] = None

    # Search / Retrieval
    return_citations: ClassVar[bool] = True
    search_domain_filter: ClassVar[list[str] | None] = None
    search_recency_filter: ClassVar[str | None] = None