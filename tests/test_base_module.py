"""
Unit tests for core.modules.base (Base module prompt configuration schema).
"""

from typing import List
from pydantic import BaseModel
import pytest

from core.modules.base import (
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_LOGPROBS,
    DEFAULT_MODELS,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_RESPONSE_MIME_TYPE,
    DEFAULT_STREAM,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    Base,
)


class SampleStructure(BaseModel):
    title: str
    items: List[str]


@pytest.mark.llm
@pytest.mark.unit
def test_base_default_values():
    module = Base()
    assert module.prompt is None
    assert module.system_prompt is None
    assert module.structure is None
    assert module.model is None
    assert module.models == DEFAULT_MODELS
    assert module.temperature == DEFAULT_TEMPERATURE
    assert module.top_p == DEFAULT_TOP_P
    assert module.top_k == DEFAULT_TOP_K
    assert module.presence_penalty == DEFAULT_PRESENCE_PENALTY
    assert module.frequency_penalty == DEFAULT_FREQUENCY_PENALTY
    assert module.response_mime_type == DEFAULT_RESPONSE_MIME_TYPE
    assert module.stream == DEFAULT_STREAM
    assert module.logprobs == DEFAULT_LOGPROBS
    assert module.candidate_count == DEFAULT_CANDIDATE_COUNT


@pytest.mark.llm
@pytest.mark.unit
def test_base_custom_values():
    module = Base(
        prompt="Explain quantum computing",
        system_prompt="You are a physics expert.",
        structure=SampleStructure,
        model="openai/gpt-4o",
        temperature=0.7,
        max_tokens=500,
        reasoning_budget="high",
        return_reasoning=True,
        stop_sequences=["END", "STOP"],
        search_domain_filter=["arxiv.org"],
    )

    assert module.prompt == "Explain quantum computing"
    assert module.system_prompt == "You are a physics expert."
    assert module.structure == SampleStructure
    assert module.model == "openai/gpt-4o"
    assert module.temperature == 0.7
    assert module.max_tokens == 500
    assert module.reasoning_budget == "high"
    assert module.return_reasoning is True
    assert module.stop_sequences == ["END", "STOP"]
    assert module.search_domain_filter == ["arxiv.org"]


@pytest.mark.llm
@pytest.mark.unit
def test_base_subclassing():
    class CustomPrompt(Base):
        prompt: str = "Subclassed Prompt"
        model: str = "anthropic/claude-3-5-sonnet"
        stream: bool = True

    cp = CustomPrompt()
    assert cp.prompt == "Subclassed Prompt"
    assert cp.model == "anthropic/claude-3-5-sonnet"
    assert cp.stream is True
