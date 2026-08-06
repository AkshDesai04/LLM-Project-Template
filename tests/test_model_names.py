"""
Unit tests for canonical model name parsing (core.llm_models.model_names).
"""

import pytest

from core.llm_models.model_names import (
    PROVIDER_ALIASES,
    split_model_name,
    strip_provider_prefix,
)


@pytest.mark.llm
@pytest.mark.unit
def test_split_model_name_prefixed():
    assert split_model_name("gemini/gemini-2.5-pro") == ("google", "gemini-2.5-pro")
    assert split_model_name("google/gemini-2.5-flash") == ("google", "gemini-2.5-flash")
    assert split_model_name("openai/gpt-4o") == ("openai", "gpt-4o")
    assert split_model_name("anthropic/claude-3-5-sonnet") == ("anthropic", "claude-3-5-sonnet")
    assert split_model_name("claude/claude-3-haiku") == ("anthropic", "claude-3-haiku")
    assert split_model_name("perplexity/sonar-pro") == ("perplexity", "sonar-pro")
    assert split_model_name("ollama/llama3") == ("ollama", "llama3")
    assert split_model_name("vllm/meta-llama/Llama-3-8B") == ("vllm", "meta-llama/Llama-3-8B")


@pytest.mark.llm
@pytest.mark.unit
def test_split_model_name_unprefixed():
    assert split_model_name("gpt-4o") == (None, "gpt-4o")
    assert split_model_name("meta-llama/Llama-3-8B") == (None, "meta-llama/Llama-3-8B")


@pytest.mark.llm
@pytest.mark.unit
def test_strip_provider_prefix():
    assert strip_provider_prefix("openai/gpt-4o") == "gpt-4o"
    assert strip_provider_prefix("gemini/gemini-2.5-flash") == "gemini-2.5-flash"
    assert strip_provider_prefix("vllm/custom-org/model-v1") == "custom-org/model-v1"
    assert strip_provider_prefix("llama3") == "llama3"
