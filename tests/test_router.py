"""
Unit tests for ModelRouter factory and fallback orchestrator (core.llm_models.router).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.base_provider import LLMProvider
from core.llm_models.router import ModelRouter
from core.modules.base import Base


@pytest.mark.llm
@pytest.mark.unit
def test_get_provider_by_model_name():
    # Prefixed
    assert ModelRouter.get_provider_by_model_name("openai/gpt-4o") == "openai"
    assert ModelRouter.get_provider_by_model_name("gemini/gemini-2.5-pro") == "google"

    # Bare names inferred
    assert ModelRouter.get_provider_by_model_name("gpt-4o") == "openai"
    assert ModelRouter.get_provider_by_model_name("gemini-2.5-flash") == "google"
    assert ModelRouter.get_provider_by_model_name("claude-3-5-sonnet") == "anthropic"
    assert ModelRouter.get_provider_by_model_name("sonar-pro") == "perplexity"
    assert ModelRouter.get_provider_by_model_name("mistral-7b") == "ollama"

    # Unknown bare name
    with pytest.raises(ValueError, match="Could not determine provider"):
        ModelRouter.get_provider_by_model_name("custom-model-xyz")


@pytest.mark.llm
@pytest.mark.unit
def test_router_fallback_chain_success():
    class TestModule(Base):
        models: list[str] = ["openai/gpt-4o", "gemini/gemini-2.5-flash"]

    module = TestModule()

    mock_openai = MagicMock(spec=LLMProvider)
    mock_openai.model_response.side_effect = RuntimeError("OpenAI rate limit")

    mock_gemini = MagicMock(spec=LLMProvider)
    mock_gemini.model_response.return_value = "Gemini Fallback Success"

    def mock_build(model_name, mod):
        if "gpt-4o" in model_name:
            return mock_openai
        return mock_gemini

    with patch.object(ModelRouter, "_build_provider", side_effect=mock_build):
        router = ModelRouter(module)
        res = router.model_response(module)
        assert res == "Gemini Fallback Success"


@pytest.mark.llm
@pytest.mark.unit
def test_router_all_models_fail():
    class TestModule(Base):
        models: list[str] = ["openai/gpt-4o", "gemini/gemini-2.5-flash"]

    module = TestModule()

    mock_fail = MagicMock(spec=LLMProvider)
    mock_fail.model_response.side_effect = RuntimeError("All failed")

    with patch.object(ModelRouter, "_build_provider", return_value=mock_fail):
        router = ModelRouter(module)
        with pytest.raises(RuntimeError, match="Failed to get response after trying all models"):
            router.model_response(module)


@pytest.mark.llm
@pytest.mark.unit
def test_router_delegation():
    class TestModule(Base):
        model: str = "openai/gpt-4o"

    module = TestModule()
    mock_provider = MagicMock(spec=LLMProvider)
    mock_provider.upload_media.return_value = "media_ref"
    mock_provider.embed_content.return_value = [0.1, 0.2]
    mock_provider.evaluate_response.return_value = "judge_res"

    with patch.object(ModelRouter, "_build_provider", return_value=mock_provider):
        router = ModelRouter(module)
        assert router.upload_media(b"bytes", "image/png") == "media_ref"
        assert router.embed_content("text") == [0.1, 0.2]
        assert router.evaluate_response("p", "o") == "judge_res"
