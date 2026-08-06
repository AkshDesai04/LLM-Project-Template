"""
Unit tests for PerplexityProvider (core.llm_models.providers.perplexity).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.perplexity.provider import PerplexityProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_perplexity_provider_init():
    base = Base(model="perplexity/sonar-pro")
    with patch("core.llm_models.providers.perplexity.provider.get_secret", return_value="mock_ppxl_key"), patch(
        "openai.OpenAI"
    ) as mock_openai:
        provider = PerplexityProvider(api_key=None, base=base)
        mock_openai.assert_called_once_with(api_key="mock_ppxl_key", base_url="https://api.perplexity.ai")


@pytest.mark.providers
@pytest.mark.unit
def test_perplexity_provider_embed_raises():
    base = Base(model="perplexity/sonar-pro")
    with patch("core.llm_models.providers.perplexity.provider.get_secret", return_value="mock_key"), patch(
        "openai.OpenAI"
    ):
        provider = PerplexityProvider(api_key="mock_key", base=base)
        with pytest.raises(NotImplementedError, match="Perplexity does not natively provide"):
            provider.embed_content("text")


@pytest.mark.providers
@pytest.mark.unit
def test_perplexity_provider_evaluate_response():
    base = Base(model="perplexity/sonar-pro")
    with patch("core.llm_models.providers.perplexity.provider.get_secret", return_value="mock_key"), patch(
        "openai.OpenAI"
    ):
        provider = PerplexityProvider(api_key="mock_key", base=base)

        with patch.object(
            provider, "model_response", return_value='{"score": 9, "reasoning": "Great answer", "improvements": "None"}'
        ):
            res = provider.evaluate_response("What is 2+2?", "4")
            assert res.score == 9
            assert res.reasoning == "Great answer"
