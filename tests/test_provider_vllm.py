"""
Unit tests for VLLMProvider (core.llm_models.providers.vllm).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.vllm.provider import VLLMProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_vllm_provider_init():
    base = Base(model="vllm/meta-llama/Llama-3-8B")
    with patch("core.llm_models.providers.vllm.provider.get_secret", side_effect=lambda k, **kw: "http://custom-vllm:8000/v1" if k == "VLLM_URL" else None), patch(
        "openai.OpenAI"
    ) as mock_openai:
        provider = VLLMProvider(api_key=None, base=base)
        assert provider.base_url == "http://custom-vllm:8000/v1"
        mock_openai.assert_called_once_with(api_key="EMPTY", base_url="http://custom-vllm:8000/v1")


@pytest.mark.providers
@pytest.mark.unit
def test_vllm_provider_embed_content():
    base = Base(model="vllm/meta-llama/Llama-3-8B")
    with patch("core.llm_models.providers.vllm.provider.get_secret", return_value=None), patch(
        "openai.OpenAI"
    ) as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_emb = MagicMock()
        mock_emb.embedding = [0.5, 0.6]
        mock_resp = MagicMock()
        mock_resp.data = [mock_emb]
        mock_resp.usage.prompt_tokens = 4
        mock_client.embeddings.create.return_value = mock_resp

        provider = VLLMProvider(api_key="EMPTY", base=base)
        res = provider.embed_content("test text")
        assert res == [0.5, 0.6]
