"""
Unit tests for OpenAIProvider (core.llm_models.providers.openai).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.openai.provider import OpenAIProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_openai_provider_init():
    base = Base(model="openai/gpt-4o")
    with patch("core.llm_models.providers.openai.provider.get_secret", return_value="mock_openai_key"), patch(
        "core.llm_models.providers.openai.provider.OpenAI"
    ) as mock_client_cls:
        provider = OpenAIProvider(api_key=None, base=base)
        assert provider.api_key == "mock_openai_key"
        mock_client_cls.assert_called_once_with(api_key="mock_openai_key")


@pytest.mark.providers
@pytest.mark.unit
def test_openai_provider_is_reasoning_model():
    assert OpenAIProvider._is_reasoning_model("o1-preview") is True
    assert OpenAIProvider._is_reasoning_model("o3-mini") is True
    assert OpenAIProvider._is_reasoning_model("gpt-4o") is False


@pytest.mark.providers
@pytest.mark.unit
def test_openai_provider_embed_content():
    base = Base(model="openai/text-embedding-3-small")
    mock_client = MagicMock()
    mock_data = MagicMock()
    mock_data.embedding = [0.01, 0.02, 0.03]
    mock_resp = MagicMock()
    mock_resp.data = [mock_data]
    mock_resp.usage.prompt_tokens = 5
    mock_client.embeddings.create.return_value = mock_resp

    with patch("core.llm_models.providers.openai.provider.OpenAI", return_value=mock_client):
        provider = OpenAIProvider(api_key="mock_key", base=base)
        vec = provider.embed_content("hello")
        assert vec == [0.01, 0.02, 0.03]


@pytest.mark.providers
@pytest.mark.unit
def test_openai_provider_upload_media():
    base = Base(model="openai/gpt-4o")
    with patch("core.llm_models.providers.openai.provider.OpenAI"):
        provider = OpenAIProvider(api_key="mock_key", base=base)
        with patch("core.llm_models.providers.openai.provider.process_openai_media", return_value={"type": "image"}) as mock_proc:
            res = provider.upload_media(b"bytes", "image/png")
            assert res == {"type": "image"}
            mock_proc.assert_called_once_with(b"bytes", "image/png")
