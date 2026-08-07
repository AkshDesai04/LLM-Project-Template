"""
Unit tests for GeminiProvider (core.llm_models.providers.gemini).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.gemini.provider import GeminiProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_gemini_provider_init_api_key():
    base = Base(model="gemini/gemini-2.5-flash")
    with patch("core.llm_models.providers.gemini.provider.get_gemini_key_type", return_value="GEMINI_KEY"), patch(
        "core.llm_models.providers.gemini.provider.get_secret", return_value="mock_gemini_key"
    ), patch("core.llm_models.providers.gemini.provider.genai.Client") as mock_client_cls:
        provider = GeminiProvider(api_key=None, base=base)
        assert provider.uses_vertex is False
        mock_client_cls.assert_called_once_with(api_key="mock_gemini_key")


@pytest.mark.providers
@pytest.mark.unit
def test_gemini_provider_init_vertex():
    base = Base(model="gemini/gemini-2.5-pro")
    mock_creds = MagicMock()
    with patch("core.llm_models.providers.gemini.provider.get_gemini_key_type", return_value="SERVICE_ACC_JSON"), patch(
        "core.llm_models.providers.gemini.provider.load_gemini_service_account_credentials", return_value=mock_creds
    ), patch("core.llm_models.providers.gemini.provider.resolve_gemini_project", return_value="proj-1"), patch(
        "core.llm_models.providers.gemini.provider.resolve_gemini_location", return_value="us-central1"
    ), patch("core.llm_models.providers.gemini.provider.genai.Client") as mock_client_cls:
        provider = GeminiProvider(api_key=None, base=base)
        assert provider.uses_vertex is True
        mock_client_cls.assert_called_once_with(
            vertexai=True,
            project="proj-1",
            location="us-central1",
            credentials=mock_creds,
        )


@pytest.mark.providers
@pytest.mark.unit
def test_gemini_provider_embed_content():
    base = Base(model="gemini/text-embedding-004")
    mock_client = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1, 0.2, 0.3]
    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_emb]
    mock_resp.usage_metadata.prompt_token_count = 10
    mock_client.models.embed_content.return_value = mock_resp

    with patch("core.llm_models.providers.gemini.provider.get_gemini_key_type", return_value="GEMINI_KEY"), patch(
        "core.llm_models.providers.gemini.provider.get_secret", return_value="mock_key"
    ), patch("core.llm_models.providers.gemini.provider.genai.Client", return_value=mock_client), patch(
        "core.llm_models.providers.gemini.provider.types.EmbedContentConfig"
    ):
        provider = GeminiProvider(api_key="mock_key", base=base)
        vec = provider.embed_content("sample text")
        assert vec == [0.1, 0.2, 0.3]
