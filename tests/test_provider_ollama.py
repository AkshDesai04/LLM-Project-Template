"""
Unit tests for OllamaProvider (core.llm_models.providers.ollama).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.ollama.provider import OllamaProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_ollama_provider_init():
    base = Base(model="ollama/llama3")
    mock_ollama_module = MagicMock()
    with patch("core.llm_models.providers.ollama.provider.get_secret", return_value=None), patch.dict(
        "sys.modules", {"ollama": mock_ollama_module}
    ):
        provider = OllamaProvider(api_key=None, base=base)
        mock_ollama_module.Client.assert_called_once_with(host="http://localhost:11434")


@pytest.mark.providers
@pytest.mark.unit
def test_ollama_provider_embed_content():
    base = Base(model="ollama/nomic-embed-text")
    mock_client = MagicMock()
    mock_client.embeddings.return_value = {"embedding": [0.1, 0.2]}
    mock_ollama_module = MagicMock()
    mock_ollama_module.Client.return_value = mock_client

    with patch("core.llm_models.providers.ollama.provider.get_secret", return_value=None), patch.dict(
        "sys.modules", {"ollama": mock_ollama_module}
    ):
        provider = OllamaProvider(api_key="local-key", base=base)
        res = provider.embed_content("test text")
        assert res == [0.1, 0.2]
