"""
Unit tests for AnthropicProvider (core.llm_models.providers.anthropic).
"""

from unittest.mock import MagicMock, patch
import pytest

from core.llm_models.providers.anthropic.provider import AnthropicProvider
from core.modules.base import Base


@pytest.mark.providers
@pytest.mark.unit
def test_anthropic_provider_init():
    base = Base(model="anthropic/claude-3-5-sonnet")
    mock_anthropic_module = MagicMock()
    with patch("core.llm_models.providers.anthropic.provider.get_secret", return_value="mock_anthropic_key"), patch.dict(
        "sys.modules", {"anthropic": mock_anthropic_module}
    ):
        provider = AnthropicProvider(api_key=None, base=base)
        assert provider.api_key == "mock_anthropic_key"
        mock_anthropic_module.Anthropic.assert_called_once_with(api_key="mock_anthropic_key")


@pytest.mark.providers
@pytest.mark.unit
def test_anthropic_provider_embed_raises():
    base = Base(model="anthropic/claude-3-5-sonnet")
    mock_anthropic_module = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        provider = AnthropicProvider(api_key="mock_key", base=base)
        with pytest.raises(NotImplementedError, match="Anthropic does not provide an embeddings API"):
            provider.embed_content("text")


@pytest.mark.providers
@pytest.mark.unit
def test_anthropic_provider_upload_media():
    base = Base(model="anthropic/claude-3-5-sonnet")
    mock_anthropic_module = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        provider = AnthropicProvider(api_key="mock_key", base=base)
        with patch("core.llm_models.providers.anthropic.provider.process_anthropic_media", return_value={"type": "image"}) as mock_proc:
            res = provider.upload_media(b"bytes", "image/png")
            assert res == {"type": "image"}
            mock_proc.assert_called_once_with(b"bytes", "image/png")
