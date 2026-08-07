"""
Anthropic Provider Implementation.
"""

from typing import Optional, List, Any, Union

from utils.logging import get_logger
from utils.env import get_secret
from core.llm_models.base_provider import LLMProvider
from core.llm_models.providers.anthropic.media_handler import process_anthropic_media
from core.llm_models.providers.anthropic.response_handler import generate_anthropic_response
from core.modules.base import Base as BaseModule

logger = get_logger("AnthropicProvider")


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_secret("ANTHROPIC_KEY", raise_error=False)
        super().__init__(api_key, base)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            self.client = None

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        if self.client is None:
            raise ImportError("anthropic is not installed. Please install 'anthropic' package to use AnthropicProvider.")
        return generate_anthropic_response(self, module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return process_anthropic_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Use the OpenAI or Gemini provider for embeddings."
        )
