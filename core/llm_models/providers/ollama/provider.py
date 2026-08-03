"""
Ollama Provider Implementation.
"""

import time
from typing import Optional, List, Any, Union

from utils.logging import get_logger
from utils.env import get_secret
from core.llm_models.base_provider import LLMProvider
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.providers.ollama.media_handler import process_ollama_media
from core.llm_models.providers.ollama.response_handler import generate_ollama_response
from core.modules.base import Base as BaseModule

logger = get_logger("OllamaProvider")

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
DEFAULT_OLLAMA_KEY: str = "local-key"


class OllamaProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_secret("OLLAMA_KEY", raise_error=False) or DEFAULT_OLLAMA_KEY
        super().__init__(api_key, base)
        ollama_url = get_secret("OLLAMA_URL", raise_error=False) or DEFAULT_OLLAMA_URL
        logger.info(f"Initializing Ollama client with host: {ollama_url}")
        try:
            from ollama import Client
            self.client = Client(host=ollama_url)
        except ImportError:
            self.client = None

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, system_prompt: Optional[str] = None, **kwargs) -> Any:
        if self.client is None:
            raise ImportError("ollama package is not installed. Please install 'ollama' package to use OllamaProvider.")
        return generate_ollama_response(self, module, uploaded_file, system_prompt, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return process_ollama_media(file_bytes, mime_type)

    def embed_content(self, text: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        if self.client is None:
            raise ImportError("ollama package is not installed.")
        try:
            model = model or self.model_name
            input_texts = [text] if isinstance(text, str) else text
            embeddings = []
            start_time = time.time()

            for t in input_texts:
                resp = self.client.embeddings(model=model, prompt=t)
                embeddings.append(resp['embedding'])

            total_duration = time.time() - start_time
            prompt_tokens = sum(max(1, len(t) // 4) for t in input_texts)

            costs = cost_tracker.calculate_cost(model, prompt_tokens, 0)
            cost_tracker.record_transaction(
                "Embedding", model, costs, total_duration,
                input_tokens=prompt_tokens, output_tokens=0, cached_tokens=0,
            )

            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise
