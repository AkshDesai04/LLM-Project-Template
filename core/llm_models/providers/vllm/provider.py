"""
vLLM Provider Implementation.
"""

import time
from typing import Optional, List, Any, Union

from utils.logging import get_logger
from utils.env import get_secret
from core.llm_models.base_provider import LLMProvider
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.providers.vllm.media_handler import process_vllm_media
from core.llm_models.providers.vllm.response_handler import generate_vllm_response
from core.modules.base import Base as BaseModule

logger = get_logger("VLLMProvider")

DEFAULT_VLLM_URL: str = "http://localhost:8000/v1"
PLACEHOLDER_API_KEY: str = "EMPTY"


class VLLMProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_secret("VLLM_KEY", raise_error=False) or PLACEHOLDER_API_KEY
        super().__init__(api_key, base)
        self.base_url = get_secret("VLLM_URL", raise_error=False) or DEFAULT_VLLM_URL
        logger.info(f"Initializing vLLM client with base URL: {self.base_url}")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError:
            self.client = None

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        if self.client is None:
            raise ImportError("openai package is required for vLLM routing. Please install 'openai'.")
        return generate_vllm_response(self, module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return process_vllm_media(file_bytes, mime_type)

    def embed_content(self, text: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        if self.client is None:
            raise ImportError("openai package is required for vLLM routing.")
        try:
            model = model or self.model_name
            input_data = [text] if isinstance(text, str) else text
            start_time = time.time()
            response = self.client.embeddings.create(model=model, input=input_data, **kwargs)
            total_duration = time.time() - start_time

            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0

            costs = cost_tracker.calculate_cost(model, prompt_tokens, 0)
            cost_tracker.record_transaction(
                "Embedding", model, costs, total_duration,
                input_tokens=prompt_tokens, output_tokens=0, cached_tokens=0,
            )

            if isinstance(text, str):
                return response.data[0].embedding
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"vLLM embedding failed: {e}")
            raise
