"""
Gemini Provider Implementation.
"""

import time
from typing import Optional, List, Any, Union
from google import genai
from google.genai import types

from utils.logging import get_logger
from utils.env import (
    get_secret,
    get_gemini_key_type,
    load_gemini_service_account_credentials,
    resolve_gemini_project,
    resolve_gemini_location,
    GEMINI_KEY_TYPE_API_KEY,
    GEMINI_KEY_TYPE_SERVICE_ACC_JSON,
    GEMINI_API_KEY_NAME,
)
from core.llm_models.base_provider import LLMProvider
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.providers.gemini.media_handler import upload_gemini_media
from core.llm_models.providers.gemini.response_handler import generate_gemini_response, split_thought_parts
from core.modules.base import Base as BaseModule

logger = get_logger("GeminiProvider")

DEFAULT_EMBED_TASK_TYPE: str = "RETRIEVAL_DOCUMENT"
DEFAULT_EMBED_DIMENSIONS: int = 1536
VERTEX_SENTINEL_KEY: str = "vertex-service-account"


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        key_type = get_gemini_key_type()
        self.key_type = key_type
        self.uses_vertex = key_type == GEMINI_KEY_TYPE_SERVICE_ACC_JSON

        if key_type == GEMINI_KEY_TYPE_API_KEY:
            resolved_key = api_key or get_secret(GEMINI_API_KEY_NAME)
        else:
            resolved_key = api_key or VERTEX_SENTINEL_KEY

        super().__init__(resolved_key, base)
        self.client = self._build_client(api_key)

    def _build_client(self, api_key: Optional[str]):
        if self.key_type == GEMINI_KEY_TYPE_API_KEY:
            key = api_key or get_secret(GEMINI_API_KEY_NAME)
            logger.info("Initializing Gemini client with API key auth.")
            return genai.Client(api_key=key)

        credentials = load_gemini_service_account_credentials()
        project = resolve_gemini_project(credentials)
        location = resolve_gemini_location()
        logger.info(
            f"Initializing Gemini client with Vertex AI service-account auth "
            f"(project={project}, location={location})."
        )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def _split_thought_parts(candidates: Any) -> tuple:
        return split_thought_parts(candidates)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        return generate_gemini_response(self, module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return upload_gemini_media(self.client, self.uses_vertex, file_bytes, mime_type)

    def embed_content(
        self,
        text: Union[str, List[str]],
        task_type: str = DEFAULT_EMBED_TASK_TYPE,
        model: Optional[str] = None,
        dimensions: int = DEFAULT_EMBED_DIMENSIONS,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        try:
            raw_model = model or self.model_name
            target_model = raw_model.split("/")[-1] if "/" in raw_model else raw_model
            if not target_model.startswith("models/"):
                target_model = f"models/{target_model}"
            input_texts = [text] if isinstance(text, str) else text
            start_time = time.time()
            embed_config = types.EmbedContentConfig(task_type=task_type)
            if dimensions and dimensions > 0:
                embed_config.output_dimensionality = dimensions
            result = self.client.models.embed_content(
                model=target_model,
                contents=input_texts,
                config=embed_config
            )
            total_duration = time.time() - start_time

            prompt_tokens = 0
            usage = getattr(result, 'usage_metadata', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            else:
                try:
                    token_count_resp = self.client.models.count_tokens(model=model, contents=input_texts)
                    prompt_tokens = token_count_resp.total_tokens
                except Exception:
                    pass

            if prompt_tokens > 0:
                costs = cost_tracker.calculate_cost(raw_model, prompt_tokens, 0, 0)
                cost_tracker.record_transaction(
                    "Embedding",
                    raw_model,
                    costs,
                    total_duration,
                    input_tokens=prompt_tokens,
                    output_tokens=0,
                    cached_tokens=0,
                )

            if isinstance(text, str):
                return result.embeddings[0].values
            return [e.values for e in result.embeddings]
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise
