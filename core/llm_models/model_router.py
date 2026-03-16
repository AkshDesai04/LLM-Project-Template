import os
from typing import Any, Optional, List, Union

from utils.logger import get_logger
from .llm_model_base import LLMModels, JudgeResult
from .gemini_model import initialize_gemini
from .openai_model import initialize_openai
from ..modules.base import Base as BaseModule

logger = get_logger("ModelRouter")


class ModelRouter:
    def __init__(self, module: BaseModule, api_keys: dict, fallback_index: int = 0):
        model_name = ""
        if fallback_index == 0:
            model_name = module.model
        else:
            fall_back_models = getattr(module, 'fall_back_models', None)
            if not fall_back_models:
                raise ValueError("fallback_index is non-zero, but no fallback models are defined.")
            try:
                model_name = fall_back_models[fallback_index - 1]
            except IndexError:
                raise ValueError(f"Fallback index {fallback_index} is out of range.")

        module_for_init = module.copy(update={'model': model_name})
        
        # Determine provider based on model name prefix
        provider = LLMModels.get_provider_by_model_name(model_name)

        logger.info(f"Routing to provider: {provider} for model '{model_name}'")

        self.model_instance: LLMModels
        if provider == 'google':
            api_key = api_keys.get("GEMINI_KEY")
            if not api_key:
                raise ValueError("Provider is 'google', but GEMINI_KEY not in api_keys.")
            self.model_instance = initialize_gemini(api_key, module_for_init)
        elif provider == 'openai':
            api_key = api_keys.get("OPEN_AI_KEY")
            if not api_key:
                raise ValueError("Provider is 'openai', but OPENAI_KEY not in api_keys.")
            self.model_instance = initialize_openai(api_key, module_for_init)
        else:
            raise ValueError(f"Unsupported provider: '{provider}'")

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Any:
        return self.model_instance.model_response(module, uploaded_file)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return self.model_instance.upload_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        return self.model_instance.embed_content(input_content, **kwargs)

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        return self.model_instance.evaluate_response(input_prompt, generated_output, rubric)
