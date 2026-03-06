import os
from typing import Any, Optional, List, Union

from utils.logger import get_logger
from utils.file_ops import read_csv
from .llm_model_base import LLMModels, JudgeResult
from .gemini_model import initialize_gemini
from .openai_model import initialize_openai
from ..modules.base import Base as BaseModule

logger = get_logger("ModelRouter")

_PROVIDER_MAPPING = None

def _load_provider_mapping() -> dict:
    """Loads a mapping from model_id to model_provider from the pricing CSV."""
    global _PROVIDER_MAPPING
    if _PROVIDER_MAPPING is not None:
        return _PROVIDER_MAPPING

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pricing_file = os.path.join(project_root, "assets", "model_pricing.csv")

        mapping = {}
        if not os.path.exists(pricing_file):
            logger.error(f"Pricing file not found at {pricing_file}")
            _PROVIDER_MAPPING = {}
            return {}
            
        rows = read_csv(pricing_file)
        for row in rows:
            if not row.get('sr', '').isdigit():
                continue
            model_id = row.get('model_id')
            provider = row.get('model_provider')
            if model_id and provider:
                mapping[model_id] = provider.lower()
        _PROVIDER_MAPPING = mapping
        return mapping
    except Exception as e:
        logger.error(f"Failed to load model provider mapping: {e}")
        _PROVIDER_MAPPING = {}
        return {}


class ModelRouter:
    def __init__(self, module: BaseModule, api_keys: dict, fallback_index: int = 0):
        """
        Initializes a router that selects, instantiates, and proxies a specific
        LLM client based on the module configuration and fallback index.
        """
        provider_mapping = _load_provider_mapping()

        model_name = ""
        if fallback_index == 0:
            model_name = module.model
        else:
            if not module.fall_back_models:
                raise ValueError("fallback_index is non-zero, but no fallback models are defined.")
            try:
                model_name = module.fall_back_models[fallback_index - 1]
            except IndexError:
                raise ValueError(f"Fallback index {fallback_index} is out of range.")

        module_for_init = module.copy(update={'model': model_name})
        provider = provider_mapping.get(model_name)

        if not provider:
            for key, value in provider_mapping.items():
                if key in model_name:
                    provider = value
                    logger.info(f"Found provider '{provider}' from similar model '{key}'.")
                    break

        if not provider:
            raise ValueError(f"Could not determine provider for model '{model_name}'.")

        logger.info(f"Routing to provider: {provider} for model '{model_name}'")

        self.model_instance: LLMModels
        if provider == 'google':
            api_key = api_keys.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Provider is 'google', but GEMINI_API_KEY not in api_keys.")
            self.model_instance = initialize_gemini(api_key, module_for_init)
        elif provider == 'openai':
            api_key = api_keys.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Provider is 'openai', but OPENAI_API_KEY not in api_keys.")
            self.model_instance = initialize_openai(api_key, module_for_init)
        else:
            raise ValueError(f"Unsupported provider: '{provider}'")

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Any:
        """Forwards the call to the underlying model instance."""
        return self.model_instance.model_response(module, uploaded_file)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        """Forwards the call to the underlying model instance."""
        return self.model_instance.upload_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        """Forwards the call to the underlying model instance."""
        return self.model_instance.embed_content(input_content, **kwargs)

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        """Forwards the call to the underlying model instance."""
        return self.model_instance.evaluate_response(input_prompt, generated_output, rubric)
