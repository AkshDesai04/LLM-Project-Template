from typing import Any, Optional, List, Union

from utils.logger import get_logger
from utils.env_ops import get_local_secret
from core.modules.base import Base as BaseModule
from .base_provider import LLMProvider, JudgeResult

logger = get_logger("ModelRouter")

class ModelRouter:
    def __init__(self, module: BaseModule, api_keys: Optional[dict] = None, fallback_index: int = 0):
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

        module_for_init = module.model_copy(update={'model': model_name})
        
        provider = self.get_provider_by_model_name(model_name)
        model_name_lower = model_name.lower()

        # Strip provider prefix if present (e.g., 'ollama/llama3.2:1b' -> 'llama3.2:1b')
        if "/" in model_name:
            # Currently only ollama/ is supported for stripping this way in this router
            if model_name_lower.startswith("ollama/"):
                model_name = model_name.split("/", 1)[1]
                module_for_init = module.model_copy(update={'model': model_name})

        logger.info(f"Routing to provider: {provider} for model '{model_name}'")

        self.model_instance: LLMProvider
        
        # Lazy load providers to avoid importing SDKs if not needed
        if provider == 'google':
            api_key = get_local_secret("GEMINI_KEY", raise_error=False)
            if not api_key and api_keys:
                api_key = api_keys.get("GEMINI_KEY")
            
            if not api_key:
                raise ValueError("Provider is 'google', but GEMINI_KEY not in .env or api_keys.")
            from .providers.gemini import GeminiProvider
            self.model_instance = GeminiProvider(api_key, LLMProvider.prepare_module(module_for_init))
            
        elif provider == 'openai':
            api_key = get_local_secret("OPEN_AI_KEY", raise_error=False)
            if not api_key and api_keys:
                api_key = api_keys.get("OPEN_AI_KEY")

            if not api_key:
                raise ValueError("Provider is 'openai', but OPEN_AI_KEY not in .env or api_keys.")
            from .providers.openai import OpenAIProvider
            self.model_instance = OpenAIProvider(api_key, LLMProvider.prepare_module(module_for_init))
            
        elif provider == 'perplexity':
            api_key = get_local_secret("PERPLEXITY_KEY", raise_error=False)
            if not api_key and api_keys:
                api_key = api_keys.get("PERPLEXITY_KEY")

            if not api_key:
                raise ValueError("Provider is 'perplexity', but PERPLEXITY_KEY not in .env or api_keys.")
            from .providers.perplexity import PerplexityProvider
            self.model_instance = PerplexityProvider(api_key, LLMProvider.prepare_module(module_for_init))
            
        elif provider == 'ollama':
            # Ollama usually doesn't need an API key, but we'll use a placeholder if not provided
            api_key = get_local_secret("OLLAMA_KEY", raise_error=False) or "local-key"
            from .providers.ollama import OllamaProvider
            self.model_instance = OllamaProvider(api_key, LLMProvider.prepare_module(module_for_init))
            
        else:
            raise ValueError(f"Unsupported provider: '{provider}'")

    @staticmethod
    def get_provider_by_model_name(model_name: str) -> str:
        """Determines the model provider based on the model name prefix."""
        model_name_lower = model_name.lower()
        if model_name_lower.startswith("ollama/"):
            return "ollama"
        if model_name_lower.startswith("gpt"):
            return "openai"
        elif model_name_lower.startswith("gemini"):
            return "google"
        elif model_name_lower.startswith(("o1", "o3", "gpt-5")):
            return "openai"
        elif model_name_lower.startswith(("sonar", "perplexity")):
            return "perplexity"
        elif model_name_lower.startswith(("ollama", "mistral", "phi", "qwen")):
            return "ollama"
        elif model_name_lower.startswith("llama"):
            # Default llama to perplexity for backward compatibility, 
            # unless it's explicitly prefixed with ollama elsewhere.
            return "perplexity"
        
        raise ValueError(f"Could not determine provider for model '{model_name}'. "
                         f"Model name should start with 'gpt', 'gemini', 'sonar', or 'ollama'.")

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        if 'model' in kwargs:
            model_name = kwargs['model']
            if model_name.lower().startswith("ollama/"):
                kwargs['model'] = model_name.split("/", 1)[1]
        return self.model_instance.model_response(module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return self.model_instance.upload_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        return self.model_instance.embed_content(input_content, **kwargs)

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        return self.model_instance.evaluate_response(input_prompt, generated_output, rubric)
