import time
from typing import Any, Optional, List, Union

from utils.logger import get_logger
from core.modules.base import Base as BaseModule
from .base_provider import LLMProvider, JudgeResult
from .cost_tracker import cost_tracker

logger = get_logger("ModelRouter")

# Prefixes that only select a provider and are stripped before the call is made.
ROUTING_PREFIXES = ("ollama/", "vllm/")

class ModelRouter:
    def __init__(self, module: BaseModule, fallback_index: int = 0):
        primary = module.model
        fallbacks = getattr(module, 'fallback_models', None) or []

        # Build an order-preserving, de-duplicated chain starting at fallback_index
        raw_chain = [primary] + list(fallbacks)
        seen = set()
        chain = []
        for name in raw_chain:
            if name and name not in seen:
                seen.add(name)
                chain.append(name)

        if fallback_index < 0 or fallback_index >= len(chain):
            raise ValueError(f"Fallback index {fallback_index} is out of range.")

        self._model_chain = chain[fallback_index:]
        self._original_module = module

        # Best-effort primary provider for proxy methods; model_response rebuilds per chain entry
        self.model_instance: Optional[LLMProvider] = None
        try:
            self.model_instance = self._build_provider(self._model_chain[0], module)
        except Exception as e:
            logger.warning(
                f"Could not initialize primary model '{self._model_chain[0]}' "
                f"during router setup (will retry in model_response / fallbacks): {e}"
            )

    @staticmethod
    def strip_routing_prefix(model_name: str) -> str:
        """
        Removes a routing-only prefix, e.g. 'ollama/llama3.2:1b' -> 'llama3.2:1b'
        and 'vllm/meta-llama/Llama-3.1-8B' -> 'meta-llama/Llama-3.1-8B'.
        """
        if model_name.lower().startswith(ROUTING_PREFIXES):
            return model_name.split("/", 1)[1]
        return model_name

    def _build_provider(self, model_name: str, module: BaseModule) -> LLMProvider:
        """Constructs the correct LLMProvider for a given model name."""
        provider = self.get_provider_by_model_name(model_name)

        model_name = self.strip_routing_prefix(model_name)
        module_for_init = module.model_copy(update={'model': model_name})

        logger.info(f"Routing to provider: {provider} for model '{model_name}'")

        if provider == 'google':
            from .providers.gemini import GeminiProvider
            return GeminiProvider(None, LLMProvider.prepare_module(module_for_init))

        if provider == 'openai':
            from .providers.openai import OpenAIProvider
            return OpenAIProvider(None, LLMProvider.prepare_module(module_for_init))

        if provider == 'anthropic':
            from .providers.anthropic import AnthropicProvider
            return AnthropicProvider(None, LLMProvider.prepare_module(module_for_init))

        if provider == 'perplexity':
            from .providers.perplexity import PerplexityProvider
            return PerplexityProvider(None, LLMProvider.prepare_module(module_for_init))

        if provider == 'ollama':
            from .providers.ollama import OllamaProvider
            return OllamaProvider(None, LLMProvider.prepare_module(module_for_init))

        if provider == 'vllm':
            from .providers.vllm import VLLMProvider
            return VLLMProvider(None, LLMProvider.prepare_module(module_for_init))

        raise ValueError(f"Unsupported provider: '{provider}'")

    @staticmethod
    def get_provider_by_model_name(model_name: str) -> str:
        """Determines the model provider based on the model name prefix."""
        model_name_lower = model_name.lower()
        if model_name_lower.startswith("vllm/"):
            return "vllm"
        if model_name_lower.startswith("ollama/"):
            return "ollama"
        if model_name_lower.startswith("gpt"):
            return "openai"
        elif model_name_lower.startswith("gemini"):
            return "google"
        elif model_name_lower.startswith(("claude", "anthropic")):
            return "anthropic"
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
                         f"Model name should start with 'gpt', 'gemini', 'claude', 'sonar', "
                         f"'ollama/', or 'vllm/'.")

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        # If the caller explicitly overrides the model, skip the fallback chain
        if 'model' in kwargs:
            kwargs['model'] = self.strip_routing_prefix(kwargs['model'])
            if self.model_instance is None:
                self.model_instance = self._build_provider(kwargs['model'], self._original_module)
            return self.model_instance.model_response(module, uploaded_file, **kwargs)

        last_exception = None
        module_name = type(module).__name__ if module is not None else type(self._original_module).__name__

        for model_name in self._model_chain:
            start_time = time.time()
            try:
                # Build a fresh provider for this model so cross-provider fallback works
                provider = self._build_provider(model_name, self._original_module)
                self.model_instance = provider

                call_kwargs = {**kwargs, 'model': self.strip_routing_prefix(model_name)}

                return provider.model_response(module, uploaded_file, **call_kwargs)

            except Exception as e:
                duration = time.time() - start_time
                last_exception = e
                cost_tracker.record_failed_attempt(module_name, model_name, duration, error=e)
                logger.warning(
                    f"Model '{model_name}' failed after {duration:.2f}s; "
                    f"trying next fallback if available. Error: {e}"
                )
                continue

        raise RuntimeError(
            f"Failed to get response after trying all models in chain: {self._model_chain}"
        ) from last_exception

    def _require_model_instance(self) -> LLMProvider:
        if self.model_instance is None:
            self.model_instance = self._build_provider(self._model_chain[0], self._original_module)
        return self.model_instance

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return self._require_model_instance().upload_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        return self._require_model_instance().embed_content(input_content, **kwargs)

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        return self._require_model_instance().evaluate_response(input_prompt, generated_output, rubric)
