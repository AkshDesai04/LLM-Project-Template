import time
from typing import Any, Optional, List, Union

from utils.logging import get_logger
from core.modules.base import Base as BaseModule
from core.llm_models.base_provider import LLMProvider, JudgeResult
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.model_names import PROVIDER_ALIASES, split_model_name

logger = get_logger("ModelRouter")

PROVIDER_GOOGLE = "google"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_PERPLEXITY = "perplexity"
PROVIDER_OLLAMA = "ollama"
PROVIDER_VLLM = "vllm"
DEFAULT_FALLBACK_INDEX = 0


class ModelRouter:
    def __init__(self, module: BaseModule, fallback_index: int = DEFAULT_FALLBACK_INDEX):
        model = getattr(module, 'model', None)
        models = getattr(module, 'models', None)

        chain = list(models) if models is not None else []
        if model:
            chain.insert(0, model)

        if not chain:
            raise ValueError("Module must specify at least 'model' or 'models'.")

        if fallback_index < 0 or fallback_index >= len(chain):
            raise ValueError(f"Fallback index {fallback_index} is out of range for chain of length {len(chain)}.")

        self._model_chain = chain[fallback_index:]
        self._original_module = module
        self.model_instance: Optional[LLMProvider] = None

        try:
            self.model_instance = self._build_provider(self._model_chain[0], module)
        except Exception as e:
            logger.warning(f"Could not initialize primary model '{self._model_chain[0]}' during router setup: {e}")

    @staticmethod
    def strip_routing_prefix(model_name: str) -> str:
        return split_model_name(model_name)[1]

    def _build_provider(self, model_name: str, module: BaseModule) -> LLMProvider:
        provider = self.get_provider_by_model_name(model_name)
        stripped_model_name = self.strip_routing_prefix(model_name)
        module_for_init = module.model_copy(update={'model': stripped_model_name})

        logger.info(f"Routing to provider: '{provider}' for model '{stripped_model_name}'")

        try:
            if provider == PROVIDER_GOOGLE:
                from core.llm_models.providers.gemini import GeminiProvider
                return GeminiProvider(None, LLMProvider.prepare_module(module_for_init))
            if provider == PROVIDER_OPENAI:
                from core.llm_models.providers.openai import OpenAIProvider
                return OpenAIProvider(None, LLMProvider.prepare_module(module_for_init))
            if provider == PROVIDER_ANTHROPIC:
                from core.llm_models.providers.anthropic import AnthropicProvider
                return AnthropicProvider(None, LLMProvider.prepare_module(module_for_init))
            if provider == PROVIDER_PERPLEXITY:
                from core.llm_models.providers.perplexity import PerplexityProvider
                return PerplexityProvider(None, LLMProvider.prepare_module(module_for_init))
            if provider == PROVIDER_OLLAMA:
                from core.llm_models.providers.ollama import OllamaProvider
                return OllamaProvider(None, LLMProvider.prepare_module(module_for_init))
            if provider == PROVIDER_VLLM:
                from core.llm_models.providers.vllm import VLLMProvider
                return VLLMProvider(None, LLMProvider.prepare_module(module_for_init))

            raise ValueError(f"Unsupported provider: '{provider}' for model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to build provider '{provider}' for model '{model_name}': {e}")
            raise

    @staticmethod
    def get_provider_by_model_name(model_name: str) -> str:
        provider, _ = split_model_name(model_name)
        if provider:
            return provider
        return ModelRouter._infer_provider_from_bare_name(model_name)

    @staticmethod
    def _infer_provider_from_bare_name(model_name: str) -> str:
        name = model_name.lower()
        if name.startswith(("gpt", "o1", "o3", "o4")):
            provider = PROVIDER_OPENAI
        elif name.startswith("gemini"):
            provider = PROVIDER_GOOGLE
        elif name.startswith(("claude", "anthropic")):
            provider = PROVIDER_ANTHROPIC
        elif name.startswith(("sonar", "perplexity")):
            provider = PROVIDER_PERPLEXITY
        elif name.startswith(("mistral", "phi", "qwen", "deepseek", "codestral", "command")):
            provider = PROVIDER_OLLAMA
        elif name.startswith("llama"):
            provider = PROVIDER_PERPLEXITY
        else:
            raise ValueError(f"Could not determine provider for model '{model_name}'. Prefix with provider e.g. 'gemini/{model_name}'.")

        logger.warning(f"Model '{model_name}' has no provider prefix; inferred '{provider}'.")
        return provider

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
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
                provider = self._build_provider(model_name, self._original_module)
                self.model_instance = provider
                call_kwargs = {**kwargs, 'model': self.strip_routing_prefix(model_name)}
                return provider.model_response(module, uploaded_file, **call_kwargs)
            except Exception as e:
                duration = time.time() - start_time
                last_exception = e
                cost_tracker.record_failed_attempt(module_name, self.strip_routing_prefix(model_name), duration, error=e)
                logger.warning(f"Model '{model_name}' failed after {duration:.2f}s; trying next fallback. Error: {e}")
                continue

        raise RuntimeError(f"Failed to get response after trying all models in chain {self._model_chain}. Last error: {last_exception}") from last_exception

    def _require_model_instance(self) -> LLMProvider:
        if self.model_instance is None:
            try:
                self.model_instance = self._build_provider(self._model_chain[0], self._original_module)
            except Exception as e:
                raise RuntimeError(f"Could not initialize primary model provider for '{self._model_chain[0]}': {e}") from e
        return self.model_instance

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return self._require_model_instance().upload_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        return self._require_model_instance().embed_content(input_content, **kwargs)

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        return self._require_model_instance().evaluate_response(input_prompt, generated_output, rubric)
