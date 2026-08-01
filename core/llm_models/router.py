import time
from typing import Any, Optional, List, Union

from utils.logger import get_logger
from core.modules.base import Base as BaseModule
from .base_provider import LLMProvider, JudgeResult
from .cost_tracker import cost_tracker
from .model_names import PROVIDER_ALIASES, split_model_name

logger = get_logger("ModelRouter")


class ModelRouter:
    def __init__(self, module: BaseModule, fallback_index: int = 0):
        model = getattr(module, 'model', None)
        models = getattr(module, 'models', None)

        chain = list(models) if models is not None else []
        if model:
            chain.insert(0, model)

        if not chain:
            raise ValueError("Module must specify at least 'model' or 'models'.")

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
        Removes the provider prefix so the SDK and the cost tracker both see the
        real model id, e.g. 'gemini/gemini-2.5-flash' -> 'gemini-2.5-flash'.
        """
        return split_model_name(model_name)[1]

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
        """Resolves the provider from the 'provider/model' prefix."""
        provider, _ = split_model_name(model_name)
        if provider:
            return provider

        return ModelRouter._infer_provider_from_bare_name(model_name)

    @staticmethod
    def _infer_provider_from_bare_name(model_name: str) -> str:
        """
        Guesses the provider for a name written without a prefix.

        Kept so existing module configs keep working, but it cannot recognise a
        model family it has never heard of, which is exactly what the explicit
        prefix solves.
        """
        model_name_lower = model_name.lower()

        if model_name_lower.startswith(("gpt", "o1", "o3", "o4")):
            provider = "openai"
        elif model_name_lower.startswith("gemini"):
            provider = "google"
        elif model_name_lower.startswith(("claude", "anthropic")):
            provider = "anthropic"
        elif model_name_lower.startswith(("sonar", "perplexity")):
            provider = "perplexity"
        elif model_name_lower.startswith(("mistral", "phi", "qwen")):
            provider = "ollama"
        elif model_name_lower.startswith("llama"):
            # Ambiguous between Perplexity and Ollama; kept on Perplexity for
            # backward compatibility.
            provider = "perplexity"
        else:
            raise ValueError(
                f"Could not determine provider for model '{model_name}'. "
                f"Prefix the model with its provider, e.g. "
                f"'gemini/{model_name}'. Valid prefixes: "
                f"{', '.join(sorted(PROVIDER_ALIASES))}."
            )

        logger.warning(
            f"Model '{model_name}' has no provider prefix; inferred '{provider}'. "
            f"Prefer the explicit form '{provider}/{model_name}'."
        )
        return provider

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
                # Record the stripped name so a failed row matches the id a
                # successful row on the same model would be recorded under.
                cost_tracker.record_failed_attempt(
                    module_name, self.strip_routing_prefix(model_name), duration, error=e
                )
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
