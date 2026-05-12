import inspect
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.file_ops import read_file
from core.modules.base import Base as BaseModule

logger = get_logger("LLMProvider")

class JudgeResult(BaseModel):
    score: int = Field(..., description="A score from 1 to 10 evaluating the response quality.")
    reasoning: str = Field(..., description="Detailed explanation for the assigned score.")
    improvements: Optional[str] = Field(None, description="Suggestions for improving the response.")

class LLMProvider(ABC):
    def __init__(self, api_key: str, base_config: BaseModule):
        self.api_key = api_key
        self.model_name = base_config.model
        self.temperature = base_config.temperature
        self.top_p = base_config.top_p
        self.top_k = base_config.top_k
        self.fall_back_models = getattr(base_config, 'fall_back_models', None)
        self.structure = getattr(base_config, 'structure', None)
        self.stream = getattr(base_config, 'stream', False)
        self.service_tier = getattr(base_config, 'service_tier', None)
        self.logprobs = getattr(base_config, 'logprobs', False)
        self.top_logprobs = getattr(base_config, 'top_logprobs', None)
        
        self.max_tokens = getattr(base_config, 'max_tokens', None)
        self.system_prompt = getattr(base_config, 'system_prompt', None)
        self.presence_penalty = getattr(base_config, 'presence_penalty', 0.0)
        self.frequency_penalty = getattr(base_config, 'frequency_penalty', 0.0)
        self.stop_sequences = getattr(base_config, 'stop_sequences', None)
        self.seed = getattr(base_config, 'seed', None)
        
        self.candidate_count = getattr(base_config, 'candidate_count', 1)
        self.safety_settings = getattr(base_config, 'safety_settings', None)
        self.tools = getattr(base_config, 'tools', None)
        self.return_citations = getattr(base_config, 'return_citations', True)
        self.search_domain_filter = getattr(base_config, 'search_domain_filter', None)
        self.search_recency_filter = getattr(base_config, 'search_recency_filter', None)
        
        self.response_mime_type = getattr(base_config, 'response_mime_type', "application/json")

    @staticmethod
    def prepare_module(module: Any) -> BaseModule:
        """Shared logic to instantiate a module and load its prompt if necessary."""
        logger.info(f"Preparing module: {module}")
        if inspect.isclass(module):
            module_instance = module()
        elif isinstance(module, BaseModule):
            module_instance = module
        else:
            logger.error(f"Invalid module type: {type(module)}")
            raise TypeError("Module must be subclass or instance of core.modules.base.Base")

        if not getattr(module_instance, "prompt", None):
            prompt_path = getattr(module_instance, "prompt_path", None)
            if prompt_path:
                logger.info(f"Loading prompt from: {prompt_path}")
                module_instance.prompt = read_file(os.path.abspath(prompt_path))

        logger.info(f"Module {type(module_instance).__name__} prepared successfully.")
        return module_instance

    @abstractmethod
    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        pass

    @abstractmethod
    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        pass

    @abstractmethod
    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        pass

    @abstractmethod
    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        """Evaluates a model's response using the LLM-as-a-Judge pattern."""
        pass
