"""
OpenAI Provider Implementation.
"""

import time
from typing import Optional, List, Any, Union
from openai import OpenAI

from utils.logging import get_logger
from utils.env import get_secret
from core.llm_models.base_provider import LLMProvider, JudgeResult
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.providers.openai.media_handler import process_openai_media
from core.llm_models.providers.openai.response_handler import generate_openai_response, is_reasoning_model
from core.modules.base import Base as BaseModule

logger = get_logger("OpenAIProvider")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_secret("OPEN_AI_KEY")
        super().__init__(api_key, base)
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        return is_reasoning_model(model)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        return generate_openai_response(self, module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return process_openai_media(file_bytes, mime_type)

    def embed_content(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        try:
            model = model or self.model_name
            input_data = [text] if isinstance(text, str) else text
            start_time = time.time()

            response = self.client.embeddings.create(model=model, input=input_data, **kwargs)
            total_duration = time.time() - start_time

            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                costs = cost_tracker.calculate_cost(model, prompt_tokens, 0, 0)
                cost_tracker.record_transaction(
                    "Embedding", model, costs, total_duration,
                    input_tokens=prompt_tokens, output_tokens=0, cached_tokens=0,
                )

            if isinstance(text, str):
                return response.data[0].embedding
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def evaluate_response(
        self,
        input_prompt: str,
        generated_output: str,
        rubric: Optional[str] = None
    ) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """

        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = self.model_name

        return self.model_response(JudgeModule())
