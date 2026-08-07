"""
Perplexity Provider Implementation.
"""

import json
from typing import Optional, List, Any, Union

from utils.logging import get_logger
from utils.env import get_secret
from core.llm_models.base_provider import LLMProvider, JudgeResult
from core.llm_models.providers.perplexity.media_handler import process_perplexity_media
from core.llm_models.providers.perplexity.response_handler import generate_perplexity_response
from core.modules.base import Base as BaseModule

logger = get_logger("PerplexityProvider")
PERPLEXITY_BASE_URL: str = "https://api.perplexity.ai"


class PerplexityProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        api_key = api_key or get_secret("PERPLEXITY_KEY", raise_error=False)
        super().__init__(api_key, base)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=PERPLEXITY_BASE_URL)
        except ImportError:
            self.client = None

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        if self.client is None:
            raise ImportError("openai package required for Perplexity routing. Run `pip install openai`.")
        return generate_perplexity_response(self, module, uploaded_file, **kwargs)

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        return process_perplexity_media(file_bytes, mime_type)

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("Perplexity does not natively provide a standard embedding API.")

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        Return your response in JSON format with fields: score (int), reasoning (str), improvements (str).
        """

        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            response_mime_type: str = "application/json"
            model: str = self.model_name

        raw_output = self.model_response(JudgeModule())
        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            return JudgeResult.model_validate(parsed)
        except Exception:
            logger.warning("Could not parse judge response as JudgeResult; returning raw text as reasoning.")
            return JudgeResult(score=0, reasoning=str(raw_output), improvements="")
