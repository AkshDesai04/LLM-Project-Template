import time
from typing import Any, Optional, List, Union

from utils.logger import get_logger
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from core.modules.base import Base as BaseModule

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = get_logger("PerplexityProvider")

class PerplexityProvider(LLMProvider):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        if OpenAI:
            self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        else:
            raise ImportError("OpenAI package required for Perplexity routing. Run `pip install openai`")

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        model = kwargs.get('model', self.model_name)
        temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
        top_p = kwargs.get('top_p', getattr(module, 'top_p', self.top_p))
        
        system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens))
        presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', self.presence_penalty))
        frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', self.frequency_penalty))
        search_domain_filter = kwargs.get('search_domain_filter', getattr(module, 'search_domain_filter', self.search_domain_filter))
        return_citations = kwargs.get('return_citations', getattr(module, 'return_citations', self.return_citations))
        search_recency_filter = kwargs.get('search_recency_filter', getattr(module, 'search_recency_filter', self.search_recency_filter))
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        full_prompt = prompt
        if uploaded_file:
            if isinstance(uploaded_file, list):
                for f in uploaded_file:
                    if isinstance(f, str): full_prompt += f"\n\n[Attached]: {f}"
            elif isinstance(uploaded_file, str):
                full_prompt += f"\n\n[Attached]: {uploaded_file}"

        messages.append({"role": "user", "content": full_prompt})

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }

        if max_tokens is not None: call_kwargs["max_tokens"] = max_tokens
        if presence_penalty is not None: call_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None: call_kwargs["frequency_penalty"] = frequency_penalty

        if search_domain_filter is not None: call_kwargs["search_domain_filter"] = search_domain_filter
        if return_citations is not None: call_kwargs["return_citations"] = return_citations
        if search_recency_filter is not None: call_kwargs["search_recency_filter"] = search_recency_filter
        
        if stream:
            call_kwargs["stream"] = True
            call_kwargs["stream_options"] = {"include_usage": True}

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)

        logger.info(f"Attempting generation with model: {model}")
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
                start_time = time.time()

                response = self.client.chat.completions.create(**call_kwargs)

                if stream:
                    def stream_wrapper():
                        for chunk in response:
                            if getattr(chunk, 'usage', None):
                                u = chunk.usage
                                prompt_tokens = getattr(u, 'prompt_tokens', 0)
                                completion_tokens = getattr(u, 'completion_tokens', 0)
                                total_duration = time.time() - start_time
                                
                                try:
                                    costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                                except ValueError:
                                    costs = {"input_cost": 0.0, "output_cost": 0.0, "cached_cost": 0.0, "total_cost": 0.0}

                                cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)
                                logger.info(f"Perplexity Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                            yield chunk
                    return stream_wrapper()

                total_duration = time.time() - start_time
                output_content = response.choices[0].message.content
                
                usage = getattr(response, 'usage', None)
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    
                    try:
                        costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                    except ValueError:
                        costs = {"input_cost": 0.0, "output_cost": 0.0, "cached_cost": 0.0, "total_cost": 0.0}

                    cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)

                return output_content

            except Exception as e:
                last_exception = e
                logger.warning(f"Perplexity response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(2)
                continue
                
        raise RuntimeError(f"Failed to get response from Perplexity after {max_retries} attempts.") from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        logger.info(f"Perplexity: processing {mime_type} media.")
        if mime_type == 'text/plain':
            return file_bytes.decode('utf-8', errors='ignore')
        return f"[Media of type {mime_type} attached]"

    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("Perplexity does not natively provide a standard embedding API.")

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """
        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            model: str = "sonar-pro"

        raw_output = self.model_response(JudgeModule())
        return JudgeResult(score=5, reasoning=raw_output, improvements="")
