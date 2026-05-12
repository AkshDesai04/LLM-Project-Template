import time
from io import BytesIO
from typing import Optional, List, Any, Union

from google import genai
from google.genai import types
from google.genai.types import ThinkingLevel

from utils.logger import get_logger
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from core.modules.base import Base as BaseModule

logger = get_logger("GeminiProvider")

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        self.client = genai.Client(api_key=api_key)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        base_model = kwargs.get('model', self.model_name)
        
        top_p = kwargs.get('top_p', getattr(module, 'top_p', self.top_p))
        top_k = kwargs.get('top_k', getattr(module, 'top_k', self.top_k))
        temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
        reasoning_budget = kwargs.get('reasoning_budget') or kwargs.get('reasoning_level') or getattr(module, 'reasoning_budget', None)
        response_mime_type = kwargs.get('response_mime_type', getattr(module, 'response_mime_type', "application/json"))
        
        system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        candidate_count = kwargs.get('candidate_count', getattr(module, 'candidate_count', self.candidate_count))
        max_output_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens))
        stop_sequences = kwargs.get('stop_sequences') or kwargs.get('stop') or getattr(module, 'stop_sequences', self.stop_sequences)
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', self.presence_penalty))
        frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', self.frequency_penalty))
        seed = kwargs.get('seed', getattr(module, 'seed', self.seed))
        tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', self.tools)
        safety_settings = kwargs.get('safety_settings', getattr(module, 'safety_settings', self.safety_settings))
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))

        contents: List[Any] = [prompt]
        if uploaded_file:
            if isinstance(uploaded_file, list):
                contents.extend(uploaded_file)
            else:
                contents.append(uploaded_file)

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)
        fallbacks = self.fall_back_models or []
        models_to_try = [base_model] + fallbacks

        for model in models_to_try:
            try:
                cost_tracker.calculate_cost(model, 0, 0)
            except ValueError as e:
                logger.warning(f"Skipping model {model} due to pricing check: {e}")
                last_exception = e
                continue

            logger.info(f"Attempting generation with model: {model}")
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")

                    thinking_config_obj = None
                    if reasoning_budget:
                        if isinstance(reasoning_budget, str):
                            reasoning = ThinkingLevel(reasoning_budget)
                            thinking_config_obj = types.ThinkingConfig(include_thoughts=True, thinking_level=reasoning)
                        else:
                            thinking_config_obj = types.ThinkingConfig(include_thoughts=True)

                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        response_mime_type=response_mime_type,
                        response_schema=structure,
                        thinking_config=thinking_config_obj,
                        system_instruction=system_prompt,
                        candidate_count=candidate_count,
                        max_output_tokens=max_output_tokens,
                        stop_sequences=stop_sequences,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        seed=seed,
                        tools=tools,
                        safety_settings=safety_settings
                    )

                    start_time = time.time()
                    if stream:
                        response_stream = self.client.models.generate_content_stream(model=model, contents=contents, config=config)
                        
                        def stream_wrapper():
                            for chunk in response_stream:
                                if chunk.usage_metadata:
                                    u = chunk.usage_metadata
                                    def get_val(obj, attr): return getattr(obj, attr, 0) or 0
                                    
                                    prompt_tokens = get_val(u, 'prompt_token_count')
                                    candidate_tokens = get_val(u, 'candidates_token_count')
                                    cached_tokens = get_val(u, 'cached_content_token_count')

                                    total_duration = time.time() - start_time
                                    costs = cost_tracker.calculate_cost(model, prompt_tokens, candidate_tokens, cached_tokens)
                                    cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)
                                    logger.info(f"Gemini Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                                yield chunk
                        return stream_wrapper()

                    response = self.client.models.generate_content(model=model, contents=contents, config=config)
                    total_duration = time.time() - start_time

                    if not response.text and not getattr(response, 'parsed', None):
                        raise ValueError("Received an empty response from Gemini.")

                    if response.usage_metadata:
                        u = response.usage_metadata
                        def get_val(obj, attr): return getattr(obj, attr, 0) or 0
                        
                        prompt_tokens = get_val(u, 'prompt_token_count')
                        candidate_tokens = get_val(u, 'candidates_token_count')
                        cached_tokens = get_val(u, 'cached_content_token_count')

                        costs = cost_tracker.calculate_cost(model, prompt_tokens, candidate_tokens, cached_tokens)
                        cost_tracker.record_transaction(type(module).__name__, model, costs, total_duration)

                        logger.info(f"Gemini Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                    if structure:
                        return response.parsed
                    return response.text

                except Exception as e:
                    last_exception = e
                    logger.warning(f"Gemini response failed on attempt {attempt + 1} for model {model}: {e}")
                    time.sleep(2)
                    continue
            break 

        raise RuntimeError("Failed to get response from Gemini after trying all specified models.") from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> types.File:
        try:
            logger.info(f"Uploading {mime_type} to Gemini...")
            file_obj = BytesIO(file_bytes)
            file_obj.seek(0)

            uploaded_file = self.client.files.upload(
                file=file_obj,
                config=types.UploadFileConfig(mime_type=mime_type)
            )

            while uploaded_file.state.name == "PROCESSING":
                logger.info(f"File {uploaded_file.name} is still processing...")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise RuntimeError(f"File {uploaded_file.name} failed to process.")

            return uploaded_file
        except Exception as e:
            logger.error(f"Gemini upload failed: {e}")
            raise RuntimeError(f"Failed to upload {mime_type} to Gemini: {e}")

    def embed_content(self, text: Union[str, List[str]], task_type: str = "RETRIEVAL_DOCUMENT", model="gemini-embedding-001", dimensions=1536, **kwargs) -> Union[List[float], List[List[float]]]:
        try:
            input_texts = [text] if isinstance(text, str) else text
            start_time = time.time()
            result = self.client.models.embed_content(
                model=model,
                contents=input_texts,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dimensions)
            )
            total_duration = time.time() - start_time

            prompt_tokens = 0
            usage = getattr(result, 'usage_metadata', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            else:
                try:
                    token_count_resp = self.client.models.count_tokens(model=model, contents=input_texts)
                    prompt_tokens = token_count_resp.total_tokens
                except Exception:
                    pass

            if prompt_tokens > 0:
                costs = cost_tracker.calculate_cost(model, prompt_tokens, 0, 0)
                cost_tracker.record_transaction("Embedding", model, costs, total_duration)

            if isinstance(text, str):
                return result.embeddings[0].values
            return [e.values for e in result.embeddings]
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

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
            structure: Any = JudgeResult
            model: str = "gemini-2.0-flash"

        return self.model_response(JudgeModule())
