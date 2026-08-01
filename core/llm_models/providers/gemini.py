import time
from io import BytesIO
from typing import Optional, List, Any, Union

from google import genai
from google.genai import types
from google.genai.types import ThinkingLevel

from utils.logger import get_logger
from utils.env_ops import (
    get_secret,
    get_gemini_key_type,
    load_gemini_service_account_credentials,
    resolve_gemini_project,
    resolve_gemini_location,
    GEMINI_KEY_TYPE_API_KEY,
    GEMINI_KEY_TYPE_SERVICE_ACC_JSON,
    GEMINI_API_KEY_NAME,
)
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..reasoning import build_result, join_reasoning, resolve_return_reasoning
from core.modules.base import Base as BaseModule

logger = get_logger("GeminiProvider")

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str], base: BaseModule):
        key_type = get_gemini_key_type()
        self.key_type = key_type
        self.uses_vertex = key_type == GEMINI_KEY_TYPE_SERVICE_ACC_JSON

        # LLMProvider stores self.api_key; for Vertex we keep a sentinel rather
        # than the service-account JSON itself.
        if key_type == GEMINI_KEY_TYPE_API_KEY:
            resolved_key = api_key or get_secret(GEMINI_API_KEY_NAME)
        else:
            resolved_key = api_key or "vertex-service-account"

        super().__init__(resolved_key, base)
        self.client = self._build_client(api_key)

    def _build_client(self, api_key: Optional[str]):
        """
        Builds a google-genai Client for the credential type set by
        GEMINI_KEY_TYPE in .env.

        GEMINI_KEY       -> Gemini Developer API (client.files.upload available)
        SERVICE_ACC_JSON -> Vertex AI (inline Parts; Files API unavailable)
        """
        if self.key_type == GEMINI_KEY_TYPE_API_KEY:
            key = api_key or get_secret(GEMINI_API_KEY_NAME)
            logger.info("Initializing Gemini client with API key auth.")
            return genai.Client(api_key=key)

        credentials = load_gemini_service_account_credentials()
        project = resolve_gemini_project(credentials)
        location = resolve_gemini_location()
        logger.info(
            f"Initializing Gemini client with Vertex AI service-account auth "
            f"(project={project}, location={location})."
        )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    @staticmethod
    def _split_thought_parts(candidates: Any) -> tuple:
        """
        Gemini returns thoughts as ordinary Parts flagged with thought=True,
        mixed in with the answer. Returns (reasoning, answer_text).
        """
        thoughts = []
        answers = []

        for candidate in candidates or []:
            content = getattr(candidate, 'content', None)
            for part in getattr(content, 'parts', None) or []:
                text = getattr(part, 'text', None)
                if not text:
                    continue
                if getattr(part, 'thought', False):
                    thoughts.append(text)
                else:
                    answers.append(text)

        return join_reasoning(thoughts), "".join(answers)

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        model = kwargs.get('model', self.model_name)
        
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
        return_reasoning = resolve_return_reasoning(module, kwargs, self.return_reasoning)

        contents: List[Any] = [prompt]
        if uploaded_file:
            if isinstance(uploaded_file, list):
                contents.extend(uploaded_file)
            else:
                contents.append(uploaded_file)

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)

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
                elif return_reasoning:
                    # Thought Parts are only returned when they are asked for, so
                    # the flag alone has to switch them on.
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
                                cost_tracker.record_transaction(
                                    type(module).__name__,
                                    model,
                                    costs,
                                    total_duration,
                                    input_tokens=prompt_tokens,
                                    output_tokens=candidate_tokens,
                                    cached_tokens=cached_tokens,
                                )
                                logger.info(f"Gemini Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                            if return_reasoning:
                                chunk_reasoning, chunk_text = self._split_thought_parts(
                                    getattr(chunk, 'candidates', None)
                                )
                                yield [chunk_text, chunk_reasoning]
                            else:
                                yield chunk
                    return stream_wrapper()

                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                total_duration = time.time() - start_time

                if response.usage_metadata:
                    u = response.usage_metadata
                    def get_val(obj, attr): return getattr(obj, attr, 0) or 0
                    
                    prompt_tokens = get_val(u, 'prompt_token_count')
                    candidate_tokens = get_val(u, 'candidates_token_count')
                    cached_tokens = get_val(u, 'cached_content_token_count')

                    costs = cost_tracker.calculate_cost(model, prompt_tokens, candidate_tokens, cached_tokens)
                    cost_tracker.record_transaction(
                        type(module).__name__,
                        model,
                        costs,
                        total_duration,
                        input_tokens=prompt_tokens,
                        output_tokens=candidate_tokens,
                        cached_tokens=cached_tokens,
                    )

                    logger.info(f"Gemini Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                # Split thought parts BEFORE checking emptiness, because
                # response.text can raise ValueError on multi-part responses
                # that include thought Parts alongside answer Parts.
                reasoning, answer_text = self._split_thought_parts(
                    getattr(response, 'candidates', None)
                )

                if not answer_text and not getattr(response, 'parsed', None):
                    raise ValueError("Received an empty response from Gemini.")

                if structure:
                    return build_result(response.parsed, reasoning, return_reasoning)

                # response.text spans every Part, so prefer the thought-free
                # text when thoughts were included.
                return build_result(
                    answer_text or response.text, reasoning, return_reasoning
                )

            except Exception as e:
                last_exception = e
                logger.warning(f"Gemini response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(2)
                continue

        raise RuntimeError(
            f"Failed to get response from Gemini model {model} after {max_retries} attempts."
        ) from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        try:
            # Vertex AI rejects the Developer Files API. Inline the bytes as a
            # Part so callers keep the same upload_media -> model_response flow.
            if self.uses_vertex:
                logger.info(
                    f"Inlining {mime_type} as a Part for Vertex AI "
                    f"(Files API is unavailable under service-account auth)."
                )
                return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

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

    def embed_content(self, text: Union[str, List[str]], task_type: str = "RETRIEVAL_DOCUMENT", model: Optional[str] = None, dimensions=1536, **kwargs) -> Union[List[float], List[List[float]]]:
        try:
            model = model or self.model_name
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
                cost_tracker.record_transaction(
                    "Embedding",
                    model,
                    costs,
                    total_duration,
                    input_tokens=prompt_tokens,
                    output_tokens=0,
                    cached_tokens=0,
                )

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
            model: str = self.model_name

        return self.model_response(JudgeModule())
