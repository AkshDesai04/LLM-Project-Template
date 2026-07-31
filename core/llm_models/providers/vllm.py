import inspect
import json
import time
from typing import Optional, List, Any, Union

from pydantic import BaseModel

from utils.logger import get_logger
from utils.env_ops import get_secret
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)
from core.modules.base import Base as BaseModule

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = get_logger("VLLMProvider")

DEFAULT_VLLM_URL = "http://localhost:8000/v1"

# vLLM only checks the key when the server was started with --api-key.
PLACEHOLDER_API_KEY = "EMPTY"


class VLLMProvider(LLMProvider):
    """
    Talks to a self-hosted vLLM OpenAI-compatible server. This is a client only;
    the `vllm` package itself is never imported, so no GPU runtime is required.
    """

    def __init__(self, api_key: Optional[str], base: BaseModule):
        if OpenAI is None:
            raise ImportError("OpenAI package required for vLLM routing. Run `pip install openai`")

        api_key = api_key or get_secret("VLLM_KEY", raise_error=False) or PLACEHOLDER_API_KEY
        super().__init__(api_key, base)

        self.base_url = get_secret("VLLM_URL", raise_error=False) or DEFAULT_VLLM_URL
        logger.info(f"Initializing vLLM client with base URL: {self.base_url}")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    @staticmethod
    def _build_response_format(structure: Any) -> Optional[dict]:
        """
        vLLM constrains decoding through the standard response_format field. The
        legacy guided_json extra was removed in vLLM v0.12.0.
        """
        if inspect.isclass(structure) and issubclass(structure, BaseModel):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": structure.__name__,
                    "schema": structure.model_json_schema(),
                },
            }

        if structure:
            return {"type": "json_object"}

        return None

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        model = kwargs.get('model', self.model_name)

        temperature = kwargs.get('temperature', getattr(module, 'temperature', self.temperature))
        top_p = kwargs.get('top_p', getattr(module, 'top_p', self.top_p))
        top_k = kwargs.get('top_k', getattr(module, 'top_k', self.top_k))
        system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens))
        seed = kwargs.get('seed', getattr(module, 'seed', self.seed))
        presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', self.presence_penalty))
        frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', self.frequency_penalty))
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))
        tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', self.tools)

        stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', self.stop_sequences)
        if isinstance(stop, str):
            stop = [stop]

        files = []
        if uploaded_file:
            if isinstance(uploaded_file, list):
                if any(isinstance(i, list) for i in uploaded_file):
                    files = [item for sublist in uploaded_file for item in sublist]
                else:
                    files = uploaded_file
            else:
                files = [uploaded_file]

        image_contents = [f for f in files if isinstance(f, dict) and f.get("type") == "image_url"]
        text_contents = [f for f in files if isinstance(f, str)]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if image_contents:
            content_block = [{"type": "text", "text": prompt}]

            if text_contents:
                content_block[0]["text"] += "\n\n" + "\n\n".join(
                    f"[Attached Content]:\n{t}" for t in text_contents
                )

            content_block.extend(image_contents)
            messages.append({"role": "user", "content": content_block})
        else:
            full_prompt = prompt
            if text_contents:
                full_prompt += "\n\n" + "\n\n".join(
                    f"[Attached Content]:\n{t}" for t in text_contents
                )
            messages.append({"role": "user", "content": full_prompt})

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if stop is not None:
            call_kwargs["stop"] = stop
        if seed is not None:
            call_kwargs["seed"] = seed
        if presence_penalty is not None:
            call_kwargs["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            call_kwargs["frequency_penalty"] = frequency_penalty
        if tools is not None:
            call_kwargs["tools"] = tools

        if stream and structure:
            logger.warning("Streaming is not supported with structured output. Disabling streaming.")
            stream = False

        if stream:
            call_kwargs["stream"] = True
            call_kwargs["stream_options"] = {"include_usage": True}

        response_format = self._build_response_format(structure)
        if response_format:
            call_kwargs["response_format"] = response_format

        # Sampling parameters vLLM accepts outside the OpenAI schema.
        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        repetition_penalty = kwargs.get('repetition_penalty')
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty
        if extra_body:
            call_kwargs["extra_body"] = extra_body

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)

        logger.info(f"Attempting generation with model: {model} (vLLM @ {self.base_url})")
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
                start_time = time.time()

                response = self.client.chat.completions.create(**call_kwargs)

                if stream:
                    def stream_wrapper():
                        for chunk in response:
                            if getattr(chunk, 'usage', None):
                                usage = chunk.usage
                                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                                completion_tokens = getattr(usage, 'completion_tokens', 0)
                                total_duration = time.time() - start_time

                                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                                cost_tracker.record_transaction(
                                    type(module).__name__,
                                    model,
                                    costs,
                                    total_duration,
                                    input_tokens=prompt_tokens,
                                    output_tokens=completion_tokens,
                                    cached_tokens=0,
                                )
                                logger.info(
                                    f"vLLM Stream Transaction Recorded: "
                                    f"{prompt_tokens} in / {completion_tokens} out"
                                )
                            yield chunk
                    return stream_wrapper()

                total_duration = time.time() - start_time
                output_content = response.choices[0].message.content

                usage = getattr(response, 'usage', None)
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)

                    costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                    cost_tracker.record_transaction(
                        type(module).__name__,
                        model,
                        costs,
                        total_duration,
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        cached_tokens=0,
                    )
                    logger.info(
                        f"vLLM Transaction Recorded: "
                        f"{prompt_tokens} in / {completion_tokens} out"
                    )

                if not output_content:
                    raise ValueError("Received an empty response from vLLM.")

                if structure:
                    try:
                        parsed = json.loads(output_content)
                        if hasattr(structure, 'model_validate'):
                            return structure.model_validate(parsed)
                        return parsed
                    except Exception as e:
                        logger.warning(f"Failed to parse structured vLLM response: {e}")
                        if attempt < max_retries - 1:
                            last_exception = e
                            time.sleep(2)
                            continue

                return output_content

            except Exception as e:
                last_exception = e
                logger.warning(f"vLLM response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(2)
                continue

        raise RuntimeError(
            f"Failed to get response from vLLM model {model} after {max_retries} attempts."
        ) from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        try:
            if mime_type == 'application/pdf':
                logger.info("Processing PDF for vLLM (Local Extraction)...")
                return extract_text_from_pdf_bytes(file_bytes)

            elif mime_type.startswith('image/'):
                logger.info(f"Processing {mime_type} for a vLLM vision model...")
                return encode_image_base64(file_bytes, mime_type)

            elif mime_type.startswith('video/'):
                logger.info(f"Processing {mime_type} for vLLM by extracting frames...")
                return process_video_frames(file_bytes)

            else:
                logger.info(f"Treating {mime_type} as plain text...")
                return file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"vLLM media processing failed: {e}")
            raise RuntimeError(f"Failed to process {mime_type} for vLLM: {e}")

    def embed_content(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Requires the server to be running an embedding model."""
        try:
            model = model or self.model_name
            input_data = [text] if isinstance(text, str) else text

            start_time = time.time()
            response = self.client.embeddings.create(model=model, input=input_data, **kwargs)
            total_duration = time.time() - start_time

            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0

            costs = cost_tracker.calculate_cost(model, prompt_tokens, 0)
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
                return response.data[0].embedding
            return [d.embedding for d in response.data]

        except Exception as e:
            logger.error(f"vLLM embedding failed: {e}")
            raise

    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated response.
        [Original Prompt]: {input_prompt}
        [AI Generated Response]: {generated_output}
        [Evaluation Rubric]: {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}
        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """

        # A vLLM server hosts one model, so it also acts as its own judge.
        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = self.model_name

        return self.model_response(JudgeModule())
