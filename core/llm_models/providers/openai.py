import inspect
import time
from typing import Optional, List, Any, Union

from pydantic import BaseModel
from openai import OpenAI

from utils.logger import get_logger
from ..base_provider import LLMProvider, JudgeResult
from ..cost_tracker import cost_tracker
from ..utils.media_utils import (
    extract_text_from_pdf_bytes,
    process_video_frames,
    encode_image_base64
)
from core.modules.base import Base as BaseModule

logger = get_logger("OpenAIProvider")


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base: BaseModule):
        super().__init__(api_key, base)
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        model = model.lower()
        return any(x in model for x in ["o1", "o3", "o4", "gpt-5"])

    @staticmethod
    def _requires_responses_api(model: str) -> bool:
        """
        Models that should use the Responses API instead of Chat Completions.
        """
        model = model.lower()

        responses_only_patterns = [
            "gpt-5.5-pro",
            "gpt-5-pro",
        ]

        return any(p in model for p in responses_only_patterns)

    @staticmethod
    def _is_chat_completion_endpoint_error(error: Exception) -> bool:
        error_str = str(error).lower()

        return (
            "not a chat model" in error_str
            or "not supported in the v1/chat/completions endpoint" in error_str
        )

    def _build_responses_input(self, messages: List[dict]) -> List[dict]:
        """
        Converts chat-completion-style messages into Responses API input format.
        """
        response_input = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                response_input.append({
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content
                        }
                    ]
                })

            elif isinstance(content, list):
                transformed_content = []

                for item in content:
                    if item.get("type") == "text":
                        transformed_content.append({
                            "type": "input_text",
                            "text": item.get("text", "")
                        })

                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url")

                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")

                        transformed_content.append({
                            "type": "input_image",
                            "image_url": image_url
                        })

                response_input.append({
                    "role": role,
                    "content": transformed_content
                })

        return response_input

    def _responses_api_generate(
        self,
        model: str,
        messages: List[dict],
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        structure: Optional[Any] = None,
    ):
        """
        Generation using the OpenAI Responses API.
        """

        response_input = self._build_responses_input(messages)

        response_kwargs = {
            "model": model,
            "input": response_input,
        }

        if reasoning_effort and isinstance(reasoning_effort, str):
            response_kwargs["reasoning"] = {
                "effort": reasoning_effort.lower()
            }

        if max_tokens is not None:
            response_kwargs["max_output_tokens"] = max_tokens

        if structure:
            response_kwargs["text"] = {
                "format": {
                    "type": "json_object"
                }
            }

        response = self.client.responses.create(**response_kwargs)

        output_content = getattr(response, "output_text", None)

        if not output_content:
            try:
                output_content = response.output[0].content[0].text
            except Exception:
                output_content = None

        return response, output_content

    def model_response(self, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
        prompt = getattr(module, 'prompt', "")
        structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
        model = kwargs.get('model', self.model_name)

        temperature = kwargs.get(
            'temperature',
            getattr(module, 'temperature', self.temperature)
        )

        top_p = kwargs.get(
            'top_p',
            getattr(module, 'top_p', self.top_p)
        )

        reasoning_effort = (
            kwargs.get('reasoning_effort')
            or kwargs.get('reasoning_budget')
            or getattr(module, 'reasoning_budget', None)
        )

        system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', self.system_prompt))
        seed = kwargs.get('seed', getattr(module, 'seed', self.seed))
        max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', self.max_tokens))
        stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', self.stop_sequences)
        presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', self.presence_penalty))
        frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', self.frequency_penalty))
        logit_bias = kwargs.get('logit_bias')
        tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', self.tools)
        tool_choice = kwargs.get('tool_choice')
        parallel_tool_calls = kwargs.get('parallel_tool_calls')

        logprobs = kwargs.get('logprobs', getattr(module, 'logprobs', self.logprobs))
        top_logprobs = kwargs.get('top_logprobs', getattr(module, 'top_logprobs', self.top_logprobs))
        service_tier = kwargs.get('service_tier', getattr(module, 'service_tier', self.service_tier))
        stream = kwargs.get('stream', getattr(module, 'stream', self.stream))

        modalities = kwargs.get('modalities')
        audio = kwargs.get('audio')

        messages = []
        files = []

        if uploaded_file:
            if isinstance(uploaded_file, list):
                if any(isinstance(i, list) for i in uploaded_file):
                    files = [item for sublist in uploaded_file for item in sublist]
                else:
                    files = uploaded_file
            else:
                files = [uploaded_file]

        image_contents = [
            f for f in files
            if isinstance(f, dict) and f.get("type") == "image_url"
        ]

        text_contents = [
            f for f in files
            if isinstance(f, str)
        ]

        if image_contents:
            content_block = [{"type": "text", "text": prompt}]
            content_block.extend(image_contents)

            if text_contents:
                extra_text = "\n\n" + "\n\n".join(
                    [f"[Attached Content]:\n{t}" for t in text_contents]
                )

                content_block[0]["text"] += extra_text

            messages.append({
                "role": "user",
                "content": content_block
            })

        else:
            full_prompt = prompt

            if text_contents:
                full_prompt += "\n\n" + "\n\n".join(
                    [f"[Attached Content]:\n{t}" for t in text_contents]
                )

            messages.append({
                "role": "user",
                "content": full_prompt
            })

        if system_prompt:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })

        last_exception = None
        max_retries = kwargs.get('max_retries', 3)

        logger.info(f"Attempting generation with model: {model}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")

                start_time = time.time()

                parsed_object = None

                use_responses_api = self._requires_responses_api(model)

                # ============================================================
                # RESPONSES API FLOW
                # ============================================================

                if use_responses_api:
                    logger.info(f"Using Responses API for model: {model}")

                    response, output_content = self._responses_api_generate(
                        model=model,
                        messages=messages,
                        reasoning_effort=reasoning_effort,
                        max_tokens=max_tokens,
                        structure=structure
                    )

                # ============================================================
                # CHAT COMPLETIONS FLOW
                # ============================================================

                else:
                    call_kwargs = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p
                    }

                    if seed is not None:
                        call_kwargs["seed"] = seed

                    if max_tokens is not None:
                        call_kwargs["max_tokens"] = max_tokens

                    if stop is not None:
                        call_kwargs["stop"] = stop

                    if presence_penalty is not None:
                        call_kwargs["presence_penalty"] = presence_penalty

                    if frequency_penalty is not None:
                        call_kwargs["frequency_penalty"] = frequency_penalty

                    if logit_bias is not None:
                        call_kwargs["logit_bias"] = logit_bias

                    if tools is not None:
                        call_kwargs["tools"] = tools

                    if tool_choice is not None:
                        call_kwargs["tool_choice"] = tool_choice

                    if parallel_tool_calls is not None:
                        call_kwargs["parallel_tool_calls"] = parallel_tool_calls

                    if logprobs is not None:
                        call_kwargs["logprobs"] = logprobs

                    if top_logprobs is not None:
                        call_kwargs["top_logprobs"] = top_logprobs

                    if service_tier is not None:
                        call_kwargs["service_tier"] = service_tier

                    if modalities is not None:
                        call_kwargs["modalities"] = modalities

                    if audio is not None:
                        call_kwargs["audio"] = audio

                    if stream:
                        if structure:
                            logger.warning(
                                "Streaming is not supported with structured output. "
                                "Disabling streaming."
                            )
                            stream = False
                        else:
                            call_kwargs["stream"] = True
                            call_kwargs["stream_options"] = {
                                "include_usage": True
                            }

                    is_reasoning_model = self._is_reasoning_model(model)

                    if is_reasoning_model:
                        call_kwargs.pop("temperature", None)
                        call_kwargs.pop("top_p", None)
                        call_kwargs.pop("presence_penalty", None)
                        call_kwargs.pop("frequency_penalty", None)

                        if "max_tokens" in call_kwargs:
                            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")

                        if reasoning_effort and isinstance(reasoning_effort, str):
                            call_kwargs["reasoning_effort"] = reasoning_effort.lower()

                    try:
                        if (
                            structure
                            and inspect.isclass(structure)
                            and issubclass(structure, BaseModel)
                        ):
                            response = self.client.beta.chat.completions.parse(
                                **call_kwargs,
                                response_format=structure
                            )

                            parsed_object = response.choices[0].message.parsed
                            output_content = response.choices[0].message.content

                        else:
                            if structure:
                                call_kwargs["response_format"] = {
                                    "type": "json_object"
                                }

                            response = self.client.chat.completions.create(
                                **call_kwargs
                            )

                            if stream:
                                def stream_wrapper():
                                    for chunk in response:
                                        if getattr(chunk, 'usage', None):
                                            u = chunk.usage

                                            prompt_tokens = getattr(
                                                u,
                                                'prompt_tokens',
                                                0
                                            )

                                            completion_tokens = getattr(
                                                u,
                                                'completion_tokens',
                                                0
                                            )

                                            cached_tokens = getattr(
                                                getattr(
                                                    u,
                                                    'prompt_tokens_details',
                                                    None
                                                ),
                                                'cached_tokens',
                                                0
                                            )

                                            total_duration = time.time() - start_time

                                            costs = cost_tracker.calculate_cost(
                                                model,
                                                prompt_tokens,
                                                completion_tokens,
                                                cached_tokens
                                            )

                                            cost_tracker.record_transaction(
                                                type(module).__name__,
                                                model,
                                                costs,
                                                total_duration
                                            )

                                            logger.info(
                                                f"OpenAI Stream Transaction Recorded: "
                                                f"${costs['total_cost']:.6f} total cost"
                                            )

                                        yield chunk

                                return stream_wrapper()

                            output_content = response.choices[0].message.content

                    except Exception as e:
                        if self._is_chat_completion_endpoint_error(e):
                            logger.warning(
                                f"Model {model} rejected chat completions API. "
                                f"Retrying with Responses API..."
                            )

                            response, output_content = self._responses_api_generate(
                                model=model,
                                messages=messages,
                                reasoning_effort=reasoning_effort,
                                max_tokens=max_tokens,
                                structure=structure
                            )

                        else:
                            raise

                end_time = time.time()
                total_duration = end_time - start_time

                if not output_content and not parsed_object:
                    raise ValueError("Received an empty response from OpenAI.")

                usage = getattr(response, 'usage', None)

                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)

                    completion_tokens = getattr(
                        usage,
                        'completion_tokens',
                        0
                    )

                    cached_tokens = getattr(
                        getattr(
                            usage,
                            'prompt_tokens_details',
                            None
                        ),
                        'cached_tokens',
                        0
                    )

                    costs = cost_tracker.calculate_cost(
                        model,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens
                    )

                    cost_tracker.record_transaction(
                        type(module).__name__,
                        model,
                        costs,
                        total_duration
                    )

                    logger.info(
                        f"OpenAI Transaction Recorded: "
                        f"${costs['total_cost']:.6f} total cost"
                    )

                return parsed_object if parsed_object else output_content

            except Exception as e:
                last_exception = e

                logger.warning(
                    f"OpenAI response failed on attempt "
                    f"{attempt + 1} for model {model}: {e}"
                )

                time.sleep(2)
                continue

        raise RuntimeError(
            f"Failed to get response from OpenAI model "
            f"{model} after {max_retries} attempts."
        ) from last_exception

    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        try:
            if mime_type == 'application/pdf':
                logger.info("Processing PDF for OpenAI (Local Extraction)...")
                return extract_text_from_pdf_bytes(file_bytes)

            elif mime_type.startswith('image/'):
                logger.info(f"Processing {mime_type} for OpenAI Vision...")
                return encode_image_base64(file_bytes, mime_type)

            elif mime_type.startswith('video/'):
                logger.info(f"Processing {mime_type} for OpenAI by extracting frames...")
                return process_video_frames(file_bytes)

            else:
                logger.info(f"Treating {mime_type} as plain text...")
                return file_bytes.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"OpenAI media upload failed: {e}")

            raise RuntimeError(
                f"Failed to process {mime_type} for OpenAI: {e}"
            )

    def embed_content(
        self,
        text: Union[str, List[str]],
        model="text-embedding-3-small",
        **kwargs
    ) -> Union[List[float], List[List[float]]]:

        try:
            input_data = [text] if isinstance(text, str) else text

            start_time = time.time()

            response = self.client.embeddings.create(
                model=model,
                input=input_data,
                **kwargs
            )

            total_duration = time.time() - start_time

            usage = getattr(response, 'usage', None)

            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)

                costs = cost_tracker.calculate_cost(
                    model,
                    prompt_tokens,
                    0,
                    0
                )

                cost_tracker.record_transaction(
                    "Embedding",
                    model,
                    costs,
                    total_duration
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

        [Original Prompt]:
        {input_prompt}

        [AI Generated Response]:
        {generated_output}

        [Evaluation Rubric]:
        {rubric if rubric else "Evaluate based on accuracy, clarity, and adherence to the prompt."}

        Please provide a score from 1-10, your reasoning, and any suggestions for improvement.
        """

        class JudgeModule(BaseModule):
            prompt: str = judge_prompt
            structure: Any = JudgeResult
            model: str = "gpt-4o-mini"

        return self.model_response(JudgeModule())