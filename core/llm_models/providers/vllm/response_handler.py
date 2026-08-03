"""
vLLM response generation and response format handlers.
"""

import inspect
import json
import time
from typing import Any, Optional
from pydantic import BaseModel

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import ThinkTagStreamSplitter, build_result, join_reasoning, resolve_return_reasoning, split_think_tags

logger = get_logger("VLLMResponseHandler")
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0


def build_vllm_response_format(structure: Any) -> Optional[dict]:
    if inspect.isclass(structure) and issubclass(structure, BaseModel):
        return {"type": "json_schema", "json_schema": {"name": structure.__name__, "schema": structure.model_json_schema()}}
    return {"type": "json_object"} if structure else None


def generate_vllm_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
    prompt = getattr(module, 'prompt', "")
    structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
    model = kwargs.get('model', provider.model_name)

    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))
    top_k = kwargs.get('top_k', getattr(module, 'top_k', provider.top_k))
    system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens))
    seed = kwargs.get('seed', getattr(module, 'seed', provider.seed))
    presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', provider.presence_penalty))
    frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', provider.frequency_penalty))
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', provider.tools)
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)

    stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', provider.stop_sequences)
    if isinstance(stop, str): stop = [stop]

    files = []
    if uploaded_file:
        files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        if any(isinstance(i, list) for i in files):
            files = [item for sublist in files for item in sublist]

    image_contents = [f for f in files if isinstance(f, dict) and f.get("type") == "image_url"]
    text_contents = [f for f in files if isinstance(f, str)]

    messages = []
    if system_prompt: messages.append({"role": "system", "content": system_prompt})

    if image_contents:
        content_block = [{"type": "text", "text": prompt}]
        if text_contents:
            content_block[0]["text"] += "\n\n" + "\n\n".join(f"[Attached Content]:\n{t}" for t in text_contents)
        content_block.extend(image_contents)
        messages.append({"role": "user", "content": content_block})
    else:
        full_prompt = prompt
        if text_contents:
            full_prompt += "\n\n" + "\n\n".join(f"[Attached Content]:\n{t}" for t in text_contents)
        messages.append({"role": "user", "content": full_prompt})

    call_kwargs = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
    if max_tokens is not None: call_kwargs["max_tokens"] = max_tokens
    if stop is not None: call_kwargs["stop"] = stop
    if seed is not None: call_kwargs["seed"] = seed
    if presence_penalty is not None: call_kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None: call_kwargs["frequency_penalty"] = frequency_penalty
    if tools is not None: call_kwargs["tools"] = tools

    if stream and structure:
        logger.warning("Streaming is not supported with structured output. Disabling streaming.")
        stream = False
    elif stream:
        call_kwargs["stream"] = True
        call_kwargs["stream_options"] = {"include_usage": True}

    response_format = build_vllm_response_format(structure)
    if response_format: call_kwargs["response_format"] = response_format

    extra_body = {}
    if top_k is not None: extra_body["top_k"] = top_k
    repetition_penalty = kwargs.get('repetition_penalty')
    if repetition_penalty is not None: extra_body["repetition_penalty"] = repetition_penalty
    if extra_body: call_kwargs["extra_body"] = extra_body

    last_exception = None
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)

    logger.info(f"Attempting generation with model: {model} (vLLM @ {provider.base_url})")
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            start_time = time.time()
            response = provider.client.chat.completions.create(**call_kwargs)

            if stream:
                def stream_wrapper():
                    splitter = ThinkTagStreamSplitter()
                    for chunk in response:
                        if getattr(chunk, 'usage', None):
                            u = chunk.usage
                            prompt_tokens = getattr(u, 'prompt_tokens', 0)
                            completion_tokens = getattr(u, 'completion_tokens', 0)
                            total_duration = time.time() - start_time
                            costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                            cost_tracker.record_transaction(
                                type(module).__name__, model, costs, total_duration,
                                input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0
                            )
                            logger.info(f"vLLM Stream Transaction Recorded: {prompt_tokens} in / {completion_tokens} out")

                        if not return_reasoning:
                            yield chunk
                            continue

                        delta = getattr((chunk.choices or [None])[0], 'delta', None) if getattr(chunk, 'choices', None) else None
                        text, thought = splitter.feed(getattr(delta, 'content', None))
                        thought = join_reasoning([getattr(delta, 'reasoning_content', None), thought]) or ""
                        if text or thought:
                            yield [text, thought]

                    if return_reasoning:
                        text, thought = splitter.flush()
                        if text or thought:
                            yield [text, thought]
                return stream_wrapper()

            total_duration = time.time() - start_time
            message = response.choices[0].message
            output_content, field_reasoning = message.content, getattr(message, 'reasoning_content', None)

            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens, completion_tokens = getattr(usage, 'prompt_tokens', 0), getattr(usage, 'completion_tokens', 0)
                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__, model, costs, total_duration,
                    input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0
                )
                logger.info(f"vLLM Transaction Recorded: {prompt_tokens} in / {completion_tokens} out")

            if not output_content:
                raise ValueError("Received an empty response from vLLM.")

            output_content, inline_reasoning = split_think_tags(output_content)
            reasoning = join_reasoning([field_reasoning, inline_reasoning])

            if structure:
                try:
                    parsed = json.loads(output_content)
                    if hasattr(structure, 'model_validate'): parsed = structure.model_validate(parsed)
                    return build_result(parsed, reasoning, return_reasoning)
                except Exception as e:
                    logger.warning(f"Failed to parse structured vLLM response: {e}")
                    if attempt < max_retries - 1:
                        last_exception = e
                        time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                        continue

            return build_result(output_content, reasoning, return_reasoning)

        except Exception as e:
            last_exception = e
            logger.warning(f"vLLM response failed on attempt {attempt + 1} for model {model}: {e}")
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
            continue

    raise RuntimeError(f"Failed to get response from vLLM model {model} after {max_retries} attempts.") from last_exception
