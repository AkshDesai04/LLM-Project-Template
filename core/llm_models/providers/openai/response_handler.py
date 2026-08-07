"""
OpenAI main generation router and workflow handler.
"""

import time
from typing import Any, Optional

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import (
    build_result,
    join_reasoning,
    resolve_return_reasoning,
    split_think_tags,
)
from core.llm_models.providers.openai.responses_api import (
    requires_responses_api,
    responses_api_generate,
)
from core.llm_models.providers.openai.chat_completions import (
    is_chat_completion_endpoint_error,
    execute_chat_completion,
    is_reasoning_model,
)

logger = get_logger("OpenAIResponseHandler")

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0


def generate_openai_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
    prompt = getattr(module, 'prompt', "")
    structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
    model = kwargs.get('model', provider.model_name)

    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))
    reasoning_effort = kwargs.get('reasoning_effort') or kwargs.get('reasoning_budget') or getattr(module, 'reasoning_budget', None)
    system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    seed = kwargs.get('seed', getattr(module, 'seed', provider.seed))
    max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens))
    stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', provider.stop_sequences)
    presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', provider.presence_penalty))
    frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', provider.frequency_penalty))
    logit_bias = kwargs.get('logit_bias')
    tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', provider.tools)
    tool_choice = kwargs.get('tool_choice')
    parallel_tool_calls = kwargs.get('parallel_tool_calls')
    logprobs = kwargs.get('logprobs', getattr(module, 'logprobs', provider.logprobs))
    top_logprobs = kwargs.get('top_logprobs', getattr(module, 'top_logprobs', provider.top_logprobs))
    service_tier = kwargs.get('service_tier', getattr(module, 'service_tier', provider.service_tier))
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)
    modalities = kwargs.get('modalities')
    audio = kwargs.get('audio')

    messages, files = [], []
    if uploaded_file:
        files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        if any(isinstance(i, list) for i in files):
            files = [item for sublist in files for item in sublist]

    image_contents = [f for f in files if isinstance(f, dict) and f.get("type") == "image_url"]
    text_contents = [f for f in files if isinstance(f, str)]

    if image_contents:
        content_block = [{"type": "text", "text": prompt}]
        content_block.extend(image_contents)
        if text_contents:
            content_block[0]["text"] += "\n\n" + "\n\n".join([f"[Attached Content]:\n{t}" for t in text_contents])
        messages.append({"role": "user", "content": content_block})
    else:
        full_prompt = prompt
        if text_contents:
            full_prompt += "\n\n" + "\n\n".join([f"[Attached Content]:\n{t}" for t in text_contents])
        messages.append({"role": "user", "content": full_prompt})

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    last_exception = None
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)
    logger.info(f"Attempting generation with model: {model}")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            start_time = time.time()
            parsed_object, reasoning = None, None

            configured_api_type = cost_tracker.get_model_api_type(model)
            use_responses_api = True if configured_api_type == "responses" else (False if configured_api_type == "chat_completions" else requires_responses_api(model))

            if use_responses_api:
                logger.info(f"Using Responses API for model: {model}")
                response, output_content, reasoning = responses_api_generate(
                    provider.client, model, messages, reasoning_effort, max_tokens, structure, return_reasoning
                )
            else:
                call_kwargs = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
                if seed is not None: call_kwargs["seed"] = seed
                if max_tokens is not None: call_kwargs["max_tokens"] = max_tokens
                if stop is not None: call_kwargs["stop"] = stop
                if presence_penalty is not None: call_kwargs["presence_penalty"] = presence_penalty
                if frequency_penalty is not None: call_kwargs["frequency_penalty"] = frequency_penalty
                if logit_bias is not None: call_kwargs["logit_bias"] = logit_bias
                if tools is not None: call_kwargs["tools"] = tools
                if tool_choice is not None: call_kwargs["tool_choice"] = tool_choice
                if parallel_tool_calls is not None: call_kwargs["parallel_tool_calls"] = parallel_tool_calls
                if logprobs is not None: call_kwargs["logprobs"] = logprobs
                if top_logprobs is not None: call_kwargs["top_logprobs"] = top_logprobs
                if service_tier is not None: call_kwargs["service_tier"] = service_tier
                if modalities is not None: call_kwargs["modalities"] = modalities
                if audio is not None: call_kwargs["audio"] = audio

                if stream and structure:
                    logger.warning("Streaming is not supported with structured output. Disabling streaming.")
                    stream = False
                elif stream:
                    call_kwargs["stream"] = True
                    call_kwargs["stream_options"] = {"include_usage": True}

                if reasoning_effort and isinstance(reasoning_effort, str):
                    call_kwargs["reasoning_effort"] = reasoning_effort.lower()

                try:
                    response, parsed_object, output_content, reasoning, stream_gen = execute_chat_completion(
                        provider.client, model, messages, call_kwargs, structure, stream, return_reasoning, module, start_time
                    )
                    if stream_gen:
                        return stream_gen()
                except Exception as e:
                    if is_chat_completion_endpoint_error(e):
                        logger.warning(f"Model {model} rejected chat completions API. Retrying with Responses API...")
                        response, output_content, reasoning = responses_api_generate(
                            provider.client, model, messages, reasoning_effort, max_tokens, structure, return_reasoning
                        )
                    else:
                        raise

            total_duration = time.time() - start_time
            if not output_content and not parsed_object:
                raise ValueError("Received an empty response from OpenAI.")

            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)
                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__, model, costs, total_duration,
                    input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=cached_tokens
                )
                logger.info(f"OpenAI Transaction Recorded: ${costs['total_cost']:.6f} total cost")

            if parsed_object:
                return build_result(parsed_object, reasoning, return_reasoning)

            output_content, inline_reasoning = split_think_tags(output_content)
            return build_result(output_content, join_reasoning([reasoning, inline_reasoning]), return_reasoning)

        except Exception as e:
            last_exception = e
            logger.warning(f"OpenAI response failed on attempt {attempt + 1} for model {model}: {e}")
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
            continue

    raise RuntimeError(f"Failed to get response from OpenAI model {model} after {max_retries} attempts.") from last_exception
