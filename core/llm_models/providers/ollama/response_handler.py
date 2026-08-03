"""
Ollama response generation and thought handling logic.
"""

import time
import json
import base64
from typing import Any, Optional

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import (
    ThinkTagStreamSplitter,
    build_result,
    join_reasoning,
    resolve_return_reasoning,
    split_think_tags,
)

logger = get_logger("OllamaResponseHandler")

DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0
DEFAULT_FORMAT_JSON: str = "json"


def generate_ollama_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, system_prompt: Optional[str] = None, **kwargs) -> Any:
    prompt = getattr(module, 'prompt', "")
    structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
    model = kwargs.get('model', provider.model_name)

    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))
    top_k = kwargs.get('top_k', getattr(module, 'top_k', provider.top_k))

    final_system_prompt = system_prompt or kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens))
    stop = kwargs.get('stop') or kwargs.get('stop_sequences') or getattr(module, 'stop_sequences', provider.stop_sequences)
    seed = kwargs.get('seed', getattr(module, 'seed', provider.seed))
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)

    options = {
        'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
        'num_predict': max_tokens, 'stop': stop, 'seed': seed,
    }
    options = {k: v for k, v in options.items() if v is not None}

    messages = []
    if final_system_prompt:
        messages.append({'role': 'system', 'content': final_system_prompt})

    images = []
    if uploaded_file:
        files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        for f in files:
            if isinstance(f, bytes):
                images.append(f)
            elif isinstance(f, dict) and f.get('type') == 'image_url':
                url = f.get('image_url', {}).get('url', '')
                if url.startswith('data:image'):
                    try:
                        images.append(base64.b64decode(url.split(',')[1]))
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
            elif isinstance(f, str):
                prompt += f"\n\n[Attached Content]:\n{f}"

    user_msg = {'role': 'user', 'content': prompt}
    if images:
        user_msg['images'] = images
    messages.append(user_msg)

    last_exception = None
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)
    pull_attempted = False

    format_param = None
    response_mime_type = kwargs.get('response_mime_type', getattr(module, 'response_mime_type', provider.response_mime_type))
    if structure or response_mime_type == "application/json":
        format_param = DEFAULT_FORMAT_JSON

    if stream and structure:
        logger.warning("Streaming is not supported with structured output in Ollama. Disabling streaming.")
        stream = False

    logger.info(f"Attempting generation with model: {model} (Ollama)")
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            start_time = time.time()

            if stream:
                response_stream = provider.client.chat(
                    model=model, messages=messages, options=options, format=format_param, stream=True
                )

                def stream_wrapper():
                    splitter = ThinkTagStreamSplitter()
                    for chunk in response_stream:
                        if chunk.get('done'):
                            prompt_tokens = chunk.get('prompt_eval_count', 0)
                            completion_tokens = chunk.get('eval_count', 0)
                            total_duration = time.time() - start_time
                            costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                            cost_tracker.record_transaction(
                                type(module).__name__, model, costs, total_duration,
                                input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0
                            )
                            logger.info(f"Ollama Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                        if not return_reasoning:
                            yield chunk
                            continue

                        msg = chunk.get('message') or {}
                        text, thought = splitter.feed(msg.get('content'))
                        thought = join_reasoning([msg.get('thinking'), thought]) or ""
                        if text or thought:
                            yield [text, thought]

                    if return_reasoning:
                        text, thought = splitter.flush()
                        if text or thought:
                            yield [text, thought]
                return stream_wrapper()

            response = provider.client.chat(model=model, messages=messages, options=options, format=format_param)
            total_duration = time.time() - start_time
            msg = response['message']
            output_content = msg['content']
            field_reasoning = msg.get('thinking')

            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)

            costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
            cost_tracker.record_transaction(
                type(module).__name__, model, costs, total_duration,
                input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0
            )
            logger.info(f"Ollama Transaction Recorded: ${costs['total_cost']:.6f} total cost")

            output_content, inline_reasoning = split_think_tags(output_content)
            reasoning = join_reasoning([field_reasoning, inline_reasoning])

            if structure:
                try:
                    parsed = json.loads(output_content)
                    if hasattr(structure, 'model_validate'):
                        parsed = structure.model_validate(parsed)
                    return build_result(parsed, reasoning, return_reasoning)
                except Exception as e:
                    logger.warning(f"Failed to parse Ollama JSON response: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                        continue

            return build_result(output_content, reasoning, return_reasoning)

        except Exception as e:
            last_exception = e
            err_str = str(e).lower()
            if getattr(e, 'status_code', None) == 404 and 'not found' in err_str and not pull_attempted:
                logger.info(f"Model '{model}' not found locally. Attempting to pull from Ollama Hub...")
                pull_attempted = True
                try:
                    logger.info(f"Pulling '{model}'. This may take a while...")
                    provider.client.pull(model)
                    logger.info(f"Successfully pulled model '{model}'. Retrying generation...")
                    continue
                except Exception as pull_error:
                    logger.error(f"Failed to pull model '{model}' from Ollama Hub: {pull_error}")
                    raise RuntimeError(
                        f"Model '{model}' is not installed locally and could not be found or pulled from Ollama Hub. "
                        f"Ensure the model name is correct. Error: {pull_error}"
                    ) from pull_error
            else:
                logger.warning(f"Ollama response failed on attempt {attempt + 1} for model {model}: {e}")
                time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
                continue

    raise RuntimeError(f"Failed to get response from Ollama after {max_retries} attempts.") from last_exception
