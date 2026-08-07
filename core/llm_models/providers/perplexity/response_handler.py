"""
Perplexity response generation and citation handling.
"""

import time
from typing import Any, Optional

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import (
    ThinkTagStreamSplitter,
    build_result,
    resolve_return_reasoning,
    split_think_tags,
)

logger = get_logger("PerplexityResponseHandler")
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0


def generate_perplexity_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
    prompt = getattr(module, 'prompt', "")
    model = kwargs.get('model', provider.model_name)
    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))

    system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens))
    presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', provider.presence_penalty))
    frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', provider.frequency_penalty))
    search_domain_filter = kwargs.get('search_domain_filter', getattr(module, 'search_domain_filter', provider.search_domain_filter))
    return_citations = kwargs.get('return_citations', getattr(module, 'return_citations', provider.return_citations))
    search_recency_filter = kwargs.get('search_recency_filter', getattr(module, 'search_recency_filter', provider.search_recency_filter))
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)

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

    call_kwargs = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p}
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
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)

    logger.info(f"Attempting generation with model: {model}")
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
                                input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0,
                            )
                            logger.info(f"Perplexity Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")

                        if not return_reasoning:
                            yield chunk
                            continue

                        delta = getattr((chunk.choices or [None])[0], 'delta', None) if getattr(chunk, 'choices', None) else None
                        text, thought = splitter.feed(getattr(delta, 'content', None))
                        if text or thought:
                            yield [text, thought]

                    if return_reasoning:
                        text, thought = splitter.flush()
                        if text or thought:
                            yield [text, thought]
                return stream_wrapper()

            total_duration = time.time() - start_time
            output_content = response.choices[0].message.content

            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                costs = cost_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__, model, costs, total_duration,
                    input_tokens=prompt_tokens, output_tokens=completion_tokens, cached_tokens=0,
                )

            output_content, reasoning = split_think_tags(output_content)
            return build_result(output_content, reasoning, return_reasoning)

        except Exception as e:
            last_exception = e
            logger.warning(f"Perplexity response failed on attempt {attempt + 1} for model {model}: {e}")
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
            continue

    raise RuntimeError(f"Failed to get response from Perplexity after {max_retries} attempts.") from last_exception
