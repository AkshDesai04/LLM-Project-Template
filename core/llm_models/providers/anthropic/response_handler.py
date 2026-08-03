"""
Anthropic response generation, tool execution, and extended thinking handlers.
"""

import inspect
import time
from typing import Any, Optional, List
from pydantic import BaseModel

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import build_result, join_reasoning, resolve_return_reasoning
from core.llm_models.providers.anthropic.media_handler import to_content_blocks

logger = get_logger("AnthropicResponseHandler")

DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0
DEFAULT_THINKING_TEMPERATURE: float = 1.0
THINKING_OUTPUT_HEADROOM: int = 1024
STRUCTURED_TOOL_NAME: str = "emit_structured_response"

EFFORT_TOKEN_BUDGETS: dict = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}


def build_structured_tool(structure: Any) -> Optional[dict]:
    if not (inspect.isclass(structure) and issubclass(structure, BaseModel)):
        return None
    return {
        "name": STRUCTURED_TOOL_NAME,
        "description": (structure.__doc__ or "Return the response using this schema.").strip(),
        "input_schema": structure.model_json_schema(),
    }


def extract_usage(usage: Any) -> tuple:
    if not usage:
        return 0, 0, 0
    def get_val(attr): return getattr(usage, attr, 0) or 0
    return get_val('input_tokens'), get_val('output_tokens'), get_val('cache_read_input_tokens')


def generate_anthropic_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
    prompt = getattr(module, 'prompt', "")
    structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
    model = kwargs.get('model', provider.model_name)

    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))
    top_k = kwargs.get('top_k', getattr(module, 'top_k', provider.top_k))
    system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    max_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens)) or DEFAULT_MAX_TOKENS
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', provider.tools)
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)
    reasoning_budget = kwargs.get('reasoning_budget') or kwargs.get('reasoning_effort') or getattr(module, 'reasoning_budget', None)

    stop_sequences = kwargs.get('stop_sequences') or kwargs.get('stop') or getattr(module, 'stop_sequences', provider.stop_sequences)
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]

    files = []
    if uploaded_file:
        if isinstance(uploaded_file, list):
            if any(isinstance(i, list) for i in uploaded_file):
                files = [item for sublist in uploaded_file for item in sublist]
            else:
                files = uploaded_file
        else:
            files = [uploaded_file]

    content_blocks = to_content_blocks(prompt, files)
    call_kwargs = {
        "model": model, "messages": [{"role": "user", "content": content_blocks}],
        "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
    }

    if top_k is not None: call_kwargs["top_k"] = top_k
    if system_prompt: call_kwargs["system"] = system_prompt
    if stop_sequences: call_kwargs["stop_sequences"] = stop_sequences

    structured_tool = build_structured_tool(structure)
    if structured_tool:
        call_kwargs["tools"] = [structured_tool]
        call_kwargs["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
    elif tools:
        call_kwargs["tools"] = tools

    if reasoning_budget:
        if isinstance(reasoning_budget, str):
            budget_tokens = EFFORT_TOKEN_BUDGETS.get(reasoning_budget.lower(), DEFAULT_MAX_TOKENS)
        else:
            budget_tokens = reasoning_budget

        call_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        call_kwargs["temperature"] = DEFAULT_THINKING_TEMPERATURE
        call_kwargs.pop("top_p", None)
        call_kwargs.pop("top_k", None)
        call_kwargs["max_tokens"] = max(max_tokens, budget_tokens + THINKING_OUTPUT_HEADROOM)

    last_exception = None
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)
    logger.info(f"Attempting generation with model: {model}")

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            start_time = time.time()

            if stream:
                response_stream = provider.client.messages.create(**call_kwargs, stream=True)

                def stream_wrapper():
                    input_tokens, output_tokens, cached_tokens = 0, 0, 0
                    try:
                        for event in response_stream:
                            event_type = getattr(event, 'type', '')
                            if event_type == 'message_start':
                                input_tokens, _, cached_tokens = extract_usage(getattr(getattr(event, 'message', None), 'usage', None))
                            elif event_type == 'message_delta':
                                _, delta_output, _ = extract_usage(getattr(event, 'usage', None))
                                output_tokens = delta_output or output_tokens
                            if not return_reasoning:
                                yield event
                                continue
                            delta = getattr(event, 'delta', None)
                            delta_type = getattr(delta, 'type', '')
                            if delta_type == 'thinking_delta':
                                yield ["", getattr(delta, 'thinking', '') or ""]
                            elif delta_type == 'text_delta':
                                yield [getattr(delta, 'text', '') or "", ""]
                    finally:
                        total_duration = time.time() - start_time
                        costs = cost_tracker.calculate_cost(model, input_tokens, output_tokens, cached_tokens)
                        cost_tracker.record_transaction(
                            type(module).__name__, model, costs, total_duration,
                            input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens
                        )
                        logger.info(f"Anthropic Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                return stream_wrapper()

            response = provider.client.messages.create(**call_kwargs)
            total_duration = time.time() - start_time

            text_parts, thinking_parts = [], []
            tool_payload = None

            for block in getattr(response, 'content', []) or []:
                block_type = getattr(block, 'type', '')
                if block_type == 'text':
                    text_parts.append(getattr(block, 'text', ''))
                elif block_type == 'thinking':
                    thinking_parts.append(getattr(block, 'thinking', ''))
                elif block_type == 'redacted_thinking':
                    thinking_parts.append("[redacted thinking block]")
                elif block_type == 'tool_use' and getattr(block, 'name', '') == STRUCTURED_TOOL_NAME:
                    tool_payload = getattr(block, 'input', None)

            reasoning = join_reasoning(thinking_parts)
            output_content = "\n".join(part for part in text_parts if part).strip()

            if not output_content and tool_payload is None:
                raise ValueError("Received an empty response from Anthropic.")

            input_tokens, output_tokens, cached_tokens = extract_usage(getattr(response, 'usage', None))
            if input_tokens or output_tokens:
                costs = cost_tracker.calculate_cost(model, input_tokens, output_tokens, cached_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__, model, costs, total_duration,
                    input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=cached_tokens
                )
                logger.info(f"Anthropic Transaction Recorded: ${costs['total_cost']:.6f} total cost")

            if tool_payload is not None:
                if hasattr(structure, 'model_validate'):
                    return build_result(structure.model_validate(tool_payload), reasoning, return_reasoning)
                return build_result(tool_payload, reasoning, return_reasoning)

            return build_result(output_content, reasoning, return_reasoning)

        except Exception as e:
            last_exception = e
            logger.warning(f"Anthropic response failed on attempt {attempt + 1} for model {model}: {e}")
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
            continue

    raise RuntimeError(f"Failed to get response from Anthropic model {model} after {max_retries} attempts.") from last_exception
