"""
OpenAI Chat Completions API handler.
"""

import inspect
from typing import Any, Optional
from pydantic import BaseModel

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import ThinkTagStreamSplitter, join_reasoning

logger = get_logger("OpenAIChatCompletions")

REASONING_MODEL_PATTERNS: tuple = ("o1", "o3", "o4", "gpt-5")
CHAT_COMPLETION_ERROR_PATTERNS: tuple = (
    "not a chat model",
    "not supported in the v1/chat/completions endpoint",
)


def is_reasoning_model(model: str) -> bool:
    return any(x in model.lower() for x in REASONING_MODEL_PATTERNS)


def is_chat_completion_endpoint_error(error: Exception) -> bool:
    return any(p in str(error).lower() for p in CHAT_COMPLETION_ERROR_PATTERNS)


def extract_message_reasoning(message: Any) -> Optional[str]:
    val = getattr(message, 'reasoning_content', None)
    return val.strip() if isinstance(val, str) and val.strip() else None


def execute_chat_completion(client: Any, model: str, messages: list, call_kwargs: dict, structure: Optional[Any], stream: bool, return_reasoning: bool, module: Any, start_time: float):
    if is_reasoning_model(model):
        call_kwargs.pop("temperature", None)
        call_kwargs.pop("top_p", None)
        call_kwargs.pop("presence_penalty", None)
        call_kwargs.pop("frequency_penalty", None)
        if "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")

    if structure and inspect.isclass(structure) and issubclass(structure, BaseModel):
        response = client.beta.chat.completions.parse(**call_kwargs, response_format=structure)
        parsed_object = response.choices[0].message.parsed
        output_content = response.choices[0].message.content
        reasoning = extract_message_reasoning(response.choices[0].message)
        return response, parsed_object, output_content, reasoning, None

    if structure:
        call_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**call_kwargs)

    if stream:
        def stream_wrapper():
            splitter = ThinkTagStreamSplitter()
            last_prompt_tokens, last_completion_tokens, last_cached_tokens = 0, 0, 0
            cost_recorded = False
            try:
                for chunk in response:
                    if getattr(chunk, 'usage', None):
                        u = chunk.usage
                        last_prompt_tokens = getattr(u, 'prompt_tokens', 0)
                        last_completion_tokens = getattr(u, 'completion_tokens', 0)
                        last_cached_tokens = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0)
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
            finally:
                if not cost_recorded:
                    total_duration = time.time() - start_time
                    costs = cost_tracker.calculate_cost(model, last_prompt_tokens, last_completion_tokens, last_cached_tokens)
                    cost_tracker.record_transaction(
                        type(module).__name__, model, costs, total_duration,
                        input_tokens=last_prompt_tokens, output_tokens=last_completion_tokens, cached_tokens=last_cached_tokens
                    )
                    cost_recorded = True
                    logger.info(f"OpenAI Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
        return None, None, None, None, stream_wrapper()

    output_content = response.choices[0].message.content
    reasoning = extract_message_reasoning(response.choices[0].message)
    return response, None, output_content, reasoning, None
