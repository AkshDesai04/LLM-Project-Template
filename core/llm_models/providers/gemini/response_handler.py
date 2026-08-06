"""
Gemini response generation and thought handling logic.
"""

import time
from typing import Any, Optional, List
from google.genai import types
from google.genai.types import ThinkingLevel

from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.llm_models.reasoning import build_result, join_reasoning, resolve_return_reasoning

logger = get_logger("GeminiResponseHandler")
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_SLEEP_SECONDS: float = 2.0


def split_thought_parts(candidates: Any) -> tuple:
    """Returns (reasoning, answer_text) from candidate thought parts."""
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


def generate_gemini_response(provider: Any, module: Any, uploaded_file: Optional[Any] = None, **kwargs) -> Any:
    """Executes Gemini content generation loop with retries, streaming, and cost tracking."""
    prompt = getattr(module, 'prompt', "")
    structure = kwargs.get('schema') or kwargs.get('structure') or getattr(module, 'structure', None)
    model = kwargs.get('model', provider.model_name)

    top_p = kwargs.get('top_p', getattr(module, 'top_p', provider.top_p))
    top_k = kwargs.get('top_k', getattr(module, 'top_k', provider.top_k))
    temperature = kwargs.get('temperature', getattr(module, 'temperature', provider.temperature))
    reasoning_budget = kwargs.get('reasoning_budget') or kwargs.get('reasoning_level') or getattr(module, 'reasoning_budget', None)
    response_mime_type = kwargs.get('response_mime_type', getattr(module, 'response_mime_type', "application/json"))

    system_prompt = kwargs.get('system_prompt', getattr(module, 'system_prompt', provider.system_prompt))
    candidate_count = kwargs.get('candidate_count', getattr(module, 'candidate_count', provider.candidate_count))
    max_output_tokens = kwargs.get('max_tokens', getattr(module, 'max_tokens', provider.max_tokens))
    stop_sequences = kwargs.get('stop_sequences') or kwargs.get('stop') or getattr(module, 'stop_sequences', provider.stop_sequences)
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]
    presence_penalty = kwargs.get('presence_penalty', getattr(module, 'presence_penalty', provider.presence_penalty))
    frequency_penalty = kwargs.get('frequency_penalty', getattr(module, 'frequency_penalty', provider.frequency_penalty))
    seed = kwargs.get('seed', getattr(module, 'seed', provider.seed))
    tools = kwargs.get('tools') or kwargs.get('function') or getattr(module, 'tools', provider.tools)
    safety_settings = kwargs.get('safety_settings', getattr(module, 'safety_settings', provider.safety_settings))
    stream = kwargs.get('stream', getattr(module, 'stream', provider.stream))
    return_reasoning = resolve_return_reasoning(module, kwargs, provider.return_reasoning)

    contents: List[Any] = [prompt]
    if uploaded_file:
        if isinstance(uploaded_file, list):
            contents.extend(uploaded_file)
        else:
            contents.append(uploaded_file)

    last_exception = None
    max_retries = kwargs.get('max_retries', DEFAULT_MAX_RETRIES)

    logger.info(f"Attempting generation with model: {model}")
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")

            thinking_config_obj = None
            if reasoning_budget:
                if isinstance(reasoning_budget, str):
                    reasoning = ThinkingLevel(reasoning_budget.upper())
                    thinking_config_obj = types.ThinkingConfig(include_thoughts=True, thinking_level=reasoning)
                else:
                    thinking_config_obj = types.ThinkingConfig(include_thoughts=True)
            elif return_reasoning:
                thinking_config_obj = types.ThinkingConfig(include_thoughts=True)

            config = types.GenerateContentConfig(
                temperature=temperature, top_p=top_p, top_k=top_k,
                response_mime_type=response_mime_type, response_schema=structure,
                thinking_config=thinking_config_obj, system_instruction=system_prompt,
                candidate_count=candidate_count, max_output_tokens=max_output_tokens,
                stop_sequences=stop_sequences, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty, seed=seed, tools=tools,
                safety_settings=safety_settings
            )

            start_time = time.time()
            if stream:
                response_stream = provider.client.models.generate_content_stream(model=model, contents=contents, config=config)

                def stream_wrapper():
                    last_prompt_tokens = 0
                    last_candidate_tokens = 0
                    last_cached_tokens = 0
                    cost_recorded = False
                    try:
                        for chunk in response_stream:
                            if chunk.usage_metadata:
                                u = chunk.usage_metadata
                                last_prompt_tokens = getattr(u, 'prompt_token_count', 0) or 0
                                last_candidate_tokens = getattr(u, 'candidates_token_count', 0) or 0
                                last_cached_tokens = getattr(u, 'cached_content_token_count', 0) or 0

                            if return_reasoning:
                                chunk_reasoning, chunk_text = split_thought_parts(getattr(chunk, 'candidates', None))
                                yield [chunk_text, chunk_reasoning]
                            else:
                                yield chunk
                    finally:
                        if not cost_recorded:
                            total_duration = time.time() - start_time
                            costs = cost_tracker.calculate_cost(model, last_prompt_tokens, last_candidate_tokens, last_cached_tokens)
                            cost_tracker.record_transaction(
                                type(module).__name__, model, costs, total_duration,
                                input_tokens=last_prompt_tokens, output_tokens=last_candidate_tokens, cached_tokens=last_cached_tokens
                            )
                            cost_recorded = True
                            logger.info(f"Gemini Stream Transaction Recorded: ${costs['total_cost']:.6f} total cost")
                return stream_wrapper()

            response = provider.client.models.generate_content(model=model, contents=contents, config=config)
            total_duration = time.time() - start_time

            if response.usage_metadata:
                u = response.usage_metadata
                prompt_tokens = getattr(u, 'prompt_token_count', 0) or 0
                candidate_tokens = getattr(u, 'candidates_token_count', 0) or 0
                cached_tokens = getattr(u, 'cached_content_token_count', 0) or 0

                costs = cost_tracker.calculate_cost(model, prompt_tokens, candidate_tokens, cached_tokens)
                cost_tracker.record_transaction(
                    type(module).__name__, model, costs, total_duration,
                    input_tokens=prompt_tokens, output_tokens=candidate_tokens, cached_tokens=cached_tokens
                )
                logger.info(f"Gemini Transaction Recorded: ${costs['total_cost']:.6f} total cost")

            reasoning, answer_text = split_thought_parts(getattr(response, 'candidates', None))

            if not answer_text and not getattr(response, 'parsed', None):
                raise ValueError("Received an empty response from Gemini.")

            if structure:
                return build_result(response.parsed, reasoning, return_reasoning)

            return build_result(answer_text or response.text, reasoning, return_reasoning)

        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            if ("thinking" in err_msg) and (reasoning_budget or return_reasoning):
                logger.warning(f"Model '{model}' issue with thinking config ({e}). Disabling thinking config and retrying.")
                reasoning_budget = None
                return_reasoning = False
                continue
            if ("invalid_argument" in err_msg or "invalid argument" in err_msg) and structure is None and response_mime_type == "application/json":
                logger.warning(f"Model '{model}' invalid argument with application/json without schema ({e}). Retrying with text/plain.")
                response_mime_type = "text/plain"
                continue
            logger.warning(f"Gemini response failed on attempt {attempt + 1} for model {model}: {e}")
            time.sleep(DEFAULT_RETRY_SLEEP_SECONDS)
            continue

    raise RuntimeError(f"Failed to get response from Gemini model {model} after {max_retries} attempts.") from last_exception
