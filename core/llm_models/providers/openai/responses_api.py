"""
OpenAI Responses API handler.
"""

import inspect
from typing import Any, Optional, List
from pydantic import BaseModel
from core.llm_models.reasoning import join_reasoning

RESPONSES_ONLY_PATTERNS: tuple = ("gpt-5.5-pro", "gpt-5-pro")


def requires_responses_api(model: str) -> bool:
    return any(p in model.lower() for p in RESPONSES_ONLY_PATTERNS)


def build_responses_input(messages: List[dict]) -> List[dict]:
    response_input = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            response_input.append({"role": role, "content": [{"type": "input_text", "text": content}]})
        elif isinstance(content, list):
            transformed = []
            for item in content:
                if item.get("type") == "text":
                    transformed.append({"type": "input_text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    img_url = item.get("image_url")
                    if isinstance(img_url, dict):
                        img_url = img_url.get("url")
                    transformed.append({"type": "input_image", "image_url": img_url})
            response_input.append({"role": role, "content": transformed})
    return response_input


def extract_responses_reasoning(response: Any) -> Optional[str]:
    summaries = []
    for item in getattr(response, 'output', None) or []:
        if getattr(item, 'type', '') != 'reasoning':
            continue
        for part in getattr(item, 'summary', None) or []:
            summaries.append(getattr(part, 'text', '') or "")
    return join_reasoning(summaries)


def responses_api_generate(client: Any, model: str, messages: List[dict], reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None, structure: Optional[Any] = None, return_reasoning: bool = False):
    response_input = build_responses_input(messages)
    response_kwargs = {"model": model, "input": response_input}

    reasoning_kwargs = {}
    if reasoning_effort and isinstance(reasoning_effort, str):
        reasoning_kwargs["effort"] = reasoning_effort.lower()
    if return_reasoning:
        reasoning_kwargs["summary"] = "auto"
    if reasoning_kwargs:
        response_kwargs["reasoning"] = reasoning_kwargs

    if max_tokens is not None:
        response_kwargs["max_output_tokens"] = max_tokens

    if structure:
        if inspect.isclass(structure) and issubclass(structure, BaseModel):
            response_kwargs["text"] = {"format": {"type": "json_schema", "name": structure.__name__, "schema": structure.model_json_schema()}}
        else:
            response_kwargs["text"] = {"format": {"type": "json_object"}}

    response = client.responses.create(**response_kwargs)
    output_content = getattr(response, "output_text", None)
    if not output_content:
        try:
            output_content = response.output[0].content[0].text
        except Exception:
            output_content = None

    return response, output_content, extract_responses_reasoning(response)
