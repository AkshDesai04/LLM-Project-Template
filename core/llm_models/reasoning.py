"""
Shared handling for the return_reasoning flag on core.modules.base.Base.

Every SDK exposes a model's chain of thought differently: Gemini flags thought
Parts, Anthropic emits thinking blocks, OpenAI's Responses API returns reasoning
summaries, vLLM populates reasoning_content, and several open models simply wrap
their thoughts in <think> tags inside the normal content.

Providers collect whatever their SDK offers and pass it through build_result, so
a caller sees the same [response, reasoning] shape no matter which model
answered. reasoning is None when the model did not produce one.
"""

import re
from typing import Any, Iterable, Optional, Tuple

# deepseek-r1, the Qwen reasoning models and Perplexity's sonar-reasoning inline
# their thoughts in the content rather than exposing a separate field.
THINK_TAG_PATTERN = re.compile(
    r"<(think|thinking|reasoning)>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)

_OPEN_TAG_PATTERN = re.compile(r"<(?:think|thinking|reasoning)>", re.IGNORECASE)
_CLOSE_TAG_PATTERN = re.compile(r"</(?:think|thinking|reasoning)>", re.IGNORECASE)

# The longest tag we recognise. A stream is held back by this much so a tag
# split across two chunks is never emitted as content.
_MAX_TAG_LEN = len("</reasoning>")


def resolve_return_reasoning(module: Any, kwargs: dict, fallback: bool = False) -> bool:
    """Resolves the flag with the same kwargs > module > provider order as
    every other setting in model_response."""
    return bool(
        kwargs.get(
            'return_reasoning',
            getattr(module, 'return_reasoning', fallback),
        )
    )


def join_reasoning(parts: Iterable[Optional[str]]) -> Optional[str]:
    """Joins reasoning fragments, returning None rather than an empty string so
    callers can distinguish "no reasoning" from "empty reasoning"."""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n".join(cleaned) if cleaned else None


def split_think_tags(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Pulls <think> blocks out of a completed response.

    Returns (content_without_tags, reasoning). When the text holds no tags it is
    returned untouched with None, so this is safe to call on any response.
    """
    if not text:
        return text, None

    thoughts = [match.group(2) for match in THINK_TAG_PATTERN.finditer(text)]
    if not thoughts:
        return text, None

    return THINK_TAG_PATTERN.sub("", text).strip(), join_reasoning(thoughts)


def build_result(response: Any, reasoning: Optional[str], return_reasoning: bool) -> Any:
    """
    Returns [response, reasoning] when the caller asked for the chain of thought
    and the bare response otherwise.

    response keeps whatever type it already had, so a structured call still
    yields its parsed model at index 0.
    """
    return [response, reasoning] if return_reasoning else response


class ThinkTagStreamSplitter:
    """
    Incremental split_think_tags, for providers whose reasoning arrives inline.

    A chunk boundary can fall in the middle of a tag, so text is buffered until
    it is long enough that it cannot be the start of one. Call flush() once the
    stream ends to release the remainder.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thought = False

    def feed(self, text: Optional[str]) -> Tuple[str, str]:
        """Returns the (content, reasoning) deltas available from this chunk."""
        if not text:
            return "", ""

        self._buffer += text
        content_parts = []
        reasoning_parts = []

        while True:
            pattern = _CLOSE_TAG_PATTERN if self._in_thought else _OPEN_TAG_PATTERN
            match = pattern.search(self._buffer)
            if not match:
                break

            target = reasoning_parts if self._in_thought else content_parts
            target.append(self._buffer[:match.start()])
            self._buffer = self._buffer[match.end():]
            self._in_thought = not self._in_thought

        # Release everything that can no longer be the opening of a tag.
        releasable = len(self._buffer) - _MAX_TAG_LEN
        if releasable > 0:
            target = reasoning_parts if self._in_thought else content_parts
            target.append(self._buffer[:releasable])
            self._buffer = self._buffer[releasable:]

        return "".join(content_parts), "".join(reasoning_parts)

    def flush(self) -> Tuple[str, str]:
        """Releases the held-back tail once no more chunks are coming."""
        remainder, self._buffer = self._buffer, ""
        return ("", remainder) if self._in_thought else (remainder, "")
