"""
Unit tests for chain of thought reasoning utilities (core.llm_models.reasoning).
"""

from unittest.mock import MagicMock
import pytest

from core.llm_models.reasoning import (
    ThinkTagStreamSplitter,
    build_result,
    join_reasoning,
    resolve_return_reasoning,
    split_think_tags,
)


@pytest.mark.llm
@pytest.mark.unit
def test_resolve_return_reasoning():
    # 1. Kwargs priority
    module = MagicMock(return_reasoning=False)
    assert resolve_return_reasoning(module, {"return_reasoning": True}) is True

    # 2. Module fallback
    assert resolve_return_reasoning(module, {}) is False

    # 3. Default fallback
    module_empty = object()
    assert resolve_return_reasoning(module_empty, {}, fallback=True) is True


@pytest.mark.llm
@pytest.mark.unit
def test_join_reasoning():
    assert join_reasoning([" part 1 ", "", None, "part 2\n"]) == "part 1\npart 2"
    assert join_reasoning([]) is None
    assert join_reasoning(["   ", None]) is None


@pytest.mark.llm
@pytest.mark.unit
def test_split_think_tags():
    # No tags
    text, reasoning = split_think_tags("Just answer")
    assert text == "Just answer"
    assert reasoning is None

    # Single tag
    raw = "<think>Internal thought process</think>Final answer"
    text, reasoning = split_think_tags(raw)
    assert text == "Final answer"
    assert reasoning == "Internal thought process"

    # Multiple tags
    raw_multi = "<think>T1</think>Middle<reasoning>T2</reasoning>End"
    text_m, reasoning_m = split_think_tags(raw_multi)
    assert text_m == "MiddleEnd"
    assert reasoning_m == "T1\nT2"


@pytest.mark.llm
@pytest.mark.unit
def test_build_result():
    assert build_result("Response", reasoning="Thought", return_reasoning=True) == ["Response", "Thought"]
    assert build_result("Response", reasoning="Thought", return_reasoning=False) == "Response"


@pytest.mark.llm
@pytest.mark.unit
def test_think_tag_stream_splitter():
    splitter = ThinkTagStreamSplitter()

    # Feed chunks with opening and closing tags split across calls
    c1, r1 = splitter.feed("Hello <thi")
    c2, r2 = splitter.feed("nk>My reasoning</thin")
    c3, r3 = splitter.feed("king> World!")

    c_tail, r_tail = splitter.flush()

    full_content = c1 + c2 + c3 + c_tail
    full_reasoning = r1 + r2 + r3 + r_tail

    assert "Hello" in full_content
    assert "World!" in full_content
    assert "My reasoning" in full_reasoning
