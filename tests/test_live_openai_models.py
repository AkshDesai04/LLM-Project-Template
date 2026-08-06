"""
Live integration tests for OpenAI LLM models (gpt-4o, gpt-4o-mini, gpt-5.4, o3-mini, o4-mini, gpt-5-mini) and Tool Calling.
Included gpt-5.4 ($15/M output). Excluded high-cost pro models (> $40/M tokens).
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import pytest

from core.llm_models.router import ModelRouter
from core.modules.base import Base as BaseModule


class CapitalInfo(BaseModel):
    country: str = Field(..., description="Name of the country")
    capital: str = Field(..., description="Capital city of the country")
    population_millions: Optional[float] = Field(None, description="Population in millions")


LONG_CONTEXT = "The quick brown fox jumps over the lazy dog. " * 200

# Global tracking list for tool executions
CALLED_TOOLS: List[str] = []


def calculate_multiply(a: float, b: float) -> float:
    """Multiply two floating-point numbers safely."""
    msg = f"[TOOL CALLED] calculate_multiply(a={a}, b={b})"
    print(msg)
    CALLED_TOOLS.append("calculate_multiply")
    try:
        return float(a) * float(b)
    except Exception:
        return 0.0


def safe_weather_lookup(location: str) -> str:
    """Get current weather status for a location safely without erroring."""
    msg = f"[TOOL CALLED] safe_weather_lookup(location='{location}')"
    print(msg)
    CALLED_TOOLS.append("safe_weather_lookup")
    try:
        clean_loc = str(location).strip().title() if location else "Unknown"
        return f"Sunny and 22°C in {clean_loc}"
    except Exception as err:
        return f"Weather unavailable: {str(err)}"


def run_model_test(
    model: str,
    prompt: str,
    reasoning_budget: Optional[str] = None,
    return_reasoning: bool = False,
    structure: Optional[type] = None,
    stream: bool = False,
    tools: Optional[list] = None,
) -> dict:
    """Helper runner for OpenAI model responses using ModelRouter."""
    module = BaseModule(
        prompt=prompt,
        model=model,
        system_prompt="You are a helpful AI assistant.",
        reasoning_budget=reasoning_budget,
        return_reasoning=return_reasoning,
        structure=structure,
        stream=stream,
        tools=tools,
        response_mime_type="text/plain" if tools else "application/json",
    )

    router = ModelRouter(module)
    res = router.model_response(module)

    if stream:
        chunks = list(res)
        assert len(chunks) > 0
        return {"result": chunks, "stream": True}

    if return_reasoning and isinstance(res, list) and len(res) == 2:
        return {"response": res[0], "reasoning": res[1]}

    return {"response": res}


@pytest.mark.live
@pytest.mark.openai
def test_openai_standard_models():
    res_mini = run_model_test("openai/gpt-4o-mini", "What is 2+2?")
    assert res_mini["response"] is not None

    res_struct = run_model_test(
        "openai/gpt-4o-mini",
        f"Provide capital details for France. Context: {LONG_CONTEXT}",
        structure=CapitalInfo,
    )
    assert res_struct["response"] is not None

    res_4o = run_model_test("openai/gpt-4o", "Name a primary color.")
    assert res_4o["response"] is not None

    res_5_mini = run_model_test("openai/gpt-5-mini", "Say hello in French.")
    assert res_5_mini["response"] is not None

    res_5_4 = run_model_test("openai/gpt-5.4", "Explain the general theory of relativity briefly.")
    assert res_5_4["response"] is not None


@pytest.mark.live
@pytest.mark.openai
def test_openai_reasoning_models():
    res_o3 = run_model_test(
        "openai/o3-mini", "Explain quantum superposition briefly.", reasoning_budget="low", return_reasoning=True
    )
    assert res_o3["response"] is not None

    res_o4 = run_model_test("openai/o4-mini", "Solve: 15 * 14")
    assert res_o4["response"] is not None


@pytest.mark.live
@pytest.mark.openai
def test_openai_streaming_and_judge():
    res_stream = run_model_test("openai/gpt-4o-mini", "Count 1 to 5.", stream=True)
    assert res_stream["stream"] is True

    module = BaseModule(model="openai/gpt-5.4", prompt="Test prompt")
    router = ModelRouter(module)
    eval_result = router.evaluate_response("What is AI?", "AI is artificial intelligence.", "Be accurate.")
    assert eval_result.score >= 1


@pytest.mark.live
@pytest.mark.openai
def test_openai_router_fallback():
    class FallbackModule(BaseModule):
        models: List[str] = ["openai/non-existent-model-xyz", "openai/gpt-4o-mini"]
        prompt: str = "Say OK"

    module = FallbackModule()
    router = ModelRouter(module)
    res = router.model_response(module)
    assert res is not None


@pytest.mark.live
@pytest.mark.openai
def test_openai_tool_calling(capsys=None):
    CALLED_TOOLS.clear()
    res = run_model_test(
        model="openai/gpt-4o-mini",
        prompt="Multiply 12.5 by 8.4 and also check weather status for Tokyo.",
        tools=[calculate_multiply, safe_weather_lookup],
    )
    assert res["response"] is not None
    assert isinstance(res["response"], str)
    assert len(res["response"]) > 0

    if capsys:
        captured = capsys.readouterr()
        assert "[TOOL CALLED]" in captured.out or len(CALLED_TOOLS) > 0
        assert "calculate_multiply" in CALLED_TOOLS or "calculate_multiply" in captured.out
    else:
        assert len(CALLED_TOOLS) > 0
