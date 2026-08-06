"""
Live integration tests for Gemini LLM models (2.5, 3.0, 3.1, 3.5, 3.6) and Tool Calling.
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
    """Helper runner for model responses using ModelRouter."""
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
@pytest.mark.gemini
def test_gemini_2_5_models():
    res_flash = run_model_test("gemini/gemini-2.5-flash", "What is 2+2?")
    assert res_flash["response"] is not None

    res_flash_reasoning = run_model_test(
        "gemini/gemini-2.5-flash", "Explain 2+2", reasoning_budget="low", return_reasoning=True
    )
    assert res_flash_reasoning["response"] is not None

    res_flash_struct = run_model_test(
        "gemini/gemini-2.5-flash", f"Provide capital details for France. Context: {LONG_CONTEXT}", structure=CapitalInfo
    )
    assert res_flash_struct["response"] is not None

    res_pro = run_model_test("gemini/gemini-2.5-pro", "Name a primary color.")
    assert res_pro["response"] is not None

    res_lite = run_model_test("gemini/gemini-2.5-flash-lite", "Say hello in French.")
    assert res_lite["response"] is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_3_0_and_3_1_models():
    res_3 = run_model_test("gemini/gemini-3-flash-preview", "What is 3+3?")
    assert res_3["response"] is not None

    res_3_1_pro = run_model_test(
        "gemini/gemini-3.1-pro-preview", "Solve: 5*5"
    )
    assert res_3_1_pro["response"] is not None

    res_3_1_lite = run_model_test("gemini/gemini-3.1-flash-lite", "Name one ocean.")
    assert res_3_1_lite["response"] is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_3_5_and_3_6_models():
    res_3_5_flash = run_model_test("gemini/gemini-3.5-flash", "What is 10-4?")
    assert res_3_5_flash["response"] is not None

    res_3_5_lite = run_model_test("gemini/gemini-3.5-flash-lite", "What is capital of France?")
    assert res_3_5_lite["response"] is not None

    res_3_6 = run_model_test("gemini/gemini-3.6-flash", "Define gravity briefly.")
    assert res_3_6["response"] is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_streaming_and_judge():
    res_stream = run_model_test("gemini/gemini-3.6-flash", "Count 1 to 5.", stream=True)
    assert res_stream["stream"] is True

    module = BaseModule(model="gemini/gemini-3.6-flash", prompt="Test prompt")
    router = ModelRouter(module)
    eval_result = router.evaluate_response("What is AI?", "AI is artificial intelligence.", "Be accurate.")
    assert eval_result.score >= 1


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_router_fallback():
    class FallbackModule(BaseModule):
        models: List[str] = ["gemini/non-existent-model-xyz", "gemini/gemini-3.6-flash"]
        prompt: str = "Say OK"

    module = FallbackModule()
    router = ModelRouter(module)
    res = router.model_response(module)
    assert res is not None


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_tool_calling(capsys):
    CALLED_TOOLS.clear()
    res = run_model_test(
        model="gemini/gemini-2.5-flash-lite",
        prompt="Multiply 12.5 by 8.4 and also check weather status for Tokyo.",
        tools=[calculate_multiply, safe_weather_lookup],
    )
    assert res["response"] is not None
    assert isinstance(res["response"], str)
    assert len(res["response"]) > 0

    captured = capsys.readouterr()
    print("\n--- Captured Tool Execution Output ---")
    print(captured.out)

    # Verify that the tool function print statement was emitted and tool was executed
    assert "[TOOL CALLED]" in captured.out or len(CALLED_TOOLS) > 0
    assert "calculate_multiply" in CALLED_TOOLS or "calculate_multiply" in captured.out
