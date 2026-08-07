"""
Unit tests for CostTracker singleton (core.llm_models.cost_tracker).
"""

from unittest.mock import patch
import pytest

from core.llm_models.cost_tracker import CostTracker, cost_tracker


@pytest.fixture(autouse=True)
def reset_cost_tracker_state():
    tracker = CostTracker()
    with tracker._lock:
        tracker._call_history.clear()
        tracker._total_input_cost = 0.0
        tracker._total_output_cost = 0.0
        tracker._total_cached_cost = 0.0
        tracker._total_overall_cost = 0.0
        tracker._total_input_tokens = 0
        tracker._total_output_tokens = 0
        tracker._total_cached_tokens = 0
        tracker._summary_printed = False
    yield


@pytest.mark.llm
@pytest.mark.unit
def test_cost_tracker_singleton():
    t1 = CostTracker()
    t2 = CostTracker()
    assert t1 is t2
    assert t1 is cost_tracker


@pytest.mark.llm
@pytest.mark.unit
def test_calculate_cost_known_model():
    tracker = CostTracker()
    # Mock pricing dictionary
    tracker.pricing = {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
            "cached": 1.25,
            "input_above_200k": 5.00,
            "output_above_200k": 20.00,
            "api_type": "CHAT",
        }
    }

    # Standard usage: 100k input, 50k output, 10k cached
    costs = tracker.calculate_cost("openai/gpt-4o", prompt_tokens=100_000, output_tokens=50_000, cached_tokens=10_000)
    assert pytest.approx(costs["input_cost"], 0.0001) == 0.25
    assert pytest.approx(costs["output_cost"], 0.0001) == 0.50
    assert pytest.approx(costs["cached_cost"], 0.0001) == 0.0125
    assert pytest.approx(costs["total_cost"], 0.0001) == 0.7625


@pytest.mark.llm
@pytest.mark.unit
def test_calculate_cost_tiered_high_context():
    tracker = CostTracker()
    tracker.pricing = {
        "gemini-2.5-pro": {
            "input": 1.25,
            "output": 5.00,
            "cached": 0.30,
            "input_above_200k": 2.50,
            "output_above_200k": 10.00,
            "api_type": "GENAI",
        }
    }

    # 300k input tokens (> 200k threshold)
    # First 200k = (200k / 1M) * 1.25 = 0.25
    # Next 100k = (100k / 1M) * 2.50 = 0.25
    # Total input cost = 0.50
    costs = tracker.calculate_cost("gemini/gemini-2.5-pro", prompt_tokens=300_000, output_tokens=0)
    assert pytest.approx(costs["input_cost"], 0.0001) == 0.50


@pytest.mark.llm
@pytest.mark.unit
def test_calculate_cost_unknown_model():
    tracker = CostTracker()
    costs = tracker.calculate_cost("custom/unknown-model", prompt_tokens=1000, output_tokens=500)
    assert costs["total_cost"] == 0.0


@pytest.mark.llm
@pytest.mark.unit
def test_record_transaction_and_failed_attempt():
    tracker = CostTracker()
    tracker.record_transaction(
        module_name="TestModule",
        model_name="openai/gpt-4o",
        costs={"input_cost": 0.1, "output_cost": 0.2, "cached_cost": 0.0, "total_cost": 0.3},
        duration=1.2,
        input_tokens=1000,
        output_tokens=500,
        status="success",
    )

    tracker.record_failed_attempt(
        module_name="TestModule",
        model_name="gemini/gemini-2.5-pro",
        duration=0.5,
        error=ValueError("API Timeout"),
    )

    assert len(tracker._call_history) == 2
    assert tracker._call_history[0]["status"] == "success"
    assert tracker._call_history[1]["status"] == "failed"
    assert tracker._total_overall_cost == 0.3
    assert tracker._total_input_tokens == 1000
