"""
Unit tests for cost summary table printing (core.llm_models.cost_summary).
"""

from unittest.mock import patch
import pytest

from core.llm_models.cost_summary import print_summary_table


@pytest.mark.llm
@pytest.mark.unit
def test_print_summary_table_empty():
    with patch("builtins.print") as mock_print:
        print_summary_table([], 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
        mock_print.assert_not_called()


@pytest.mark.llm
@pytest.mark.unit
def test_print_summary_table_with_history(capsys):
    history = [
        {
            "module": "TestMod",
            "model": "openai/gpt-4o",
            "duration": 1.25,
            "input_tokens": 100,
            "output_tokens": 50,
            "cached_tokens": 0,
            "status": "success",
            "input_cost": 0.00025,
            "output_cost": 0.0005,
            "cached_cost": 0.0,
            "total_cost": 0.00075,
        },
        {
            "module": "TestMod",
            "model": "gemini/gemini-2.5-flash",
            "duration": 0.5,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "status": "failed",
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cached_cost": 0.0,
            "total_cost": 0.0,
        },
    ]

    print_summary_table(
        call_history=history,
        total_input_cost=0.00025,
        total_output_cost=0.0005,
        total_cached_cost=0.0,
        total_overall_cost=0.00075,
        total_input_tokens=100,
        total_output_tokens=50,
        total_cached_tokens=0,
    )

    captured = capsys.readouterr()
    assert "ITEMIZED TRANSACTION PRICING SUMMARY" in captured.out
    assert "TestMod" in captured.out
    assert "gemini/gemini-2.5-flash (failed)" in captured.out
    assert "TOTALS" in captured.out
