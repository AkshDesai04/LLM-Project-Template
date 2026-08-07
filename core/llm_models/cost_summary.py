"""
Cost tracker summary table formatting and reporting routines.
"""

from typing import List, Dict, Any
from utils.logging import get_logger

logger = get_logger("CostSummary")
DEFAULT_TABLE_WIDTH = 168


def print_summary_table(
    call_history: List[Dict[str, Any]],
    total_input_cost: float,
    total_output_cost: float,
    total_cached_cost: float,
    total_overall_cost: float,
    total_input_tokens: int,
    total_output_tokens: int,
    total_cached_tokens: int,
) -> None:
    """Prints itemized transactions and consolidated session costs cleanly on program exit."""
    if not call_history:
        return

    table_width = DEFAULT_TABLE_WIDTH
    print("\n" + "=" * table_width)
    print("ITEMIZED TRANSACTION PRICING SUMMARY (NEW ARCHITECTURE)")
    print("=" * table_width)
    print(
        f"{'SR.':<4} | {'MODULE':<22} | {'MODEL':<32} | {'WALL TIME':<12} | "
        f"{'IN TOK':<8} | {'OUT TOK':<8} | {'CACHE TOK':<9} | "
        f"{'INPUT':<12} | {'OUTPUT':<12} | {'CACHED':<12} | {'TOTAL':<12}"
    )
    print("-" * table_width)

    for i, call in enumerate(call_history, 1):
        model_label = call['model']
        if call.get('status') == 'failed':
            model_label = f"{call['model']} (failed)"

        print(
            f"{i:<4} | {call['module']:<22} | {model_label:<32} | "
            f"{call['duration']:<11.2f}s | "
            f"{call.get('input_tokens', 0):<8} | {call.get('output_tokens', 0):<8} | "
            f"{call.get('cached_tokens', 0):<9} | "
            f"${call['input_cost']:<11.6f} | ${call['output_cost']:<11.6f} | "
            f"${call['cached_cost']:<11.6f} | ${call['total_cost']:<11.6f}"
        )

    print("-" * table_width)

    successful_calls = [c for c in call_history if c.get('status') != 'failed']
    num_calls = len(call_history) or 1
    num_successful = len(successful_calls) or 1
    avg_duration = sum(c['duration'] for c in call_history) / num_calls
    print(
        f"{'':<4} | {'AVERAGE':<22} | {'':<32} | "
        f"{avg_duration:<11.2f}s | "
        f"{(total_input_tokens / num_successful):<8.1f} | "
        f"{(total_output_tokens / num_successful):<8.1f} | "
        f"{(total_cached_tokens / num_successful):<9.1f} | "
        f"${(total_input_cost / num_successful):<11.6f} | "
        f"${(total_output_cost / num_successful):<11.6f} | "
        f"${(total_cached_cost / num_successful):<11.6f} | "
        f"${(total_overall_cost / num_successful):<11.6f}"
    )

    print("-" * table_width)
    total_duration = sum(c['duration'] for c in call_history)
    print(
        f"{'':<4} | {'TOTALS':<22} | {'':<32} | "
        f"{total_duration:<11.2f}s | "
        f"{total_input_tokens:<8} | {total_output_tokens:<8} | "
        f"{total_cached_tokens:<9} | "
        f"${total_input_cost:<11.6f} | ${total_output_cost:<11.6f} | "
        f"${total_cached_cost:<11.6f} | ${total_overall_cost:<11.6f}"
    )
    print("=" * table_width + "\n")

    try:
        import sys
        if not sys.stdout.closed:
            logger.info({"session_history": call_history})
            logger.info({
                "session_totals": {
                    "input_cost": total_input_cost,
                    "output_cost": total_output_cost,
                    "cached_cost": total_cached_cost,
                    "overall_cost": total_overall_cost,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "cached_tokens": total_cached_tokens,
                    "overall_duration": total_duration
                }
            })
    except Exception:
        pass
