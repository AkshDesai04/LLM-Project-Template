import os
import atexit
from typing import Optional, Dict

from utils.logger import get_logger
from utils.file_ops import read_csv

logger = get_logger("CostTracker")

class CostTracker:
    _instance = None
    _exit_handler_registered = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CostTracker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._call_history = []
        self._total_input_cost = 0.0
        self._total_output_cost = 0.0
        self._total_cached_cost = 0.0
        self._total_overall_cost = 0.0
        self._summary_printed = False

        # Load pricing dynamically
        self.pricing = self._load_pricing()

        if not CostTracker._exit_handler_registered:
            atexit.register(self.print_final_summary)
            CostTracker._exit_handler_registered = True

    def _load_pricing(self) -> Dict[str, Dict[str, float]]:
        """Loads model pricing from the assets/model_pricing.csv file."""
        pricing_data = {}
        # Resolve the root relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, "assets", "model_pricing.csv")

        try:
            rows = read_csv(csv_path)
            for row in rows:
                if not row.get('sr', '').isdigit():
                    continue

                model_id = row.get('model_id')
                if not model_id:
                    continue

                def to_float(value_str: Optional[str]) -> float:
                    if not value_str or value_str.strip().upper() == 'N/A':
                        return 0.0
                    try:
                        return float(value_str)
                    except (ValueError, TypeError):
                        return 0.0

                pricing_data[model_id] = {
                    "input": to_float(row.get('input_cost_per_million')),
                    "output": to_float(row.get('output_cost_per_million')),
                    "cached": to_float(row.get('context_caching_cost_per_million')),
                    "input_above_200k": to_float(row.get('input_cost_per_million_above_200k')),
                    "output_above_200k": to_float(row.get('output_cost_per_million_above_200k')),
                }
        except Exception as e:
            logger.error(f"Failed to load pricing from {csv_path}: {e}")
        return pricing_data

    def calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        """Calculates estimated cost based on token counts, including tiered pricing thresholds."""
        rates = self.pricing.get(model_name)

        if not rates:
            for key in self.pricing:
                if key in model_name:
                    rates = self.pricing[key]
                    break

        if not rates:
            logger.warning(f"No pricing information found for model '{model_name}'. Cost will be $0.00.")
            return {"input_cost": 0.0, "output_cost": 0.0, "cached_cost": 0.0, "total_cost": 0.0}

        tier_threshold = 200_000

        # Input Cost
        input_cost = 0.0
        if prompt_tokens > 0:
            rate_input = rates.get("input", 0.0)
            rate_input_above = rates.get("input_above_200k", 0.0)

            if rate_input_above > 0 and prompt_tokens > tier_threshold:
                cost_below = (tier_threshold / 1_000_000) * rate_input
                cost_above = ((prompt_tokens - tier_threshold) / 1_000_000) * rate_input_above
                input_cost = cost_below + cost_above
            else:
                input_cost = (prompt_tokens / 1_000_000) * rate_input

        # Output Cost
        output_cost = 0.0
        if output_tokens > 0:
            rate_output = rates.get("output", 0.0)
            rate_output_above = rates.get("output_above_200k", 0.0)

            if rate_output_above > 0 and output_tokens > tier_threshold:
                cost_below = (tier_threshold / 1_000_000) * rate_output
                cost_above = ((output_tokens - tier_threshold) / 1_000_000) * rate_output_above
                output_cost = cost_below + cost_above
            else:
                output_cost = (output_tokens / 1_000_000) * rate_output

        # Cached Cost
        cached_cost = (cached_tokens / 1_000_000) * rates.get("cached", 0.0)

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_cost": cached_cost,
            "total_cost": input_cost + output_cost + cached_cost,
        }

    def record_transaction(self, module_name: str, model_name: str, costs: dict, duration: float):
        """Records a single transaction and updates global metrics."""
        self._call_history.append({
            "module": module_name,
            "model": model_name,
            "duration": duration,
            **costs
        })
        self._total_input_cost += costs["input_cost"]
        self._total_output_cost += costs["output_cost"]
        self._total_cached_cost += costs["cached_cost"]
        self._total_overall_cost += costs["total_cost"]

    def print_final_summary(self):
        """Prints itemized transactions and consolidated session costs cleanly on program exit."""
        if self._summary_printed or not self._call_history:
            return

        self._summary_printed = True

        print("\n" + "=" * 132)
        print("ITEMIZED TRANSACTION PRICING SUMMARY (NEW ARCHITECTURE)")
        print("=" * 132)
        print(f"{'SR.':<4} | {'MODULE':<22} | {'MODEL':<25} | {'WALL TIME':<12} | {'INPUT':<12} | {'OUTPUT':<12} | {'CACHED':<12} | {'TOTAL':<12}")
        print("-" * 132)
        
        for i, call in enumerate(self._call_history, 1):
            print(f"{i:<4} | {call['module']:<22} | {call['model']:<25} | "
                  f"{call['duration']:<11.2f}s | "
                  f"${call['input_cost']:<11.6f} | ${call['output_cost']:<11.6f} | "
                  f"${call['cached_cost']:<11.6f} | ${call['total_cost']:<11.6f}")

        print("-" * 132)
        
        num_calls = len(self._call_history)
        avg_duration = sum(c['duration'] for c in self._call_history) / num_calls
        print(f"{'':<4} | {'AVERAGE':<22} | {'':<25} | "
              f"{avg_duration:<11.2f}s | "
              f"${(self._total_input_cost / num_calls):<11.6f} | ${(self._total_output_cost / num_calls):<11.6f} | "
              f"${(self._total_cached_cost / num_calls):<11.6f} | ${(self._total_overall_cost / num_calls):<11.6f}")

        print("-" * 132)
        total_duration = sum(c['duration'] for c in self._call_history)
        print(f"{'':<4} | {'TOTALS':<22} | {'':<25} | "
              f"{total_duration:<11.2f}s | "
              f"${self._total_input_cost:<11.6f} | ${self._total_output_cost:<11.6f} | "
              f"${self._total_cached_cost:<11.6f} | ${self._total_overall_cost:<11.6f}")
        print("=" * 132 + "\n")

        try:
            logger.info({"session_history": self._call_history})
            logger.info({
                "session_totals": {
                    "input_cost": self._total_input_cost,
                    "output_cost": self._total_output_cost,
                    "cached_cost": self._total_cached_cost,
                    "overall_cost": self._total_overall_cost,
                    "overall_duration": total_duration
                }
            })
        except Exception:
            pass

# Export the singleton instance explicitly
cost_tracker = CostTracker()
