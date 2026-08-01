import os
import atexit
from typing import Optional, Dict

from utils.logger import get_logger
from utils.file_ops import read_csv
from .model_names import strip_provider_prefix

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
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cached_tokens = 0
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
                    "api_type": (row.get('api_type') or '').strip(),
                }
        except Exception as e:
            logger.error(f"Failed to load pricing from {csv_path}: {e}")
        return pricing_data

    def get_model_api_type(self, model_name: str) -> Optional[str]:
        """Returns the api_type (e.g. 'chat_completions', 'responses') configured for an OpenAI model if available."""
        model_name = strip_provider_prefix(model_name)
        rates = self.pricing.get(model_name)
        if rates:
            api_type = rates.get("api_type")
            if api_type and api_type.upper() != "N/A":
                return api_type
        return None

    def calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        """Calculates estimated cost based on token counts, including tiered pricing thresholds."""
        model_name = strip_provider_prefix(model_name)
        rates = self.pricing.get(model_name)

        if not rates:
            logger.warning(
                f"No pricing row for model '{model_name}'; it will be reported at "
                f"$0.00. The call itself is unaffected. Add a row to "
                f"assets/model_pricing.csv to track its cost."
            )
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

    def record_transaction(
        self,
        module_name: str,
        model_name: str,
        costs: dict,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        status: str = "success",
    ):
        """Records a single transaction and updates global metrics."""
        self._call_history.append({
            "module": module_name,
            "model": model_name,
            "duration": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "status": status,
            **costs
        })
        if status == "success":
            self._total_input_cost += costs["input_cost"]
            self._total_output_cost += costs["output_cost"]
            self._total_cached_cost += costs["cached_cost"]
            self._total_overall_cost += costs["total_cost"]
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_cached_tokens += cached_tokens

    def record_failed_attempt(
        self,
        module_name: str,
        model_name: str,
        duration: float,
        error: Optional[Exception] = None,
    ):
        """Records a failed model attempt with zero tokens and zero cost."""
        self._call_history.append({
            "module": module_name,
            "model": model_name,
            "duration": duration,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "status": "failed",
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cached_cost": 0.0,
            "total_cost": 0.0,
            "error": str(error) if error else None,
        })
        if error:
            logger.warning(f"Recorded failed attempt for model '{model_name}': {error}")

    def print_final_summary(self):
        """Prints itemized transactions and consolidated session costs cleanly on program exit."""
        if self._summary_printed or not self._call_history:
            return

        self._summary_printed = True

        table_width = 168
        print("\n" + "=" * table_width)
        print("ITEMIZED TRANSACTION PRICING SUMMARY (NEW ARCHITECTURE)")
        print("=" * table_width)
        print(
            f"{'SR.':<4} | {'MODULE':<22} | {'MODEL':<32} | {'WALL TIME':<12} | "
            f"{'IN TOK':<8} | {'OUT TOK':<8} | {'CACHE TOK':<9} | "
            f"{'INPUT':<12} | {'OUTPUT':<12} | {'CACHED':<12} | {'TOTAL':<12}"
        )
        print("-" * table_width)

        for i, call in enumerate(self._call_history, 1):
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

        successful_calls = [c for c in self._call_history if c.get('status') != 'failed']
        num_calls = len(self._call_history) or 1
        num_successful = len(successful_calls) or 1
        avg_duration = sum(c['duration'] for c in self._call_history) / num_calls
        print(
            f"{'':<4} | {'AVERAGE':<22} | {'':<32} | "
            f"{avg_duration:<11.2f}s | "
            f"{(self._total_input_tokens / num_successful):<8.1f} | "
            f"{(self._total_output_tokens / num_successful):<8.1f} | "
            f"{(self._total_cached_tokens / num_successful):<9.1f} | "
            f"${(self._total_input_cost / num_successful):<11.6f} | "
            f"${(self._total_output_cost / num_successful):<11.6f} | "
            f"${(self._total_cached_cost / num_successful):<11.6f} | "
            f"${(self._total_overall_cost / num_successful):<11.6f}"
        )

        print("-" * table_width)
        total_duration = sum(c['duration'] for c in self._call_history)
        print(
            f"{'':<4} | {'TOTALS':<22} | {'':<32} | "
            f"{total_duration:<11.2f}s | "
            f"{self._total_input_tokens:<8} | {self._total_output_tokens:<8} | "
            f"{self._total_cached_tokens:<9} | "
            f"${self._total_input_cost:<11.6f} | ${self._total_output_cost:<11.6f} | "
            f"${self._total_cached_cost:<11.6f} | ${self._total_overall_cost:<11.6f}"
        )
        print("=" * table_width + "\n")

        try:
            logger.info({"session_history": self._call_history})
            logger.info({
                "session_totals": {
                    "input_cost": self._total_input_cost,
                    "output_cost": self._total_output_cost,
                    "cached_cost": self._total_cached_cost,
                    "overall_cost": self._total_overall_cost,
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                    "cached_tokens": self._total_cached_tokens,
                    "overall_duration": total_duration
                }
            })
        except Exception:
            pass

# Export the singleton instance explicitly
cost_tracker = CostTracker()
