import os
import atexit
import threading
from typing import Optional, Dict

from utils.logging import get_logger
from utils.io import read_csv
from core.llm_models.model_names import strip_provider_prefix
from core.llm_models.cost_summary import print_summary_table

logger = get_logger("CostTracker")

HIGH_CONTEXT_TIER_THRESHOLD = 200_000
TOKENS_PER_MILLION = 1_000_000
PRICING_CSV_RELATIVE_PATH = os.path.join("assets", "model_pricing.csv")


class CostTracker:
    _instance = None
    _exit_handler_registered = False
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super(CostTracker, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._lock = threading.Lock()
        self._call_history = []
        self._total_input_cost = 0.0
        self._total_output_cost = 0.0
        self._total_cached_cost = 0.0
        self._total_overall_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cached_tokens = 0
        self._summary_printed = False

        self.pricing = self._load_pricing()

        if not CostTracker._exit_handler_registered:
            atexit.register(self.print_final_summary)
            CostTracker._exit_handler_registered = True

    def _load_pricing(self) -> Dict[str, Dict[str, float]]:
        pricing_data = {}
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(project_root, PRICING_CSV_RELATIVE_PATH)

        try:
            rows = read_csv(csv_path)
            for row in rows:
                if not row.get('sr', '').isdigit():
                    continue
                model_id = row.get('model_id')
                if not model_id:
                    continue

                def to_float(val: Optional[str]) -> float:
                    if not val or val.strip().upper() == 'N/A':
                        return 0.0
                    try:
                        return float(val)
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
            logger.error(f"Failed to load pricing from {csv_path}: {e}. Defaulting to empty pricing table ($0.00 reporting).")
        return pricing_data

    def get_model_api_type(self, model_name: str) -> Optional[str]:
        model_name = strip_provider_prefix(model_name)
        rates = self.pricing.get(model_name)
        if rates:
            api_type = rates.get("api_type")
            if api_type and api_type.upper() != "N/A":
                return api_type
        return None

    def calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        model_name = strip_provider_prefix(model_name)
        rates = self.pricing.get(model_name)

        if not rates:
            logger.warning(
                f"No pricing row for model '{model_name}'; it will be reported at $0.00. "
                f"The call itself is unaffected. Add a row to assets/model_pricing.csv to track its cost."
            )
            return {"input_cost": 0.0, "output_cost": 0.0, "cached_cost": 0.0, "total_cost": 0.0}

        tier = HIGH_CONTEXT_TIER_THRESHOLD
        input_cost = 0.0
        if prompt_tokens > 0:
            rate_in, rate_in_above = rates.get("input", 0.0), rates.get("input_above_200k", 0.0)
            if rate_in_above > 0 and prompt_tokens > tier:
                input_cost = (tier / TOKENS_PER_MILLION) * rate_in + ((prompt_tokens - tier) / TOKENS_PER_MILLION) * rate_in_above
            else:
                input_cost = (prompt_tokens / TOKENS_PER_MILLION) * rate_in

        output_cost = 0.0
        if output_tokens > 0:
            rate_out, rate_out_above = rates.get("output", 0.0), rates.get("output_above_200k", 0.0)
            if rate_out_above > 0 and output_tokens > tier:
                output_cost = (tier / TOKENS_PER_MILLION) * rate_out + ((output_tokens - tier) / TOKENS_PER_MILLION) * rate_out_above
            else:
                output_cost = (output_tokens / TOKENS_PER_MILLION) * rate_out

        cached_cost = (cached_tokens / TOKENS_PER_MILLION) * rates.get("cached", 0.0)
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_cost": cached_cost,
            "total_cost": input_cost + output_cost + cached_cost,
        }

    def record_transaction(self, module_name: str, model_name: str, costs: dict, duration: float, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0, status: str = "success"):
        with self._lock:
            self._call_history.append({
                "module": module_name, "model": model_name, "duration": duration,
                "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens,
                "status": status, **costs
            })
            if status == "success":
                self._total_input_cost += costs["input_cost"]
                self._total_output_cost += costs["output_cost"]
                self._total_cached_cost += costs["cached_cost"]
                self._total_overall_cost += costs["total_cost"]
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self._total_cached_tokens += cached_tokens

    def record_failed_attempt(self, module_name: str, model_name: str, duration: float, error: Optional[Exception] = None):
        with self._lock:
            self._call_history.append({
                "module": module_name, "model": model_name, "duration": duration,
                "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "status": "failed",
                "input_cost": 0.0, "output_cost": 0.0, "cached_cost": 0.0, "total_cost": 0.0,
                "error": str(error) if error else None,
            })
        if error:
            logger.warning(f"Recorded failed attempt for model '{model_name}': {error}")

    def print_final_summary(self):
        if self._summary_printed or not self._call_history:
            return
        self._summary_printed = True
        print_summary_table(
            self._call_history, self._total_input_cost, self._total_output_cost,
            self._total_cached_cost, self._total_overall_cost, self._total_input_tokens,
            self._total_output_tokens, self._total_cached_tokens
        )


cost_tracker = CostTracker()
