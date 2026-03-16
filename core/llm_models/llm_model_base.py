import os
import inspect
import atexit
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.file_ops import read_file, read_csv
from ..modules.base import Base as BaseModule

logger = get_logger("LLM_Base")


class JudgeResult(BaseModel):
    score: int = Field(..., description="A score from 1 to 10 evaluating the response quality.")
    reasoning: str = Field(..., description="Detailed explanation for the assigned score.")
    improvements: Optional[str] = Field(None, description="Suggestions for improving the response.")


def load_pricing(csv_path: str) -> dict:
    """Loads model pricing from a CSV file."""
    pricing = {}
    try:
        rows = read_csv(csv_path)
        for row in rows:
            if not row.get('sr', '').isdigit():
                continue

            model_id = row.get('model_id')
            if not model_id:
                continue

            def to_float(value_str: Optional[str]) -> float:
                """Safely convert a string to float, handling None and 'N/A'."""
                if not value_str or value_str.strip().upper() == 'N/A':
                    return 0.0
                try:
                    return float(value_str)
                except (ValueError, TypeError):
                    return 0.0

            pricing[model_id] = {
                "input": to_float(row.get('input_cost_per_million')),
                "output": to_float(row.get('output_cost_per_million')),
                "cached": to_float(row.get('context_caching_cost_per_million')),
                "input_above_200k": to_float(row.get('input_cost_per_million_above_200k')),
                "output_above_200k": to_float(row.get('output_cost_per_million_above_200k')),
            }
    except Exception as e:
        logger.error(f"Failed to load pricing from {csv_path}: {e}")
    return pricing


# Load pricing from assets
# Now at core/llm_models/llm_model_base.py, root is 3 levels up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRICING_FILE = os.path.join(PROJECT_ROOT, "assets", "model_pricing.csv")
PRICING = load_pricing(PRICING_FILE)


class LLMModels(ABC):
    # Class-level variables to track total run costs and history across all instances
    _call_history = []
    _total_input_cost = 0.0
    _total_output_cost = 0.0
    _total_cached_cost = 0.0
    _total_overall_cost = 0.0
    _summary_printed = False

    def __init__(self, api_key: str, base_config: BaseModule):
        self.api_key = api_key
        self.model_name = base_config.model
        self.temperature = base_config.temperature
        self.top_p = base_config.top_p
        self.top_k = base_config.top_k
        self.fall_back_models = getattr(base_config, 'fall_back_models', None)
        self.structure = base_config.structure
        self.response_mime_type = base_config.response_mime_type

        # Register the exit handler once
        if not hasattr(LLMModels, "_exit_handler_registered"):
            def exit_handler():
                if not LLMModels._summary_printed:
                    LLMModels._summary_printed = True
                    LLMModels.print_final_summary()

            atexit.register(exit_handler)
            LLMModels._exit_handler_registered = True

    @staticmethod
    def print_final_summary():
        """Prints itemized transactions and consolidated session costs."""
        if LLMModels._call_history:
            # 1. Print consolidated human-readable table to console
            print("\n" + "=" * 132)
            print("ITEMIZED TRANSACTION PRICING SUMMARY")
            print("=" * 132)
            print(
                f"{'SR.':<4} | {'MODULE':<22} | {'MODEL':<25} | {'WALL TIME':<12} | {'INPUT':<12} | {'OUTPUT':<12} | {'CACHED':<12} | {'TOTAL':<12}")
            print("-" * 132)
            for i, call in enumerate(LLMModels._call_history, 1):
                print(f"{i:<4} | {call['module']:<22} | {call['model']:<25} | "
                      f"{call['duration']:<11.2f}s | "
                      f"${call['input_cost']:<11.6f} | ${call['output_cost']:<11.6f} | "
                      f"${call['cached_cost']:<11.6f} | ${call['total_cost']:<11.6f}")

            print("-" * 132)
            num_calls = len(LLMModels._call_history)
            avg_duration = sum(c['duration'] for c in LLMModels._call_history) / num_calls
            avg_input = LLMModels._total_input_cost / num_calls
            avg_output = LLMModels._total_output_cost / num_calls
            avg_cached = LLMModels._total_cached_cost / num_calls
            avg_total = LLMModels._total_overall_cost / num_calls

            print(f"{'':<4} | {'AVERAGE':<22} | {'':<25} | "
                  f"{avg_duration:<11.2f}s | "
                  f"${avg_input:<11.6f} | ${avg_output:<11.6f} | "
                  f"${avg_cached:<11.6f} | ${avg_total:<11.6f}")

            print("-" * 132)
            total_duration = sum(c['duration'] for c in LLMModels._call_history)
            print(f"{'':<4} | {'TOTALS':<22} | {'':<25} | "
                  f"{total_duration:<11.2f}s | "
                  f"${LLMModels._total_input_cost:<11.6f} | ${LLMModels._total_output_cost:<11.6f} | "
                  f"${LLMModels._total_cached_cost:<11.6f} | ${LLMModels._total_overall_cost:<11.6f}")
            print("=" * 132 + "\n")

            # 2. Log as JSON to file
            try:
                # Log the itemized call history
                logger.info({"session_history": LLMModels._call_history})

                # Log the final totals
                logger.info({
                    "session_totals": {
                        "input_cost": LLMModels._total_input_cost,
                        "output_cost": LLMModels._total_output_cost,
                        "cached_cost": LLMModels._total_cached_cost,
                        "overall_cost": LLMModels._total_overall_cost,
                        "overall_duration": total_duration
                    }
                })
            except Exception:
                # Silently fail if logger is already shut down
                pass

    @classmethod
    def record_transaction(cls, module_name: str, model_name: str, costs: dict, duration: float):
        """Records a single transaction and updates global totals."""
        LLMModels._call_history.append({
            "module": module_name,
            "model": model_name,
            "duration": duration,
            **costs
        })
        LLMModels._total_input_cost += costs["input_cost"]
        LLMModels._total_output_cost += costs["output_cost"]
        LLMModels._total_cached_cost += costs["cached_cost"]
        LLMModels._total_overall_cost += costs["total_cost"]

    def _calculate_cost(self, model_name: str, prompt_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict:
        """Calculates estimated cost based on token counts, including tiered pricing."""
        # Exact match first
        rates = PRICING.get(model_name)

        # Substring fallback
        if not rates:
            for key in PRICING:
                if key in model_name:
                    rates = PRICING[key]
                    break

        if not rates:
            raise ValueError(f"No pricing information found for model '{model_name}'.")

        # Tiered pricing threshold
        TIER_THRESHOLD = 200_000

        # --- Input Cost Calculation ---
        input_cost = 0.0
        if prompt_tokens > 0:
            rate_input = rates.get("input", 0.0)
            rate_input_above = rates.get("input_above_200k", 0.0)

            if rate_input_above > 0 and prompt_tokens > TIER_THRESHOLD:
                tokens_below = TIER_THRESHOLD
                tokens_above = prompt_tokens - TIER_THRESHOLD
                cost_below = (tokens_below / 1_000_000) * rate_input
                cost_above = (tokens_above / 1_000_000) * rate_input_above
                input_cost = cost_below + cost_above
            else:
                input_cost = (prompt_tokens / 1_000_000) * rate_input

        # --- Output Cost Calculation ---
        output_cost = 0.0
        if output_tokens > 0:
            rate_output = rates.get("output", 0.0)
            rate_output_above = rates.get("output_above_200k", 0.0)

            if rate_output_above > 0 and output_tokens > TIER_THRESHOLD:
                tokens_below = TIER_THRESHOLD
                tokens_above = output_tokens - TIER_THRESHOLD
                cost_below = (tokens_below / 1_000_000) * rate_output
                cost_above = (tokens_above / 1_000_000) * rate_output_above
                output_cost = cost_below + cost_above
            else:
                output_cost = (output_tokens / 1_000_000) * rate_output

        # --- Cached Cost Calculation ---
        cached_cost = (cached_tokens / 1_000_000) * rates.get("cached", 0.0)

        total_cost = input_cost + output_cost + cached_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_cost": cached_cost,
            "total_cost": total_cost,
        }

    @staticmethod
    def get_provider_by_model_name(model_name: str) -> str:
        """Determines the model provider based on the model name prefix."""
        model_name_lower = model_name.lower()
        if model_name_lower.startswith("gpt"):
            return "openai"
        elif model_name_lower.startswith("gemini"):
            return "google"
        elif model_name_lower.startswith(("o1-", "o3-")):
            return "openai"
        
        raise ValueError(f"Could not determine provider for model '{model_name}'. "
                         f"Model name should start with 'gpt' or 'gemini'.")

    @staticmethod
    def prepare_module(module: Any) -> BaseModule:
        """Shared logic to instantiate a module and load its prompt if necessary."""
        logger.info(f"Preparing module: {module}")
        if inspect.isclass(module):
            module_instance = module()
        elif isinstance(module, BaseModule):
            module_instance = module
        else:
            logger.error(f"Invalid module type: {type(module)}")
            raise TypeError("Module must be subclass or instance of modules.base.Base")

        if not getattr(module_instance, "prompt", None):
            prompt_path = getattr(module_instance, "prompt_path", None)
            if prompt_path:
                logger.info(f"Loading prompt from: {prompt_path}")
                module_instance.prompt = read_file(os.path.abspath(prompt_path))

        logger.info(f"Module {type(module_instance).__name__} prepared successfully.")
        return module_instance

    @abstractmethod
    def model_response(self, module: Any, uploaded_file: Optional[Any] = None) -> Optional[str]:
        pass

    @abstractmethod
    def upload_media(self, file_bytes: bytes, mime_type: str) -> Any:
        pass

    @abstractmethod
    def embed_content(self, input_content: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        pass

    @abstractmethod
    def evaluate_response(self, input_prompt: str, generated_output: str, rubric: Optional[str] = None) -> JudgeResult:
        """Evaluates a model's response using the LLM-as-a-Judge pattern."""
        pass