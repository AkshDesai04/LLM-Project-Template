"""
High-speed parallel orchestrator executing OpenAI test scenarios with concurrency max_threads=8.
Excluded high-cost models (output cost > $40/M tokens).
"""

import sys
import time
from typing import Tuple, Any

from utils.logging import get_logger
from utils.concurrency.parallel_executor import parallel_execute
from core.llm_models.cost_tracker import cost_tracker
from tests.test_live_openai_models import (
    test_openai_standard_models,
    test_openai_reasoning_models,
    test_openai_streaming_and_judge,
    test_openai_router_fallback,
    test_openai_tool_calling,
)
from tests.test_live_openai_embeddings import (
    test_openai_embeddings_short_inputs,
    test_openai_embeddings_large_inputs,
    test_openai_embeddings_batch_and_options,
)
from tests.test_live_openai_multimodal import (
    test_openai_single_file_uploads,
    test_openai_dual_media_combos,
    test_openai_quad_multi_media_combo,
)

logger = get_logger("ParallelOpenAIRunner")

TEST_FUNCTIONS = [
    ("Models: OpenAI Standard", test_openai_standard_models),
    ("Models: OpenAI Reasoning", test_openai_reasoning_models),
    ("Models: Streaming & Judge", test_openai_streaming_and_judge),
    ("Models: Router Fallback", test_openai_router_fallback),
    ("Models: Tool Calling", test_openai_tool_calling),
    ("Embeddings: Short Inputs", test_openai_embeddings_short_inputs),
    ("Embeddings: Large Inputs", test_openai_embeddings_large_inputs),
    ("Embeddings: Batch & Options", test_openai_embeddings_batch_and_options),
    ("Multimodal: Single Uploads", test_openai_single_file_uploads),
    ("Multimodal: Dual Combos", test_openai_dual_media_combos),
    ("Multimodal: Quad Combo", test_openai_quad_multi_media_combo),
]


def _worker_run(name: str, func: Any) -> Tuple[str, bool, Any]:
    """Worker function executed by ThreadPoolExecutor."""
    start_t = time.time()
    try:
        logger.info(f"Starting parallel test suite: {name}")
        func()
        duration = time.time() - start_t
        logger.info(f"SUCCESS: {name} finished in {duration:.2f}s")
        return name, True, duration
    except Exception as exc:
        duration = time.time() - start_t
        logger.error(f"FAILURE: {name} failed in {duration:.2f}s with error: {exc}")
        return name, False, exc


def main():
    print("=================================================================")
    print(" Starting Parallel OpenAI Live Integration Test Suite (Conc = 8) ")
    print("=================================================================")
    start_time = time.time()

    results = parallel_execute(
        target_function=_worker_run,
        data=TEST_FUNCTIONS,
        max_threads=8,
    )

    total_duration = time.time() - start_time
    passed = 0
    failed = 0

    print("\n---------------------- TEST SUMMARY REPORT ----------------------")
    for res in results:
        if isinstance(res, tuple):
            name, status, info = res
            if status:
                passed += 1
                print(f"  [PASSED] {name} ({info:.2f}s)")
            else:
                failed += 1
                print(f"  [FAILED] {name}: {info}")
        elif isinstance(res, Exception):
            failed += 1
            print(f"  [CRITICAL ERROR]: {res}")

    print(f"\nCompleted in {total_duration:.2f}s | Passed: {passed} | Failed: {failed}")
    print("=================================================================\n")

    cost_tracker.print_final_summary()
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
