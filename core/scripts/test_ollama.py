"""
Comprehensive Ollama provider integration test script.

Tests:
1. Text generation (ollama/qwen3.5:2b)
2. Structured JSON output validation (ollama/qwen3.5:2b)
3. Streaming text response (ollama/qwen3.5:2b)
4. Vision multimodal image analysis (ollama/qwen3-vl:2b)
5. Text vector embeddings (ollama/qwen3-embedding:8b-fp16)
6. Automatic model download / pull fallback (ollama/gemma3:1b)
"""

import sys
from utils.logging import get_logger
from core.llm_models.cost_tracker import cost_tracker
from core.scripts.ollama_test_cases import (
    test_text_generation,
    test_structured_output,
    test_streaming_output,
    test_vision_multimodal,
    test_embedding,
    test_missing_model_download,
)

logger = get_logger("TestOllamaScript")


def run_all_ollama_tests():
    print("\n" + "=" * 70)
    print("      OLLAMA PROVIDER EXTENSIVE COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Text Generation
    try:
        results["Text Generation (qwen3.5:2b)"] = test_text_generation()
    except Exception as e:
        logger.error(f"Text Generation Test Failed: {e}")
        results["Text Generation (qwen3.5:2b)"] = False

    # Test 2: Structured Output
    try:
        results["Structured Output (qwen3.5:2b)"] = test_structured_output()
    except Exception as e:
        logger.error(f"Structured Output Test Failed: {e}")
        results["Structured Output (qwen3.5:2b)"] = False

    # Test 3: Streaming
    try:
        chunks = test_streaming_output()
        results["Streaming (qwen3.5:2b)"] = len(chunks) > 0
    except Exception as e:
        logger.error(f"Streaming Test Failed: {e}")
        results["Streaming (qwen3.5:2b)"] = False

    # Test 4: Vision Multimodal
    try:
        results["Vision Multimodal (qwen3-vl:2b)"] = test_vision_multimodal()
    except Exception as e:
        logger.error(f"Vision Multimodal Test Failed: {e}")
        results["Vision Multimodal (qwen3-vl:2b)"] = False

    # Test 5: Embeddings
    try:
        results["Embeddings (qwen3-embedding:8b-fp16)"] = test_embedding()
    except Exception as e:
        logger.error(f"Embeddings Test Failed: {e}")
        results["Embeddings (qwen3-embedding:8b-fp16)"] = False

    # Test 6: Auto-Download Fallback
    try:
        results["Auto-Download Missing Model (gemma3:1b)"] = test_missing_model_download()
    except Exception as e:
        logger.error(f"Auto-Download Test Failed: {e}")
        results["Auto-Download Missing Model (gemma3:1b)"] = False

    print("\n" + "=" * 70)
    print("                 OLLAMA TEST RESULTS SUMMARY")
    print("=" * 70)
    all_passed = True
    for test_name, status in results.items():
        symbol = "PASS" if status else "FAIL"
        print(f"[{symbol}] {test_name}")
        if not status:
            all_passed = False

    print("=" * 70 + "\n")

    # Print Cost Tracker Summary
    cost_tracker.print_final_summary()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_all_ollama_tests()
