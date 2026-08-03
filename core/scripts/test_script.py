import os
from core.llm_models.router import ModelRouter
from core.modules.test_module import FileSummaryPrompt
from utils.logging import get_logger
from utils.concurrency import parallel_execute

# Initialize logger for this test script
logger = get_logger("TestFile")

# Top-level Constants
RESULTS_DIR_NAME: str = "results"
DEFAULT_TOTAL_CALLS: int = 1000
DEFAULT_MAX_THREADS: int = 20
DEFAULT_MAX_RETRIES: int = 1
DEFAULT_RETRY_TIMER: float = 2.0
FILE_ENCODING: str = "utf-8"


def run_llm_call(index: int):
    """
    Worker function to execute a single LLM call and save the result.
    """
    logger.info(f"Starting parallel call #{index}")
    try:
        prompt_module = FileSummaryPrompt()
        router = ModelRouter(prompt_module)
        response = router.model_response(prompt_module)
        
        # Ensure response is treated as text
        full_response_text = str(response)

        # Ensure the results directory exists
        os.makedirs(RESULTS_DIR_NAME, exist_ok=True)
        
        # Save the result to a file
        file_path = os.path.join(RESULTS_DIR_NAME, f"{index}.md")
        with open(file_path, "w", encoding=FILE_ENCODING) as f:
            f.write(full_response_text)
            
        logger.info(f"Call #{index} completed and saved to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error in parallel call #{index}: {e}")
        return e


def main():
    # Number of parallel executions
    total_calls = DEFAULT_TOTAL_CALLS
    indices = list(range(total_calls))
    
    logger.info(f"Starting parallel execution of {total_calls} LLM calls...")
    
    # Execute the calls in parallel
    results = parallel_execute(
        target_function=run_llm_call,
        data=indices,
        max_threads=DEFAULT_MAX_THREADS,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_timer=DEFAULT_RETRY_TIMER
    )
    
    # Log summary of results
    success_count = sum(1 for res in results if isinstance(res, str))
    error_count = total_calls - success_count
    
    logger.info(f"Parallel Execution Summary:")
    logger.info(f"Total Calls: {total_calls}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {error_count}")

if __name__ == "__main__":
    main()
