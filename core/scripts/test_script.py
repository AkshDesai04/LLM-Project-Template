import os
import inspect
from core.llm_models.router import ModelRouter
from core.modules.test_module import FileSummaryPrompt
from utils.logger import get_logger
from utils.parallel_executor import parallel_execute

# Initialize logger for this test script
logger = get_logger("TestFile")

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
        os.makedirs("results", exist_ok=True)
        
        # Save the result to a file
        file_path = f"results/{index}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_response_text)
            
        logger.info(f"Call #{index} completed and saved to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error in parallel call #{index}: {e}")
        return e

def main():
    # Number of parallel executions
    total_calls = 1000
    indices = list(range(total_calls))
    
    logger.info(f"Starting parallel execution of 100 LLM calls...")
    
    # Execute the calls in parallel
    # We use max_threads=20 to balance speed and system/API stability
    results = parallel_execute(
        target_function=run_llm_call,
        data=indices,
        max_threads=20,
        max_retries=1,
        retry_timer=2
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
