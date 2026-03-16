import os
import mimetypes
from core.llm_models.model_router import ModelRouter
from tests.TestModule import FileSummaryPrompt
from utils.env_ops import get_keys_dict
from utils.file_ops import get_file
from utils.logger import get_logger
from utils.parallel_executor import parallel_execute
import time

# Initialize logger for this test script
logger = get_logger("TestFile")

def _summarize_with_model(model_name: str, api_keys: dict, test_files_dir: str) -> dict:
    """
    Function to be executed in parallel for each model.
    Initializes ModelRouter, uploads files, and gets a response for a given model.
    """
    logger.info(f"Starting summarization for model: {model_name}")
    try:
        # 1. Initialize the Prompt Module (TestModule) for this model
        prompt_module = FileSummaryPrompt(model=model_name)
        
        # 2. Initialize Model Router for this model
        # Each parallel task gets its own ModelRouter instance
        router = ModelRouter(prompt_module, api_keys)

        # 3. Gather and upload all files for this model's execution
        uploaded_files = []
        if not os.path.exists(test_files_dir):
            raise FileNotFoundError(f"Test files directory not found: {test_files_dir}")

        for filename in os.listdir(test_files_dir):
            file_path = os.path.join(test_files_dir, filename)
            if os.path.isfile(file_path):
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.jpg', '.jpeg']:
                        mime_type = "image/jpeg"
                    elif ext == '.png':
                        mime_type = "image/png"
                    elif ext == '.mp4':
                        mime_type = "video/mp4"
                    elif ext == '.pdf':
                        mime_type = "application/pdf"
                    else:
                        mime_type = "application/octet-stream"
                
                logger.info(f"[{model_name}] Processing '{filename}' (MIME: {mime_type})...")
                file_bytes = get_file(file_path)
                uploaded_file = router.upload_media(file_bytes, mime_type)
                uploaded_files.append(uploaded_file)
                logger.info(f"[{model_name}] Successfully uploaded '{filename}'.")

        if not uploaded_files:
            logger.warning(f"[{model_name}] No files found for summarization.")
            return {"model": model_name, "warning": "No files uploaded for summarization."}

        # 4. Hit the model with all uploaded files and the prompt
        logger.info(f"[{model_name}] Sending request to model with {len(uploaded_files)} files...")
        response = router.model_response(prompt_module, uploaded_file=uploaded_files)
        
        logger.info(f"Successfully received response from model: {model_name}")
        return {"model": model_name, "response": response}
    except Exception as e:
        logger.error(f"Error during summarization for model {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

def main():
    """
    Test script that uses FileSummaryPrompt to summarize all files in tests/test_files/
    via the ModelRouter, executing for multiple models in parallel.
    """
    logger.info("Starting TestFile execution...")

    # 1. Get API Keys from environment/AWS Secrets Manager
    try:
        api_keys = get_keys_dict()
    except Exception as e:
        logger.error(f"Failed to retrieve API keys: {e}")
        return

    # 2. Determine the test files directory (absolute path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files_dir = os.path.join(project_root, "tests", "test_files")
    
    if not os.path.exists(test_files_dir):
        logger.error(f"Test files directory not found: {test_files_dir}")
        return

    # 3. Define the list of models to test
    models_to_test = [
        "gemini-3.1-pro-preview",
        "gpt-5-nano-2025-08-07",
        "gemini-2.5-flash-lite",
        "gpt-5.4-2026-03-05",
    ]

    # Prepare data for parallel execution
    # Each item in data will be (model_name, api_keys, test_files_dir)
    parallel_data = [(model_name, api_keys, test_files_dir) for model_name in models_to_test]

    # 4. Execute summarization for all models in parallel
    if parallel_data: # Check if there are models to test
        logger.info(f"Executing summarization for {len(models_to_test)} models in parallel...")
        start_time_parallel = time.time()
        
        results = parallel_execute(
            target_function=_summarize_with_model,
            data=parallel_data,
            max_threads=-1, # Use all available CPU cores
            max_req_per_min=20, # Example rate limit: 20 requests per minute
            max_retries=2,
            retry_timer=1
        )
        duration_parallel = time.time() - start_time_parallel
        logger.info(f"Parallel execution completed in {duration_parallel:.2f}s.")

        # 5. Process and print results
        print("="*70)
        print("AGGREGATED MODEL SUMMARIZATION RESULTS")
        print("="*70)
        for res in results:
            if isinstance(res, Exception):
                print(f"Overall error in parallel task: {res}")
            elif "error" in res:
                print(f"Model: {res['model']}  Status: Failed  Error: {res['error']}")
            elif "warning" in res:
                print(f"Model: {res['model']} Status: Warning  Message: {res['warning']}")
            else:
                print(f"Model: {res['model']}  Status: Success Response: {res['response']}")
        print("="*70 )
            
    else:
        logger.warning("No models defined for summarization.")

    logger.info("TestFile execution completed.")

if __name__ == "__main__":
    main()