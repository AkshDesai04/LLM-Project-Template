import os
from core.llm_models_new.router import ModelRouter
from tests.TestModule import FileSummaryPrompt
from utils.env_ops import get_keys_dict
from utils.logger import get_logger
import time

# Initialize logger for this test script
logger = get_logger("TestFile")

def main():
    """
    Simplified test script to hit Ollama model via the new ModelRouter.
    """
    logger.info("Starting TestFile execution for Ollama...")

    # 1. Get API Keys (Ollama might use OLLAMA_URL from .env)
    try:
        api_keys = get_keys_dict()
    except Exception as e:
        logger.warning(f"Could not retrieve API keys (continuing for Ollama): {e}")
        api_keys = {}

    model_name = "ollama/llama3.2:1b"
    
    try:
        # 2. Initialize the Prompt Module
        prompt_module = FileSummaryPrompt(model=model_name)
        
        # 3. Initialize Model Router (New Architecture)
        router = ModelRouter(prompt_module, api_keys)

        # 4. Hit the model
        logger.info(f"Sending request to model: {model_name}...")
        start_time = time.time()
        response = router.model_response(prompt_module)
        duration = time.time() - start_time
        
        logger.info(f"Successfully received response from {model_name} in {duration:.2f}s")
        print("="*70)
        print(f"MODEL: {model_name}")
        print("="*70)
        print(f"RESPONSE: {response}")
        print("="*70)

    except Exception as e:
        logger.error(f"Error during Ollama test: {e}")

    logger.info("TestFile execution completed.")

if __name__ == "__main__":
    main()
