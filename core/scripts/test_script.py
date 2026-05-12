from core.llm_models.router import ModelRouter
from core.modules.test_module import FileSummaryPrompt
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

    try:
        # 2. Initialize the Prompt Module (Uses default model from class definition)
        prompt_module = FileSummaryPrompt()
        
        # 3. Initialize Model Router (New Architecture)
        router = ModelRouter(prompt_module, api_keys)

        # 4. Hit the model
        logger.info(f"Sending request to model: {prompt_module.model}...")
        start_time = time.time()
        response = router.model_response(prompt_module)
        duration = time.time() - start_time
        
        logger.info(f"Successfully received response from {prompt_module.model} in {duration:.2f}s")
        print("="*70)
        print(f"MODEL: {prompt_module.model}")
        print("="*70)
        print(f"RESPONSE: {response}")
        print("="*70)

    except Exception as e:
        logger.error(f"Error during Ollama test: {e}")

    logger.info("TestFile execution completed.")

if __name__ == "__main__":
    main()
