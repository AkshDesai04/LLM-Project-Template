import sys
import os

# Add project root to sys.path to allow imports from core and utils
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.modules.test_module import TestModule
from core.llm_models_new.router import ModelRouter
from utils.env_ops import get_keys_dict
from utils.logger import get_logger

logger = get_logger("RunTestModule")

def main():
    logger.info("Starting TestModule execution script...")
    
    # Load API keys
    api_keys = get_keys_dict()
    
    # Instantiate the module
    module = TestModule()

    # Initialize the router
    # Note: TestModule defines model = "gpt-4.1-nano-2025-04-14", which routes to OpenAI
    router = ModelRouter(module, api_keys)
    
    logger.info(f"Running module with model: {module.model}")
    
    # Call model_response - it will now pick up settings from the module
    response = router.model_response(module)
    
    print("\n--- STREAMING OUTPUT ---")
    
    # Check if the response is an iterator (streaming)
    if hasattr(response, '__iter__') and not isinstance(response, (str, dict, list)):
        for chunk in response:
            # Handle OpenAI-style chunks
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                
                # Print content delta
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                    print(choice.delta.content, end="", flush=True)
                
                # Print logprobs if requested and available
                if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                    lp = choice.logprobs.content[0]
                    # Print the logprob of the chosen token for verification
                    print(f" [lp: {lp.logprob:.2f}]", end="", flush=True)
            
            # Handle Gemini-style chunks (if it were routed to Gemini)
            elif hasattr(chunk, 'text'):
                print(chunk.text, end="", flush=True)
                
        print("\n--- STREAMING FINISHED ---")
    else:
        # Fallback if streaming didn't trigger for some reason
        print(f"Output: {response}")

if __name__ == "__main__":
    main()
