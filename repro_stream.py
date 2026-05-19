from core.llm_models.router import ModelRouter
from core.modules.test_module import FileSummaryPrompt
from utils.logger import get_logger
import inspect

# Initialize logger
logger = get_logger("ReproScript")

def main():
    prompt_module = FileSummaryPrompt()
    # Ensure stream is True for this test
    prompt_module.stream = True
    
    router = ModelRouter(prompt_module)
    print(f"Calling model_response for {prompt_module.model}...")
    response = router.model_response(prompt_module)
    
    print(f"Response type: {type(response)}")
    
    if inspect.isgenerator(response):
        print("Iterating over generator...")
        for chunk in response:
            # For Ollama, chunk is a dict
            if isinstance(chunk, dict):
                content = chunk.get('message', {}).get('content', '')
                print(content, end='', flush=True)
            else:
                # Fallback for other providers if we were testing them
                print(f"\n[Unknown chunk type: {type(chunk)}]", end='')
        print("\nStream finished.")
    else:
        print("Response is not a generator:")
        print(response)

if __name__ == "__main__":
    main()
