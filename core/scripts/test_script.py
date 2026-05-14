from core.llm_models.router import ModelRouter
from core.modules.test_module import FileSummaryPrompt
from utils.logger import get_logger

# Initialize logger for this test script
logger = get_logger("TestFile")

def main():
    prompt_module = FileSummaryPrompt()
    router = ModelRouter(prompt_module)
    print(router.model_response(prompt_module))

if __name__ == "__main__":
    main()
