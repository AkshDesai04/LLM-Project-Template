from core.modules.base import Base
from utils.file_ops import read_prompt


class FileSummaryPrompt(Base):
    prompt: str = read_prompt('test_prompt')
    model: str = 'ollama/llama3.2:1b'
