from core.modules.base import Base
from utils.file_ops import read_prompt


class FileSummaryPrompt(Base):
    prompt: str = read_prompt('test_prompt')
    model: str = 'o3-mini-2025-01-31'
    stream: bool = False
    # reasoning_budget: str = 'xhigh'
    # return_reasoning = True