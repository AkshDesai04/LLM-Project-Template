from core.modules.base import Base
from utils.io import read_prompt


class FileSummaryPrompt(Base):
    prompt: str = read_prompt("test_prompt")
    model: str = "openai/o3-mini-2025-01-31"
    stream: bool = False