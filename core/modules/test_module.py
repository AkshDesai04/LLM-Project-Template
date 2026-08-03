from core.modules.base import Base
from utils.io import read_prompt


class FileSummaryPrompt(Base):
    prompt: str = ""
    model: str = "openai/o3-mini-2025-01-31"
    stream: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if not self.prompt:
            self.prompt = read_prompt("test_prompt")