from core.modules.base import Base


class FileSummaryPrompt(Base):
    def __init__(self, model: str):
        super().__init__(
            prompt="write the first 10 words of lorem ipsum.",
            model=model
        )
