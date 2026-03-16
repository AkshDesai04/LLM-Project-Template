from core.modules.base import Base


class FileSummaryPrompt(Base):
    def __init__(self, model: str):
        super().__init__()

        self.prompt: str = ("Summarize all the provided files and return a detailed summary of each file. For each "
                            "file, provide a short but to the point summary and then a total summary of all included "
                            "files. At the end, have a final summary including all files. Keep everything short.")
        self.model: str = model
