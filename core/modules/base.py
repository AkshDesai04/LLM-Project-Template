from pydantic import BaseModel
from typing import Optional, Any

from utils.file_ops import read_prompt


class Base(BaseModel):
    prompt: str = read_prompt("prompt")

    structure: Optional[Any] = None
    model: str = "gemini-2.5-pro"

    top_p: float = 0.8
    top_k: int = 40
    temperature: float = 0.2
    reasoning_budget: Optional[int] = None

    response_mime_type: str = "application/json"