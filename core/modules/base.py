from pydantic import BaseModel


class Base(BaseModel):
    prompt = ""

    structure = None
    model = "gemini-2.5-pro"

    top_p = 0.8
    top_k = 40
    temperature = 0.2
    reasoning_budget = None

    response_mime_type = "application/json"