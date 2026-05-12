from pydantic import Field

from core.modules.base import Base


class TestStruct(Base):
    response: str = Field(
        ...,
        description="The first 1000 words of Lorem ipsum."
    )


class TestModule(Base):
    # model: str = "gpt-5-nano-2025-08-07"
    # model = "gpt-4.1-nano-2025-04-14"
    model = "gpt-5.5-pro-2026-04-23"

    prompt = "Give me the first 1000 words of Lorem ipsum."

    stream = False
    logprobs = False

    # structure = TestStruct

    service_tier = None

    response_mime_type = "application/json"