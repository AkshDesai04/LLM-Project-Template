from typing import Any, Literal

from pydantic import BaseModel, Field


class Base(BaseModel):
    # Core
    prompt: str | None = None
    system_prompt: str | None = None
    structure: Any | None = None

    # Model Selection
    model: str = "gemini-2.5-pro"

    fallback_models: list[str] = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    # Generation
    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 40
    max_tokens: int | None = None
    reasoning_budget: int | Literal["low", "medium", "high"] | None = None

    # Penalties & Sampling
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop_sequences: list[str] | None = None

    # Response
    response_mime_type: str = "application/json"
    stream: bool = False

    # Logging / Debugging
    logprobs: bool = False
    top_logprobs: int | None = None

    # Provider Features
    service_tier: Literal["auto", "default"] | None = None
    candidate_count: int = 1
    safety_settings: Any | None = None
    tools: Any | None = None

    # Search / Retrieval
    return_citations: bool = True
    search_domain_filter: list[str] | None = None
    search_recency_filter: str | None = None

    def to_runnable(self, api_keys: dict | None = None) -> Any:
        """
        Converts the module configuration into a LangChain Runnable.
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
            from core.llm_models.router import ModelRouter
        except ImportError:
            raise ImportError("LangChain dependencies missing. Install them to use to_runnable().")

        # 1. Create Prompt Template
        messages = []
        if self.system_prompt:
            messages.append(("system", self.system_prompt))
        
        # Use simple string prompt for now, or handle variables if needed
        messages.append(("user", self.prompt or "{input}"))
        prompt_template = ChatPromptTemplate.from_messages(messages)

        # 2. Get Model via Router
        # We need a way to get a LangChain-compatible model from the router.
        # For now, let's assume ModelRouter has a to_langchain_model() method.
        router = ModelRouter(self, api_keys=api_keys)
        model = router.to_langchain_model()

        # 3. Determine Output Parser
        if self.response_mime_type == "application/json" or self.structure:
            parser = JsonOutputParser(pydantic_object=self.structure) if self.structure else JsonOutputParser()
        else:
            parser = StrOutputParser()

        return prompt_template | model | parser
