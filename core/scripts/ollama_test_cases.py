"""
Test case implementations for Ollama provider integration testing.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from utils.logging import get_logger
from core.modules.base import Base
from core.llm_models.router import ModelRouter

logger = get_logger("OllamaTestCases")


class PlanetInfo(BaseModel):
    name: str = Field(..., description="Name of the planet")
    type: str = Field(..., description="Type of planet e.g. Terrestrial or Gas Giant")
    distance_from_sun_au: float = Field(..., description="Distance from Sun in AU")
    moons: int = Field(..., description="Number of natural satellites")


class QwenTextPrompt(Base):
    prompt: str = "Explain quantum superposition in two concise sentences."
    system_prompt: str = "You are an expert physics teacher."
    model: str = "ollama/qwen3.5:2b"
    models: list[str] = []
    temperature: float = 0.2
    stream: bool = False


class QwenStructuredPrompt(Base):
    prompt: str = 'Return JSON for Mars: {"name": "Mars", "type": "Terrestrial", "distance_from_sun_au": 1.52, "moons": 2}'
    system_prompt: str = 'You are a JSON generator. Output only valid JSON.'
    model: str = "ollama/qwen3.5:2b"
    models: list[str] = []
    structure: Any = PlanetInfo
    response_mime_type: str = "application/json"


class QwenVisionPrompt(Base):
    prompt: str = "What colors or shapes are visible in this test image?"
    model: str = "ollama/qwen3-vl:2b"
    models: list[str] = []


def test_text_generation() -> bool:
    logger.info("=== 1. Testing Text Generation (ollama/qwen3.5:2b) ===")
    module = QwenTextPrompt()
    router = ModelRouter(module)
    res = router.model_response(module)
    logger.info(f"Response: {res}")
    return isinstance(res, str) and len(res) > 0


def test_structured_output() -> bool:
    logger.info("=== 2. Testing Structured JSON Output (ollama/qwen3.5:2b) ===")
    module = QwenStructuredPrompt()
    router = ModelRouter(module)
    res = router.model_response(module)
    logger.info(f"Structured Output: {res}")
    return isinstance(res, PlanetInfo) and res.name.lower() == "mars"


def test_streaming_output() -> List[str]:
    logger.info("=== 3. Testing Streaming Output (ollama/qwen3.5:2b) ===")
    module = QwenTextPrompt(stream=True)
    router = ModelRouter(module)
    stream_gen = router.model_response(module)
    chunks = []
    for chunk in stream_gen:
        text = chunk['message']['content'] if isinstance(chunk, dict) else str(chunk)
        chunks.append(text)
    logger.info(f"Streamed {len(chunks)} chunks.")
    return chunks


def test_vision_multimodal() -> bool:
    logger.info("=== 4. Testing Vision Multimodal (ollama/qwen3-vl:2b) ===")
    # Simple 1x1 PNG image as base64 data URL
    dummy_image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        }
    }
    module = QwenVisionPrompt()
    router = ModelRouter(module)
    res = router.model_response(module, uploaded_file=dummy_image)
    logger.info(f"Vision Response: {res}")
    return isinstance(res, str) and len(res) > 0


def test_embedding() -> bool:
    logger.info("=== 5. Testing Text Embeddings (ollama/qwen3-embedding:8b-fp16) ===")
    class EmbedModule(Base):
        model: str = "ollama/qwen3-embedding:8b-fp16"
        models: list[str] = []

    module = EmbedModule()
    router = ModelRouter(module)
    vector = router.embed_content("Hello Ollama Embeddings")
    logger.info(f"Generated vector embedding of length {len(vector)}")
    return isinstance(vector, list) and len(vector) > 0


def test_missing_model_download() -> bool:
    logger.info("=== 6. Testing Auto-Download / Pull (ollama/gemma3:1b) ===")
    class DownloadTestModule(Base):
        prompt: str = "Say hello in one word."
        model: str = "ollama/gemma3:1b"
        models: list[str] = []

    module = DownloadTestModule()
    router = ModelRouter(module)
    res = router.model_response(module)
    logger.info(f"Response after auto-downloading gemma3:1b: {res}")
    return isinstance(res, str) and len(res) > 0
