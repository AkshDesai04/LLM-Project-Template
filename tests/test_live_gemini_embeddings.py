"""
Live integration tests for Gemini Embedding models across all supported file formats.
"""

from typing import List, Union
import pytest

from core.llm_models.router import ModelRouter
from core.modules.base import Base as BaseModule

EMBEDDING_MODELS = [
    "gemini/gemini-embedding-001",
    "gemini/gemini-embedding-2",
]

SHORT_FORMAT_SAMPLES = {
    "txt": "Artificial Intelligence is transforming software development.",
    "md": "# Title\n\n- Bullet point 1\n- Bullet point 2\n```python\nprint('hello')\n```",
    "json": '{"app": "llm-template", "version": "1.0.0", "active": true}',
    "csv": "id,name,role\n1,Alice,Engineer\n2,Bob,Designer\n",
    "py": "def add(a: int, b: int) -> int:\n    return a + b\n",
}

LONG_FORMAT_SAMPLES = {
    "txt": "Machine learning algorithms build a model based on sample data. " * 200,
    "md": ("# Architecture Document\n\n" + "## Overview\nDetailed specs go here.\n" * 100),
    "json": '{"dataset": [' + ','.join([f'{{"id": {i}, "val": "data_{i}"}}' for i in range(100)]) + ']}',
    "csv": "id,item,cost\n" + "\n".join([f"{i},item_{i},{i*1.5}" for i in range(100)]),
    "py": "# Comprehensive module\n" + "\n".join([f"def func_{i}():\n    return {i}" for i in range(60)]),
}


def run_embedding_test(target_model: str, content: Union[str, List[str]], **kwargs) -> List[float]:
    """Helper runner for embed_content using ModelRouter."""
    module = BaseModule(model=target_model)
    router = ModelRouter(module)
    return router.embed_content(content, model=target_model, **kwargs)


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_embeddings_short_inputs():
    for model in EMBEDDING_MODELS:
        for fmt, text_sample in SHORT_FORMAT_SAMPLES.items():
            vec = run_embedding_test(model, text_sample)
            assert isinstance(vec, list)
            assert len(vec) > 0
            assert isinstance(vec[0], float)


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_embeddings_large_inputs():
    for model in EMBEDDING_MODELS:
        for fmt, text_sample in LONG_FORMAT_SAMPLES.items():
            vec = run_embedding_test(model, text_sample)
            assert isinstance(vec, list)
            assert len(vec) > 0


@pytest.mark.live
@pytest.mark.gemini
def test_gemini_embeddings_batch_and_options():
    model = "gemini/gemini-embedding-2"
    batch_input = [SHORT_FORMAT_SAMPLES["txt"], SHORT_FORMAT_SAMPLES["md"]]
    vecs = run_embedding_test(model, batch_input, dimensions=768, task_type="SEMANTIC_SIMILARITY")
    assert isinstance(vecs, list)
    assert len(vecs) > 0
    assert len(vecs[0]) > 0
