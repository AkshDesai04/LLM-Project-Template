import pytest

from core.modules.base import Base
from core.llm_models.router import ModelRouter


class CustomModuleOnlyModels(Base):
    models: list[str] = ["openai/gpt-4o", "gemini/gemini-2.5-flash", "anthropic/claude-3-5-sonnet"]


class CustomModuleBoth(Base):
    model: str = "perplexity/sonar-pro"
    models: list[str] = ["gemini/gemini-2.5-flash", "openai/gpt-4o"]


class CustomModuleOnlyModel(Base):
    model: str = "ollama/llama3"
    models: list[str] = []


@pytest.mark.llm
@pytest.mark.unit
def test_base_defaults():
    module = Base()
    assert module.model is None
    assert module.models == [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
    ]
    
    # ModelRouter with default Base uses models[0] as primary and rest as fallbacks
    router = ModelRouter(module)
    assert router._model_chain == [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
    ]


@pytest.mark.llm
@pytest.mark.unit
def test_only_models_given():
    module = CustomModuleOnlyModels()
    router = ModelRouter(module)
    # First model in models is primary, other models are ordered fallbacks
    assert router._model_chain == [
        "openai/gpt-4o",
        "gemini/gemini-2.5-flash",
        "anthropic/claude-3-5-sonnet",
    ]


@pytest.mark.llm
@pytest.mark.unit
def test_both_model_and_models_given():
    module = CustomModuleBoth()
    router = ModelRouter(module)
    # model parameter is primary, models list contains ordered fallbacks
    assert router._model_chain == [
        "perplexity/sonar-pro",
        "gemini/gemini-2.5-flash",
        "openai/gpt-4o",
    ]


@pytest.mark.llm
@pytest.mark.unit
def test_only_model_given():
    module = CustomModuleOnlyModel()
    router = ModelRouter(module)
    # model parameter is primary with no fallbacks
    assert router._model_chain == ["ollama/llama3"]


@pytest.mark.llm
@pytest.mark.unit
def test_neither_model_nor_models_raises_error():
    class EmptyModule(Base):
        model: str | None = None
        models: list[str] = []

    module = EmptyModule()
    try:
        ModelRouter(module)
        assert False, "Expected ValueError when neither model nor models specified"
    except ValueError as e:
        assert "Module must specify at least 'model' or 'models'" in str(e)


@pytest.mark.llm
@pytest.mark.unit
def test_model_prepending_without_deduplication():
    class DuplicateModule(Base):
        model: str = "openai/gpt-4o"
        models: list[str] = ["openai/gpt-4o", "gemini/gemini-2.5-flash", "openai/gpt-4o"]

    module = DuplicateModule()
    router = ModelRouter(module)
    assert router._model_chain == ["openai/gpt-4o", "openai/gpt-4o", "gemini/gemini-2.5-flash", "openai/gpt-4o"]


if __name__ == '__main__':
    test_base_defaults()
    test_only_models_given()
    test_both_model_and_models_given()
    test_only_model_given()
    test_neither_model_nor_models_raises_error()
    test_model_prepending_without_deduplication()
    print("All 6 model selection tests passed!")
