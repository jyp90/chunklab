import pytest

from chunklab.core.embedders import MissingApiKeyError
from chunklab.core.questions.llm import FakeLLM, GeminiChat, OpenAIChat, build_llm


def test_fake_llm_records_prompts():
    llm = FakeLLM(reply="Q?")
    assert llm.complete("p1") == "Q?"
    assert llm.prompts == ["p1"]


def test_build_llm_specs():
    assert isinstance(build_llm("fake"), FakeLLM)
    assert isinstance(build_llm("openai"), OpenAIChat)
    assert build_llm("openai:gpt-4o").model == "gpt-4o"  # type: ignore[attr-defined]
    assert isinstance(build_llm("gemini"), GeminiChat)
    with pytest.raises(KeyError, match="unknown llm provider"):
        build_llm("anthropic")


def test_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        OpenAIChat().complete("x")
    with pytest.raises(MissingApiKeyError):
        GeminiChat().complete("x")
