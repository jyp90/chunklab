import pytest

from chunklab.core.embedders import MissingApiKeyError, build_embedder
from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.embedders.gemini import GeminiEmbedder
from chunklab.core.embedders.local import LocalEmbedder
from chunklab.core.embedders.openai import OpenAIEmbedder


def test_build_fake():
    assert isinstance(build_embedder("fake"), FakeEmbedder)


def test_build_openai_default_and_custom_model():
    e = build_embedder("openai")
    assert isinstance(e, OpenAIEmbedder)
    assert e.name == "openai:text-embedding-3-small"
    assert build_embedder("openai:text-embedding-3-large").name == "openai:text-embedding-3-large"


def test_build_gemini_and_local_names():
    g = build_embedder("gemini")
    assert isinstance(g, GeminiEmbedder)
    assert g.name == "gemini:gemini-embedding-001"
    loc = build_embedder("local:all-MiniLM-L6-v2")
    assert isinstance(loc, LocalEmbedder)
    assert loc.name == "local:all-MiniLM-L6-v2"


def test_build_unknown_provider():
    with pytest.raises(KeyError, match="unknown embedder provider"):
        build_embedder("cohere:embed-v3")


def test_openai_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        OpenAIEmbedder().embed(["x"])


def test_gemini_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="GEMINI_API_KEY"):
        GeminiEmbedder().embed(["x"])


def test_local_missing_dependency_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"chunklab\[local\]"):
        LocalEmbedder().embed(["x"])
