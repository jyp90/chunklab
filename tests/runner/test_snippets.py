import pytest

from chunklab.core.runner.run import ComboResult
from chunklab.core.runner.snippets import FRAMEWORKS, snippet


def _combo(
    chunker="recursive",
    params=None,
    embedder="openai:text-embedding-3-small",
    k=5,
    hybrid=False,
):
    return ComboResult(
        f"{chunker}|{embedder}|k={k}|hybrid={hybrid}",
        chunker,
        params or {"chunk_size": 512, "overlap": 64},
        embedder,
        k,
        hybrid,
    )


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_snippet_mentions_params_and_model(fw):
    code = snippet(_combo(), fw)
    assert "512" in code and "64" in code and "text-embedding-3-small" in code and "5" in code


def test_langchain_recursive_and_hybrid():
    code = snippet(_combo(hybrid=True), "langchain")
    assert "RecursiveCharacterTextSplitter" in code and "EnsembleRetriever" in code
    assert "BM25" in code


def test_llamaindex_sentence_window():
    code = snippet(
        _combo("sentence_window", {"window": 2}, "gemini:gemini-embedding-001"), "llamaindex"
    )
    assert "SentenceWindowNodeParser" in code and "window_size=2" in code
    assert "gemini-embedding-001" in code


def test_python_snippet_uses_chunklab_api():
    code = snippet(
        _combo("markdown", {"chunk_size": 800}, "local:all-MiniLM-L6-v2", hybrid=True), "python"
    )
    assert 'build_chunker("markdown", chunk_size=800)' in code
    assert 'build_embedder("local:all-MiniLM-L6-v2")' in code and "hybrid=True" in code


def test_langchain_sentence_window_notes_no_equivalent():
    code = snippet(_combo("sentence_window", {"window": 1}), "langchain")
    assert "no direct LangChain equivalent" in code


def test_unknown_framework():
    with pytest.raises(KeyError, match="unknown framework"):
        snippet(_combo(), "haystack")
