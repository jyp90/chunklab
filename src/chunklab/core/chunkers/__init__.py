from __future__ import annotations

from chunklab.core.chunkers.base import Chunker
from chunklab.core.chunkers.markdown import MarkdownChunker
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.chunkers.sentence_window import SentenceWindowChunker

CHUNKERS: dict[str, type] = {
    "recursive": RecursiveChunker,
    "sentence_window": SentenceWindowChunker,
    "markdown": MarkdownChunker,
}


def build_chunker(name: str, **params) -> Chunker:
    try:
        cls = CHUNKERS[name]
    except KeyError:
        raise KeyError(f"unknown chunker '{name}'; available: {sorted(CHUNKERS)}") from None
    return cls(**params)


__all__ = ["CHUNKERS", "Chunker", "build_chunker"]
