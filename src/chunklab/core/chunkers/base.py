from __future__ import annotations

from typing import Protocol, runtime_checkable

from chunklab.core.models import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def make_chunk(doc: Document, start: int, end: int, **metadata) -> Chunk | None:
    text = doc.text
    end = min(end, len(text))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Chunk(doc.id, start, end, text[start:end], dict(metadata))


def assert_chunks_valid(doc: Document, chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.doc_id == doc.id, f"chunk doc_id {c.doc_id} != {doc.id}"
        assert 0 <= c.start < c.end <= len(doc.text), f"bad offsets {c.start}:{c.end}"
        assert doc.text[c.start : c.end] == c.text, f"text mismatch at {c.start}:{c.end}"
