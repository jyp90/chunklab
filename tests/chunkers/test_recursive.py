from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.models import Document
from chunklab.core.text import load_document


def _covers_all_non_whitespace(doc: Document, chunks) -> bool:
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    return all(i in covered for i, ch in enumerate(doc.text) if not ch.isspace())


def test_short_doc_is_single_chunk():
    doc = Document("d", "short text\n")
    chunks = RecursiveChunker(chunk_size=100).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_offsets_invariant_and_size_bound(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunker = RecursiveChunker(chunk_size=120, overlap=0)
    chunks = chunker.chunk(doc)
    assert_chunks_valid(doc, chunks)
    assert len(chunks) > 3
    assert all(len(c.text) <= 120 for c in chunks)
    assert _covers_all_non_whitespace(doc, chunks)


def test_chunks_are_ordered_and_non_overlapping_without_overlap(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = RecursiveChunker(chunk_size=150).chunk(doc)
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.end <= cur.start


def test_overlap_extends_start_backwards(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    base = RecursiveChunker(chunk_size=150, overlap=0).chunk(doc)
    over = RecursiveChunker(chunk_size=150, overlap=30).chunk(doc)
    assert_chunks_valid(doc, over)
    assert len(base) == len(over)
    assert over[0].start == base[0].start
    for b, o in zip(base[1:], over[1:], strict=True):
        assert o.start <= b.start
        assert b.start - o.start <= 30
        assert o.end == b.end


def test_prefers_paragraph_boundaries():
    doc = Document("d", "para one is here.\n\npara two is here.\n\npara three is here.\n")
    # each paragraph incl. separator is 19-20 chars; 20 forbids merging neighbours
    chunks = RecursiveChunker(chunk_size=20).chunk(doc)
    assert [c.text for c in chunks] == [
        "para one is here.",
        "para two is here.",
        "para three is here.",
    ]


def test_hard_split_when_no_separator():
    doc = Document("d", "x" * 25)
    chunks = RecursiveChunker(chunk_size=10).chunk(doc)
    assert [c.text for c in chunks] == ["x" * 10, "x" * 10, "x" * 5]
    assert_chunks_valid(doc, chunks)
