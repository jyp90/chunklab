from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.markdown import MarkdownChunker
from chunklab.core.models import Document
from chunklab.core.text import load_document


def test_sections_by_heading_with_paths(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = MarkdownChunker(chunk_size=2000).chunk(doc)
    assert_chunks_valid(doc, chunks)
    paths = [c.metadata["heading_path"] for c in chunks]
    assert paths == [
        ("Refund Policy",),
        ("Refund Policy", "Exceptions"),
        ("Shipping",),
        ("Shipping", "Lost Packages"),
    ]
    assert chunks[0].text.startswith("# Refund Policy")
    assert chunks[1].text.startswith("## Exceptions")


def test_preamble_before_first_heading_has_empty_path():
    doc = Document("d", "intro text\n\n# H1\n\nbody\n")
    chunks = MarkdownChunker().chunk(doc)
    assert chunks[0].text == "intro text"
    assert chunks[0].metadata["heading_path"] == ()
    assert chunks[1].metadata["heading_path"] == ("H1",)


def test_large_section_is_subdivided_with_correct_offsets():
    body = " ".join(f"word{i}" for i in range(200))
    doc = Document("d", f"# Big\n\n{body}\n")
    chunks = MarkdownChunker(chunk_size=200).chunk(doc)
    assert len(chunks) > 3
    assert_chunks_valid(doc, chunks)
    assert all(c.metadata["heading_path"] == ("Big",) for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_no_headings_falls_back_to_whole_doc():
    doc = Document("d", "plain text only\n")
    chunks = MarkdownChunker().chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ()
