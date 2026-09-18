from chunklab.core.chunkers.base import assert_chunks_valid, make_chunk
from chunklab.core.models import Chunk, Document


def test_make_chunk_trims_whitespace_and_keeps_offsets():
    doc = Document("d", "  hello world  \n")
    c = make_chunk(doc, 0, len(doc.text))
    assert c is not None
    assert (c.start, c.end) == (2, 13)
    assert c.text == "hello world"
    assert doc.text[c.start : c.end] == c.text


def test_make_chunk_returns_none_for_whitespace_only():
    doc = Document("d", "   \n")
    assert make_chunk(doc, 0, 4) is None


def test_make_chunk_passes_metadata():
    doc = Document("d", "abc")
    c = make_chunk(doc, 0, 3, heading_path=("H",))
    assert c is not None
    assert c.metadata == {"heading_path": ("H",)}


def test_assert_chunks_valid_detects_offset_mismatch():
    doc = Document("d", "abcdef")
    bad = [Chunk("d", 0, 3, "xyz")]
    try:
        assert_chunks_valid(doc, bad)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError")
