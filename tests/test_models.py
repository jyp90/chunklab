import pytest

from chunklab.core.models import Chunk, Document, Question, RetrievedChunk, Span


def test_span_length_and_overlap():
    a = Span("d", 0, 10)
    b = Span("d", 5, 20)
    assert a.length == 10
    assert a.overlap(b) == 5
    assert b.overlap(a) == 5


def test_span_no_overlap_different_doc():
    assert Span("d1", 0, 10).overlap(Span("d2", 0, 10)) == 0


def test_span_no_overlap_disjoint():
    assert Span("d", 0, 10).overlap(Span("d", 10, 20)) == 0


def test_span_rejects_empty_or_negative():
    with pytest.raises(ValueError):
        Span("d", 5, 5)
    with pytest.raises(ValueError):
        Span("d", -1, 5)


def test_chunk_span_property():
    c = Chunk("d", 3, 8, "hello")
    assert c.span == Span("d", 3, 8)


def test_chunk_is_hashable_with_metadata():
    c = Chunk("d", 0, 5, "hello", {"heading_path": ("A",)})
    assert hash(c) == hash(Chunk("d", 0, 5, "hello"))


def test_question_and_retrieved_chunk_construct():
    q = Question("q1", "what?", (Span("d", 0, 5),))
    r = RetrievedChunk(Chunk("d", 0, 5, "hello"), 0.9, 1)
    assert q.spans[0].doc_id == "d"
    assert r.rank == 1


def test_document_defaults():
    d = Document("d", "text")
    assert d.source == ""
