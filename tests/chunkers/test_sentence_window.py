from pathlib import Path

from chunklab.core.chunkers.base import assert_chunks_valid
from chunklab.core.chunkers.sentence_window import SentenceWindowChunker, split_sentences
from chunklab.core.models import Document
from chunklab.core.text import load_document


def test_split_sentences_offsets():
    text = "First one. Second one!  Third?\n"
    sents = split_sentences(text)
    assert [text[s:e] for s, e in sents] == ["First one.", "Second one!", "Third?"]


def test_split_sentences_treats_blank_line_as_boundary():
    text = "no period here\n\nnext para\n"
    sents = split_sentences(text)
    assert [text[s:e] for s, e in sents] == ["no period here", "next para"]


def test_window_zero_is_one_chunk_per_sentence():
    doc = Document("d", "A one. B two. C three.\n")
    chunks = SentenceWindowChunker(window=0).chunk(doc)
    assert [c.text for c in chunks] == ["A one.", "B two.", "C three."]
    assert [c.metadata["center"] for c in chunks] == [0, 1, 2]


def test_window_one_spans_neighbors():
    doc = Document("d", "A one. B two. C three.\n")
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    assert [c.text for c in chunks] == [
        "A one. B two.",
        "A one. B two. C three.",
        "B two. C three.",
    ]
    assert_chunks_valid(doc, chunks)


def test_sample_doc_invariant(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    chunks = SentenceWindowChunker(window=2).chunk(doc)
    assert_chunks_valid(doc, chunks)
    assert len(chunks) == len(split_sentences(doc.text))
