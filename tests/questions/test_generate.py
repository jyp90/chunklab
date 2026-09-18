from pathlib import Path

from chunklab.core.models import Document
from chunklab.core.questions.generate import QUESTION_PROMPT, generate_questions, sample_passages
from chunklab.core.questions.llm import FakeLLM
from chunklab.core.text import load_document


def _doc_with_paragraphs(n: int, size: int) -> Document:
    paras = [f"Paragraph {i}. " + ("word " * (size // 5)) for i in range(n)]
    return Document("d", "\n\n".join(paras) + "\n")


def test_sample_passages_respects_min_len_and_n():
    doc = _doc_with_paragraphs(10, 300)
    spans = sample_passages(doc, n=3, seed=1, min_len=200, max_len=600)
    assert len(spans) == 3
    for s in spans:
        assert s.doc_id == "d"
        assert 200 <= s.length <= 600
        assert doc.text[s.start : s.end].startswith("Paragraph")


def test_sample_passages_deterministic_by_seed():
    doc = _doc_with_paragraphs(10, 300)
    assert sample_passages(doc, 4, seed=7) == sample_passages(doc, 4, seed=7)
    assert sample_passages(doc, 4, seed=7) != sample_passages(doc, 4, seed=8)


def test_sample_passages_truncates_to_max_len():
    doc = _doc_with_paragraphs(2, 2000)
    spans = sample_passages(doc, 2, min_len=100, max_len=500)
    assert all(s.length == 500 for s in spans)


def test_sample_passages_returns_fewer_when_short_doc():
    doc = Document("d", "tiny\n\nalso tiny\n")
    assert sample_passages(doc, 5, min_len=200) == []


def test_generate_questions_builds_questions_with_spans(sample_doc_path: Path):
    doc = load_document(sample_doc_path)
    llm = FakeLLM(reply="What is the refund window?\nignored second line")
    qs = generate_questions([doc], llm, per_doc=2, seed=0, min_len=50)
    assert len(qs) == 2
    assert qs[0].id == "sample-q0"
    assert qs[0].text == "What is the refund window?"
    assert len(qs[0].spans) == 1
    assert qs[0].spans[0].doc_id == "sample"
    assert len(llm.prompts) == 2
    assert QUESTION_PROMPT.split("{passage}")[0] in llm.prompts[0]
    passage = doc.text[qs[0].spans[0].start : qs[0].spans[0].end]
    assert passage in llm.prompts[0]


def test_generate_questions_warns_when_doc_yields_too_few_passages():
    doc = Document("tiny", "short one\n\nshort two\n")
    warnings: list[str] = []
    qs = generate_questions([doc], FakeLLM(), per_doc=3, min_len=200, warn=warnings.append)
    assert qs == []
    assert len(warnings) == 1
    assert warnings[0].startswith("tiny: only 0 paragraph(s) >= 200 chars (requested 3)")
    assert "--min-len" in warnings[0]


def test_generate_questions_warns_when_fewer_than_requested():
    doc = _doc_with_paragraphs(2, 300)
    warnings: list[str] = []
    generate_questions([doc], FakeLLM(), per_doc=5, min_len=200, warn=warnings.append)
    assert len(warnings) == 1
    assert "only 2 paragraph(s)" in warnings[0]


def test_generate_questions_does_not_warn_when_enough_passages():
    doc = _doc_with_paragraphs(5, 300)
    warnings: list[str] = []
    generate_questions([doc], FakeLLM(), per_doc=3, min_len=200, warn=warnings.append)
    assert warnings == []


def test_generate_questions_skips_empty_reply():
    doc = _doc_with_paragraphs(3, 300)
    qs = generate_questions([doc], FakeLLM(reply="   \n"), per_doc=3)
    assert qs == []
