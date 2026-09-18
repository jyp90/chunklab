from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence

from chunklab.core.models import Document, Question, Span
from chunklab.core.questions.llm import LLM

QUESTION_PROMPT = (
    "You are creating retrieval test data. Write ONE question that can be answered ONLY "
    "using the passage below. The question must be specific to the passage's content and "
    "must not mention 'the passage'. Output only the question, nothing else.\n\n"
    'Passage:\n"""\n{passage}\n"""'
)

_PARA_RE = re.compile(r"\n{2,}")


def _paragraph_spans(doc: Document) -> list[Span]:
    spans: list[Span] = []
    pos = 0
    text = doc.text
    for m in _PARA_RE.finditer(text):
        if m.start() > pos:
            spans.append(Span(doc.id, pos, m.start()))
        pos = m.end()
    if pos < len(text):
        end = len(text.rstrip("\n"))
        if end > pos:
            spans.append(Span(doc.id, pos, end))
    return spans


def sample_passages(
    doc: Document, n: int, seed: int = 0, min_len: int = 200, max_len: int = 600
) -> list[Span]:
    candidates = [s for s in _paragraph_spans(doc) if s.length >= min_len]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = sorted(candidates[:n], key=lambda s: s.start)
    return [Span(s.doc_id, s.start, min(s.end, s.start + max_len)) for s in chosen]


def generate_questions(
    docs: Sequence[Document],
    llm: LLM,
    per_doc: int,
    seed: int = 0,
    min_len: int = 200,
    max_len: int = 600,
    warn: Callable[[str], None] | None = None,
) -> list[Question]:
    out: list[Question] = []
    for doc in docs:
        spans = sample_passages(doc, per_doc, seed=seed, min_len=min_len, max_len=max_len)
        if warn is not None and len(spans) < per_doc:
            warn(
                f"{doc.id}: only {len(spans)} paragraph(s) >= {min_len} chars "
                f"(requested {per_doc}); lower --min-len"
            )
        for i, span in enumerate(spans):
            passage = doc.text[span.start : span.end]
            reply = llm.complete(QUESTION_PROMPT.format(passage=passage))
            first_line = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
            if not first_line:
                continue
            out.append(Question(f"{doc.id}-q{i}", first_line, (span,)))
    return out
