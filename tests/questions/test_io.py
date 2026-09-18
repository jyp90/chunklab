import json
from pathlib import Path

from chunklab.core.models import Question, Span
from chunklab.core.questions.io import load_questions, save_questions


def test_roundtrip(tmp_path: Path):
    qs = [
        Question("q1", "what?", (Span("d", 0, 10),)),
        Question("q2", "why?", (Span("d", 5, 15), Span("e", 0, 3))),
    ]
    p = tmp_path / "q.json"
    save_questions(qs, p)
    data = json.loads(p.read_text())
    assert data["version"] == 1
    assert data["questions"][1]["spans"][1] == {"doc_id": "e", "start": 0, "end": 3}
    assert load_questions(p) == qs
