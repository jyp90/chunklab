from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from chunklab.core.models import Question, Span

FORMAT_VERSION = 1


def save_questions(questions: Sequence[Question], path: Path) -> None:
    payload = {
        "version": FORMAT_VERSION,
        "questions": [
            {
                "id": q.id,
                "text": q.text,
                "spans": [{"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in q.spans],
            }
            for q in questions
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported questions file version: {data.get('version')}")
    return [
        Question(
            q["id"],
            q["text"],
            tuple(Span(s["doc_id"], s["start"], s["end"]) for s in q["spans"]),
        )
        for q in data["questions"]
    ]
