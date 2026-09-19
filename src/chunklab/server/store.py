from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chunklab.core.models import Document, Question, Span
from chunklab.core.runner.run import ComboResult, RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL, text TEXT NOT NULL,
  content_hash TEXT
);
CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY, text TEXT NOT NULL, spans_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
  config_yaml TEXT NOT NULL, result_json TEXT, error TEXT
);
"""


@dataclass(frozen=True)
class DocumentRow:
    id: str
    name: str
    source: str
    n_chars: int
    n_questions: int


@dataclass
class RunRow:
    run_id: str
    created_at: str
    status: str
    config_yaml: str
    result: RunResult | None
    error: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _result_from_dict(data: dict) -> RunResult:
    from dataclasses import fields

    known = {f.name for f in fields(ComboResult)}
    return RunResult(
        run_id=data["run_id"],
        created_at=data["created_at"],
        config=data["config"],
        combos=[ComboResult(**{k: v for k, v in c.items() if k in known}) for c in data["combos"]],
    )


class Store:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            try:  # a database created before content_hash existed
                self._conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            except sqlite3.OperationalError:
                pass  # the column is already there
            # A run only lives in the process that started it, so anything still
            # marked `running` when we open the DB was killed with the server.
            self._conn.execute(
                "UPDATE runs SET status = 'error', error = ? WHERE status = 'running'",
                ("interrupted (server restarted)",),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- documents -------------------------------------------------------
    def upsert_document(self, doc: Document, name: str | None = None) -> None:
        digest = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents (id, name, source, text, content_hash)"
                " VALUES (?, ?, ?, ?, ?)",
                (doc.id, name or doc.id, doc.source, doc.text, digest),
            )
            self._conn.commit()

    def has_document(self, doc_id: str) -> bool:
        """Whether a row exists — distinct from ``content_hash`` returning ``None``,
        which a row written before the ``content_hash`` column existed also does."""
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return row is not None

    def content_hash(self, doc_id: str) -> str | None:
        """sha256 of the stored text, or ``None`` if the document is unknown or
        predates the ``content_hash`` column."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row else None

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, source FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return Document(id=row[0], text=row[1], source=row[2]) if row else None

    def list_documents(self) -> list[DocumentRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, source, length(text) FROM documents ORDER BY name"
            ).fetchall()
            counts: dict[str, int] = {}
            for (spans_json,) in self._conn.execute("SELECT spans_json FROM questions"):
                doc_ids = {s["doc_id"] for s in json.loads(spans_json)}
                for doc_id in doc_ids:
                    counts[doc_id] = counts.get(doc_id, 0) + 1
        return [DocumentRow(r[0], r[1], r[2], r[3], counts.get(r[0], 0)) for r in rows]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            victims = [
                qid
                for qid, spans_json in self._conn.execute("SELECT id, spans_json FROM questions")
                if any(s["doc_id"] == doc_id for s in json.loads(spans_json))
            ]
            self._conn.executemany("DELETE FROM questions WHERE id = ?", [(q,) for q in victims])
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        return len(victims)

    # ---- questions -------------------------------------------------------
    def upsert_question(self, q: Question) -> None:
        spans = json.dumps([{"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in q.spans])
        with self._lock:
            existing = self._conn.execute(
                "SELECT created_at FROM questions WHERE id = ?", (q.id,)
            ).fetchone()
            created = existing[0] if existing else _now()
            self._conn.execute(
                "INSERT OR REPLACE INTO questions (id, text, spans_json, created_at)"
                " VALUES (?, ?, ?, ?)",
                (q.id, q.text, spans, created),
            )
            self._conn.commit()

    @staticmethod
    def _question(row) -> Question:
        return Question(
            row[0],
            row[1],
            tuple(Span(s["doc_id"], s["start"], s["end"]) for s in json.loads(row[2])),
        )

    def get_question(self, qid: str) -> Question | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, text, spans_json FROM questions WHERE id = ?", (qid,)
            ).fetchone()
        return self._question(row) if row else None

    def list_questions(self) -> list[Question]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, spans_json FROM questions ORDER BY created_at, id"
            ).fetchall()
        return [self._question(r) for r in rows]

    def delete_question(self, qid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM questions WHERE id = ?", (qid,))
            self._conn.commit()

    # ---- runs ------------------------------------------------------------
    def save_run(
        self,
        run_id: str,
        status: str,
        config_yaml: str,
        result: RunResult | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            existing = self._conn.execute(
                "SELECT created_at FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            created = existing[0] if existing else _now()
            self._conn.execute(
                "INSERT OR REPLACE INTO runs"
                " (run_id, created_at, status, config_yaml, result_json, error)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    created,
                    status,
                    config_yaml,
                    json.dumps(result.to_dict()) if result else None,
                    error,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _run(row) -> RunRow:
        return RunRow(
            run_id=row[0],
            created_at=row[1],
            status=row[2],
            config_yaml=row[3],
            result=_result_from_dict(json.loads(row[4])) if row[4] else None,
            error=row[5],
        )

    def get_run(self, run_id: str) -> RunRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, created_at, status, config_yaml, result_json, error FROM runs"
                " WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._run(row) if row else None

    def list_runs(self) -> list[RunRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_at, status, config_yaml, result_json, error FROM runs"
                " ORDER BY created_at DESC, run_id DESC"
            ).fetchall()
        return [self._run(r) for r in rows]
