from pathlib import Path

from chunklab.core.models import Document, Question, Span
from chunklab.core.runner.run import ComboResult, RunResult
from chunklab.server.store import Store


def _doc(i="a", text="hello world\n"):
    return Document(id=i, text=text, source=f"/x/{i}.md")


def test_document_roundtrip_and_list(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        s.upsert_document(_doc("b", "bbb\n"))
        s.upsert_document(_doc("a"))
        assert s.get_document("a") == _doc("a")
        rows = s.list_documents()
        assert [r.id for r in rows] == ["a", "b"]
        assert rows[0].n_chars == len("hello world\n")
        assert rows[0].n_questions == 0


def test_upsert_document_replaces_text(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        s.upsert_document(_doc("a", "v1\n"))
        s.upsert_document(_doc("a", "v2\n"))
        assert s.get_document("a").text == "v2\n"
        assert len(s.list_documents()) == 1


def test_questions_roundtrip_and_counts(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        s.upsert_document(_doc("a"))
        q = Question("q1", "what?", (Span("a", 0, 5), Span("a", 6, 11)))
        s.upsert_question(q)
        assert s.get_question("q1") == q
        assert s.list_questions() == [q]
        assert s.list_documents()[0].n_questions == 1
        s.upsert_question(Question("q1", "edited?", q.spans))
        assert s.get_question("q1").text == "edited?"
        s.delete_question("q1")
        assert s.list_questions() == []


def test_delete_document_cascades_questions(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        s.upsert_document(_doc("a"))
        s.upsert_document(_doc("b"))
        s.upsert_question(Question("q1", "?", (Span("a", 0, 5),)))
        s.upsert_question(Question("q2", "?", (Span("b", 0, 3),)))
        assert s.delete_document("a") == 1
        assert [q.id for q in s.list_questions()] == ["q2"]
        assert s.get_document("a") is None


def test_runs_roundtrip(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        s.save_run("r1", "running", "documents: []\n")
        row = s.get_run("r1")
        assert row.status == "running" and row.result is None and row.created_at
        res = RunResult(
            "r1",
            "2026-01-01T00:00:00+00:00",
            {},
            [
                ComboResult(
                    "c|fake|k=5|hybrid=False",
                    "recursive",
                    {},
                    "fake",
                    5,
                    False,
                    {"hit@5": 1.0},
                    3,
                )
            ],
        )
        s.save_run("r1", "done", "documents: []\n", result=res)
        row = s.get_run("r1")
        assert row.status == "done"
        assert row.result.combos[0].metrics == {"hit@5": 1.0}
        s.save_run("r2", "error", "x", error="boom")
        assert [r.run_id for r in s.list_runs()] == ["r2", "r1"]
        assert s.list_runs()[0].error == "boom"


def test_missing_rows_return_none(tmp_path: Path):
    with Store(tmp_path / "c.db") as s:
        assert s.get_document("nope") is None
        assert s.get_question("nope") is None
        assert s.get_run("nope") is None
