import json
from pathlib import Path

from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def _upload(client: TestClient):
    with SAMPLE.open("rb") as f:
        client.post("/documents", files=[("files", ("sample.md", f, "text/markdown"))])
    return client.app.state.store.get_document("sample")


def test_add_edit_respan_delete_question(client: TestClient):
    doc = _upload(client)
    s = doc.text.index("Customers may request")
    e = doc.text.index("Digital goods")
    r = client.post(
        "/questions", data={"text": "refund window?", "doc_id": "sample", "start": s, "end": e}
    )
    assert r.status_code == 200 and "refund window?" in r.text
    q = client.app.state.store.list_questions()[0]
    assert q.spans == ((doc.id, s, e),) or (
        q.spans[0].doc_id,
        q.spans[0].start,
        q.spans[0].end,
    ) == ("sample", s, e)

    r = client.post(f"/questions/{q.id}/text", data={"text": "how long for a refund?"})
    assert "how long for a refund?" in r.text
    r = client.post(
        f"/questions/{q.id}/span", data={"doc_id": "sample", "start": s + 1, "end": e - 1}
    )
    assert client.app.state.store.get_question(q.id).spans[0].start == s + 1
    r = client.delete(f"/questions/{q.id}")
    assert r.status_code == 200 and client.app.state.store.list_questions() == []


def test_invalid_span_rejected(client: TestClient):
    _upload(client)
    r = client.post("/questions", data={"text": "x?", "doc_id": "sample", "start": 10, "end": 5})
    assert r.status_code == 400 and "span" in r.text
    r = client.post("/questions", data={"text": "x?", "doc_id": "nope", "start": 0, "end": 5})
    assert r.status_code == 400 and "nope" in r.text
    r = client.post(
        "/questions", data={"text": "x?", "doc_id": "sample", "start": 0, "end": 10_000}
    )
    assert r.status_code == 400


def test_generate_from_selection_with_fake_llm(client: TestClient):
    doc = _upload(client)
    s = doc.text.index("Defective products")
    e = doc.text.index("start a claim.") + len("start a claim.")
    r = client.post(
        "/questions/generate-from-selection",
        data={"doc_id": "sample", "start": s, "end": e, "llm": "fake"},
    )
    assert r.status_code == 200
    qs = client.app.state.store.list_questions()
    assert len(qs) == 1 and qs[0].text == "What is described in this passage?"
    assert (qs[0].spans[0].start, qs[0].spans[0].end) == (s, e)


def test_generate_missing_key_is_400(client: TestClient, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _upload(client)
    r = client.post(
        "/questions/generate-from-selection",
        data={"doc_id": "sample", "start": 0, "end": 20, "llm": "openai"},
    )
    assert r.status_code == 400 and "OPENAI_API_KEY" in r.text


def test_auto_generate_and_export_import(client: TestClient, tmp_path: Path):
    _upload(client)
    r = client.post(
        "/questions/auto-generate",
        data={"per_doc": 2, "llm": "fake", "min_len": 50, "max_len": 600},
    )
    assert r.status_code == 200
    assert len(client.app.state.store.list_questions()) == 2

    r = client.get("/questions/export")
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith("attachment")
    data = json.loads(r.text)
    assert data["version"] == 1 and len(data["questions"]) == 2

    for q in client.app.state.store.list_questions():
        client.delete(f"/questions/{q.id}")
    data["questions"].append(
        {"id": "ghost", "text": "?", "spans": [{"doc_id": "missing", "start": 0, "end": 3}]}
    )
    r = client.post(
        "/questions/import",
        files=[("file", ("questions.json", json.dumps(data).encode(), "application/json"))],
    )
    assert r.status_code == 200
    assert len(client.app.state.store.list_questions()) == 2
    assert "skipped 1" in r.text


def test_panel_lists_questions(client: TestClient):
    _upload(client)
    client.post("/questions", data={"text": "hello?", "doc_id": "sample", "start": 0, "end": 5})
    r = client.get("/questions/panel")
    assert "hello?" in r.text and 'hx-get="/documents/sample/view?q=' in r.text


def test_edit_text_rejects_empty(client: TestClient):
    _upload(client)
    client.post("/questions", data={"text": "keep me?", "doc_id": "sample", "start": 0, "end": 5})
    q = client.app.state.store.list_questions()[0]
    r = client.post(f"/questions/{q.id}/text", data={"text": "   "})
    assert r.status_code == 400 and "empty" in r.text
    assert client.app.state.store.get_question(q.id).text == "keep me?"


class _BoomLLM:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("upstream exploded")


def test_llm_failure_is_400_not_500(client: TestClient, monkeypatch):
    doc = _upload(client)
    monkeypatch.setattr(
        "chunklab.server.routes_questions.build_llm", lambda spec: _BoomLLM(), raising=True
    )
    r = client.post(
        "/questions/generate-from-selection",
        data={"doc_id": "sample", "start": 0, "end": 20, "llm": "fake"},
    )
    assert r.status_code == 400 and "RuntimeError: upstream exploded" in r.text
    assert client.app.state.store.list_questions() == []

    r = client.post("/questions/auto-generate", data={"per_doc": 1, "llm": "fake", "min_len": 50})
    assert r.status_code == 400 and "RuntimeError: upstream exploded" in r.text
    assert doc is not None


def test_auto_generate_without_documents_is_an_error(client: TestClient):
    r = client.post("/questions/auto-generate", data={"per_doc": 2, "llm": "fake"})
    assert r.status_code == 400
    assert '<div class="error">' in r.text and "generated 0 question(s)" in r.text


def test_import_rejects_id_with_trailing_newline(client: TestClient):
    doc = _upload(client)
    payload = {
        "version": 1,
        "questions": [
            {"id": "q\n", "text": "sneaky?", "spans": [{"doc_id": doc.id, "start": 0, "end": 5}]},
            {"id": "ok-1", "text": "fine?", "spans": [{"doc_id": doc.id, "start": 0, "end": 5}]},
        ],
    }
    r = client.post(
        "/questions/import",
        files=[("file", ("questions.json", json.dumps(payload).encode(), "application/json"))],
    )
    assert r.status_code == 200 and "imported 1, skipped 1" in r.text
    assert [q.id for q in client.app.state.store.list_questions()] == ["ok-1"]
