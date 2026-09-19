import time
from pathlib import Path

from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"
FORM = {
    "chunker_markdown": "on",
    "markdown_chunk_size": "512",
    "chunker_recursive": "on",
    "recursive_chunk_size": "128",
    "recursive_overlap": "0",
    "embedders": "fake",
    "top_k": "3",
    "hybrid": "both",
    "hit_threshold": "0.5",
}


def _seed(client: TestClient):
    with SAMPLE.open("rb") as f:
        client.post("/documents", files=[("files", ("sample.md", f, "text/markdown"))])
    doc = client.app.state.store.get_document("sample")
    s = doc.text.index("Customers")
    e = doc.text.index("Digital")
    client.post(
        "/questions", data={"text": "refund 30 days", "doc_id": "sample", "start": s, "end": e}
    )


def _await_finish(client: TestClient, rid: str) -> None:
    """Let the run thread finish before the fixture closes the store."""
    for _ in range(300):
        if client.app.state.runs.status(rid).state != "running":
            return
        time.sleep(0.02)
    raise AssertionError("never finished")


def test_experiment_page_lists_registry_and_counts(client: TestClient):
    _seed(client)
    r = client.get("/experiment")
    assert r.status_code == 200
    for name in ("recursive", "sentence_window", "markdown", "openai", "gemini", "local", "fake"):
        assert name in r.text
    assert "1 question" in r.text
    assert r.text.count('hx-post="/experiment/run"') == 1
    assert r.text.count("btn-export") == 1
    assert 'id="export-form"' in r.text and 'action="/experiment/export"' in r.text


def test_run_without_questions_is_rejected(client: TestClient):
    r = client.post("/experiment/run", data=FORM)
    assert r.status_code == 400 and "question" in r.text


def test_run_polls_to_done_and_redirects(client: TestClient):
    _seed(client)
    r = client.post("/experiment/run", data=FORM)
    assert r.status_code == 200 and "/experiment/status/" in r.text
    rid = r.text.split("/experiment/status/")[1].split('"')[0]
    for _ in range(200):
        s = client.get(f"/experiment/status/{rid}")
        if s.headers.get("HX-Redirect"):
            assert s.headers["HX-Redirect"] == f"/results/{rid}"
            break
        time.sleep(0.02)
    else:
        raise AssertionError("never finished")
    assert client.app.state.store.get_run(rid).status == "done"
    assert "4 / 4" in s.text or "4/4" in s.text


def test_bad_form_is_400(client: TestClient):
    _seed(client)
    r = client.post("/experiment/run", data={**FORM, "top_k": "five"})
    assert r.status_code == 400 and "integer" in r.text


def test_run_status_wrapper_is_not_nested(client: TestClient):
    _seed(client)
    page = client.get("/experiment")
    assert page.text.count('id="run-status"') == 1
    r = client.post("/experiment/run", data=FORM)
    assert r.status_code == 200
    assert r.text.count('id="run-status"') == 0
    assert 'id="run-status-inner"' in r.text
    rid = r.text.split("/experiment/status/")[1].split('"')[0]
    _await_finish(client, rid)


def test_unknown_hybrid_mode_is_400_not_500(client: TestClient):
    _seed(client)
    r = client.post("/experiment/run", data={**FORM, "hybrid": "bogus"})
    assert r.status_code == 400 and "unknown hybrid mode" in r.text


def test_export_error_escapes_user_input(client: TestClient):
    r = client.post(
        "/experiment/export",
        data={**FORM, "recursive_chunk_size": "<img src=x onerror=alert(1)>"},
    )
    assert r.status_code == 400
    assert "&lt;img" in r.text and "<img" not in r.text


def test_export_yaml(client: TestClient):
    r = client.post("/experiment/export", data=FORM)
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('filename="exp.yaml"')
    assert "name: markdown" in r.text and "questions: questions.json" in r.text
