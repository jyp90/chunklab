import time
from pathlib import Path

from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def _run(client: TestClient) -> str:
    with SAMPLE.open("rb") as f:
        client.post("/documents", files=[("files", ("sample.md", f, "text/markdown"))])
    doc = client.app.state.store.get_document("sample")
    s = doc.text.index("Customers")
    e = doc.text.index("Digital")
    client.post(
        "/questions",
        data={
            "text": "refund within 30 days of purchase",
            "doc_id": "sample",
            "start": s,
            "end": e,
        },
    )
    r = client.post(
        "/experiment/run",
        data={
            "chunker_markdown": "on",
            "markdown_chunk_size": "512",
            "chunker_recursive": "on",
            "recursive_chunk_size": "128",
            "recursive_overlap": "0",
            "embedders": "fake",
            "top_k": "3",
            "hybrid": "dense",
        },
    )
    rid = r.text.split("/experiment/status/")[1].split('"')[0]
    for _ in range(300):
        if client.app.state.runs.status(rid).state != "running":
            break
        time.sleep(0.02)
    return rid


def test_results_table_and_recommendation(client: TestClient):
    rid = _run(client)
    r = client.get(f"/results/{rid}")
    assert r.status_code == 200
    assert "markdown(chunk_size=512)|fake|k=3|hybrid=False" in r.text
    assert "hit@3" in r.text and "precision" in r.text
    assert 'class="badge"' in r.text and "Recommended" in r.text
    assert 'class="recommended"' in r.text or "recommended" in r.text


def test_results_layout_is_stacked_full_width(client: TestClient):
    rid = _run(client)
    r = client.get(f"/results/{rid}")
    assert r.status_code == 200
    assert 'class="col-narrow"' not in r.text
    detail_pos = r.text.index('id="question-detail"')
    snippet_pos = r.text.index('id="snippet"')
    assert detail_pos < snippet_pos


def test_results_sort_by_precision(client: TestClient):
    rid = _run(client)
    r = client.get(f"/results/{rid}?sort=precision")
    rows = [ln for ln in r.text.splitlines() if "|fake|k=3|" in ln and "<td" in ln]
    assert len(rows) == 2


def test_question_detail_highlights_gold_and_hits(client: TestClient):
    rid = _run(client)
    qid = client.app.state.store.list_questions()[0].id
    r = client.get(f"/results/{rid}/question/{qid}")
    assert r.status_code == 200
    assert 'class="gold"' in r.text or 'class="both"' in r.text
    assert "rank 1" in r.text.lower() or "#1" in r.text
    r2 = client.get(
        f"/results/{rid}/question/{qid}?combo=recursive(chunk_size=128,overlap=0)|fake|k=3|hybrid=False"
    )
    assert r2.status_code == 200 and "recursive(chunk_size=128" in r2.text


def test_snippet_partial(client: TestClient):
    rid = _run(client)
    cid = "markdown(chunk_size=512)|fake|k=3|hybrid=False"
    for fw, needle in (
        ("python", "build_chunker"),
        ("langchain", "MarkdownHeaderTextSplitter"),
        ("llamaindex", "MarkdownNodeParser"),
    ):
        r = client.get(f"/results/{rid}/snippet", params={"combo": cid, "framework": fw})
        assert r.status_code == 200 and needle in r.text


def test_snippet_error_escapes_user_input(client: TestClient):
    rid = _run(client)
    cid = "markdown(chunk_size=512)|fake|k=3|hybrid=False"
    r = client.get(
        f"/results/{rid}/snippet",
        params={"combo": cid, "framework": "<img src=x onerror=alert(1)>"},
    )
    assert r.status_code == 400
    assert "&lt;img" in r.text and "<img" not in r.text


def test_runs_history_and_404s(client: TestClient):
    rid = _run(client)
    r = client.get("/runs")
    assert rid in r.text and "done" in r.text
    assert client.get("/results/nope").status_code == 404
    assert client.get(f"/results/{rid}/question/nope").status_code == 404
    assert (
        client.get(
            f"/results/{rid}/snippet", params={"combo": "nope", "framework": "python"}
        ).status_code
        == 404
    )
