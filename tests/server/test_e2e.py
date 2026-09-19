"""Upload → label → auto-generate → run → results → snippet, through the HTTP surface only."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.md"


def test_full_flow(client: TestClient, workspace: Path):
    with SAMPLE.open("rb") as f:
        assert (
            client.post(
                "/documents", files=[("files", ("sample.md", f, "text/markdown"))]
            ).status_code
            == 200
        )
    assert client.get("/documents/sample/view").status_code == 200
    text = client.app.state.store.get_document("sample").text
    s = text.index("International shipping")
    e = text.index("7 to 14 days.") + len("7 to 14 days.")
    assert (
        client.post(
            "/questions",
            data={
                "text": "how long does international shipping take",
                "doc_id": "sample",
                "start": s,
                "end": e,
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/questions/auto-generate", data={"per_doc": 2, "llm": "fake", "min_len": 50}
        ).status_code
        == 200
    )
    assert len(client.app.state.store.list_questions()) == 3

    r = client.post(
        "/experiment/run",
        data={
            "chunker_recursive": "on",
            "recursive_chunk_size": "128,256",
            "recursive_overlap": "0",
            "chunker_sentence_window": "on",
            "sentence_window_window": "1",
            "chunker_markdown": "on",
            "markdown_chunk_size": "512",
            "embedders": "fake",
            "top_k": "3",
            "hybrid": "both",
        },
    )
    rid = r.text.split("/experiment/status/")[1].split('"')[0]
    for _ in range(500):
        if client.app.state.runs.status(rid).state != "running":
            break
        time.sleep(0.02)
    assert client.app.state.runs.status(rid).state == "done"

    page = client.get(f"/results/{rid}").text
    assert "Recommended" in page and page.count("|fake|k=3|") >= 8
    qid = client.app.state.store.list_questions()[0].id
    assert client.get(f"/results/{rid}/question/{qid}").status_code == 200
    rec_combo = page.split("<strong>")[1].split("</strong>")[0]
    code = client.get(
        f"/results/{rid}/snippet", params={"combo": rec_combo, "framework": "langchain"}
    ).text
    assert "chunklab recommendation" in code
    assert (workspace / "chunklab.db").exists() and (workspace / ".chunklab-cache.db").exists()
    exported = client.get("/questions/export").json()
    assert len(exported["questions"]) == 3
