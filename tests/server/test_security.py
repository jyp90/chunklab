from pathlib import Path

from fastapi.testclient import TestClient

from chunklab.server.app import create_app


def test_unknown_host_is_rejected(client: TestClient):
    r = client.get("/health", headers={"Host": "evil.example"})
    assert r.status_code == 421 and r.text == "host not allowed"


def test_host_port_is_ignored(client: TestClient):
    assert client.get("/health", headers={"Host": "127.0.0.1:7860"}).status_code == 200


def test_default_allowed_hosts_exclude_testserver(workspace: Path):
    with TestClient(create_app(workspace)) as c:
        assert c.get("/health").status_code == 421


def test_cross_origin_post_is_rejected(client: TestClient):
    r = client.post("/questions/panel", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and r.text == "cross-origin request not allowed"


def test_cross_site_fetch_metadata_post_is_rejected(client: TestClient):
    r = client.post("/questions/panel", headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_origin_post_passes_through(client: TestClient):
    r = client.post(
        "/questions",
        data={"text": "q?", "doc_id": "nope", "start": 0, "end": 1},
        headers={"Sec-Fetch-Site": "same-origin"},
    )
    assert r.status_code == 400 and "unknown document" in r.text


def test_post_without_origin_or_fetch_metadata_passes_through(client: TestClient):
    r = client.post("/questions", data={"text": "q?", "doc_id": "nope", "start": 0, "end": 1})
    assert r.status_code == 400


def test_cross_origin_get_is_allowed(client: TestClient):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200


def test_same_origin_origin_header_passes(client: TestClient):
    r = client.post(
        "/questions",
        data={"text": "q?", "doc_id": "nope", "start": 0, "end": 1},
        headers={"Origin": "http://127.0.0.1:7860"},
    )
    assert r.status_code == 400
