from pathlib import Path

from fastapi.testclient import TestClient


def test_health_reports_workspace(client: TestClient, workspace: Path):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "workspace": str(workspace.resolve())}


def test_static_htmx_served(client: TestClient):
    r = client.get("/static/htmx.min.js")
    assert r.status_code == 200
    assert "htmx" in r.text[:2000]


def test_root_renders_base_layout(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "chunklab" in r.text
    assert 'src="/static/htmx.min.js"' in r.text
