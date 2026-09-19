from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chunklab.server.app import create_app


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def client(workspace: Path) -> TestClient:
    # TestClient sends `Host: testserver`; the app itself only allows localhost.
    app = create_app(workspace, allowed_hosts={"127.0.0.1", "localhost", "testserver"})
    with TestClient(app) as c:
        yield c
