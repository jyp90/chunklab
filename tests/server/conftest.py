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
    app = create_app(workspace)
    with TestClient(app) as c:
        yield c
