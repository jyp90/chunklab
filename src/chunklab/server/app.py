from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from chunklab.server import (
    routes_documents,
    routes_experiment,
    routes_questions,
    routes_results,
)
from chunklab.server.runs import RunManager
from chunklab.server.security import LocalOnlyMiddleware
from chunklab.server.store import Store

_HERE = Path(__file__).parent


def create_app(workspace: Path, allowed_hosts: set[str] | None = None) -> FastAPI:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app.state.store.close()

    app = FastAPI(title="chunklab", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.workspace = workspace
    app.state.store = Store(workspace / "chunklab.db")
    app.state.runs = RunManager(app.state.store, workspace / ".chunklab-cache.db")
    app.state.templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app.add_middleware(LocalOnlyMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    app.include_router(routes_documents.router)
    app.include_router(routes_questions.router)
    app.include_router(routes_experiment.router)
    app.include_router(routes_results.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "workspace": str(workspace)}

    return app
