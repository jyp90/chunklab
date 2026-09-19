from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_HERE = Path(__file__).parent


def create_app(workspace: Path) -> FastAPI:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="chunklab", docs_url=None, redoc_url=None)
    app.state.workspace = workspace
    app.state.templates = Jinja2Templates(directory=str(_HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "workspace": str(workspace)}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return app.state.templates.TemplateResponse(
            request, "base.html", {"title": "chunklab", "active": "documents"}
        )

    return app
