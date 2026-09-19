from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from chunklab.core.chunkers import CHUNKERS
from chunklab.server.forms import config_from_form, config_to_yaml

router = APIRouter()
EMBEDDER_CHOICES = [
    "fake",
    "openai:text-embedding-3-small",
    "openai:text-embedding-3-large",
    "gemini:gemini-embedding-001",
    "local:all-MiniLM-L6-v2",
]


def _ctx(request: Request, **extra) -> dict:
    store = request.app.state.store
    return {
        "title": "Experiment",
        "active": "experiment",
        "chunkers": sorted(CHUNKERS),
        "embedder_choices": EMBEDDER_CHOICES,
        "n_docs": len(store.list_documents()),
        "n_questions": len(store.list_questions()),
        "error": None,
        **extra,
    }


def _multi(form) -> dict:
    out: dict[str, str | list[str]] = {}
    for k, v in form.multi_items():
        if k in out:
            cur = out[k]
            out[k] = (cur if isinstance(cur, list) else [cur]) + [v]
        else:
            out[k] = v
    return out


@router.get("/experiment", response_class=HTMLResponse)
def experiment_page(request: Request):
    return request.app.state.templates.TemplateResponse(request, "experiment.html", _ctx(request))


@router.post("/experiment/run", response_class=HTMLResponse)
async def start_run(request: Request):
    form = await request.form()
    store = request.app.state.store
    questions = store.list_questions()
    docs = [store.get_document(d.id) for d in store.list_documents()]
    tpl = request.app.state.templates
    if not questions:
        return tpl.TemplateResponse(
            request,
            "experiment.html",
            _ctx(request, error="add at least one question first"),
            status_code=400,
        )
    try:
        cfg = config_from_form(_multi(form))
        rid = request.app.state.runs.start(cfg, docs, questions)
    except ValueError as e:
        return tpl.TemplateResponse(
            request, "experiment.html", _ctx(request, error=str(e)), status_code=400
        )
    return tpl.TemplateResponse(
        request, "partials/run_status.html", {"st": request.app.state.runs.status(rid)}
    )


@router.get("/experiment/status/{run_id}", response_class=HTMLResponse)
def run_status(request: Request, run_id: str):
    st = request.app.state.runs.status(run_id)
    if st is None:
        return HTMLResponse("<p class='error'>unknown run</p>", status_code=404)
    resp = request.app.state.templates.TemplateResponse(
        request, "partials/run_status.html", {"st": st}
    )
    if st.state == "done":
        resp.headers["HX-Redirect"] = f"/results/{run_id}"
    return resp


@router.post("/experiment/export")
async def export_yaml(request: Request):
    form = await request.form()
    try:
        cfg = config_from_form(_multi(form))
    except ValueError as e:
        return HTMLResponse(f"<p class='error'>{e}</p>", status_code=400)
    return Response(
        config_to_yaml(cfg),
        media_type="application/x-yaml",
        headers={"Content-Disposition": 'attachment; filename="exp.yaml"'},
    )
