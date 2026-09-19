from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from chunklab.core.runner import FRAMEWORKS, RunResult, recommend, snippet
from chunklab.server.highlight import render_highlighted
from chunklab.server.store import RunRow

router = APIRouter()

#: Same shape as document/question ids elsewhere: anything else cannot exist.
_ID_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")


def _tpl(request: Request, name: str, ctx: dict, status: int = 200) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request, name, ctx, status_code=status)


def _metric_keys(result: RunResult) -> list[str]:
    keys: list[str] = []
    for c in result.combos:
        for k in c.metrics:
            if k not in keys:
                keys.append(k)
    return keys


def _finished_run(request: Request, run_id: str) -> RunRow | None:
    if not _ID_RE.match(run_id):
        return None
    row = request.app.state.store.get_run(run_id)
    return row if row is not None and row.result is not None else None


@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    rows = request.app.state.store.list_runs()
    summary = []
    for r in rows:
        best = (
            max(
                (c.metrics.get(f"hit@{c.top_k}", 0.0) for c in r.result.combos if not c.error),
                default=None,
            )
            if r.result
            else None
        )
        summary.append({"row": r, "n": len(r.result.combos) if r.result else 0, "best_hit": best})
    return _tpl(request, "runs.html", {"title": "Runs", "active": "runs", "runs": summary})


@router.get("/results/{run_id}", response_class=HTMLResponse)
def results_page(request: Request, run_id: str, sort: str | None = None):
    row = _finished_run(request, run_id)
    if row is None or row.result is None:
        return HTMLResponse("<p class='error'>run not found or not finished</p>", status_code=404)
    result = row.result
    combos = list(result.combos)
    if sort:
        combos.sort(key=lambda c: (c.error is not None, -c.metrics.get(sort, float("-inf"))))
    return _tpl(
        request,
        "results.html",
        {
            "title": f"Results {run_id}",
            "active": "runs",
            "run": row,
            "result": result,
            "combos": combos,
            "metric_keys": _metric_keys(result),
            "rec": recommend(result),
            "sort": sort,
            "questions": request.app.state.store.list_questions(),
            "frameworks": FRAMEWORKS,
            "quote": quote,
        },
    )


@router.get("/results/{run_id}/question/{qid}", response_class=HTMLResponse)
def question_detail(request: Request, run_id: str, qid: str, combo: str | None = None):
    store = request.app.state.store
    row = _finished_run(request, run_id)
    q = store.get_question(qid) if _ID_RE.match(qid) else None
    if row is None or row.result is None or q is None:
        return HTMLResponse("<p class='error'>not found</p>", status_code=404)
    cards: list[dict[str, Any]] = []
    for c in row.result.combos:
        pq = next((x for x in c.per_question if x["id"] == qid), None)
        cards.append({"combo": c, "pq": pq})
    chosen = next((c for c in cards if c["combo"].combo_id == combo), cards[0] if cards else None)
    html = None
    doc_id = q.spans[0].doc_id
    doc = store.get_document(doc_id)
    if chosen and chosen["pq"] and doc:
        gold = [(s.start, s.end) for s in q.spans if s.doc_id == doc_id]
        hits = [(r["start"], r["end"]) for r in chosen["pq"]["retrieved"] if r["doc_id"] == doc_id]
        html = render_highlighted(doc.text, gold, hits)

    def preview(r: dict[str, Any]) -> str:
        d = store.get_document(r["doc_id"])
        return d.text[r["start"] : r["end"]][:160] if d else ""

    return _tpl(
        request,
        "partials/question_detail.html",
        {
            "run_id": run_id,
            "q": q,
            "cards": cards,
            "chosen": chosen,
            "html": html,
            "doc_id": doc_id,
            "preview": preview,
        },
    )


@router.get("/results/{run_id}/snippet", response_class=HTMLResponse)
def snippet_partial(request: Request, run_id: str, combo: str, framework: str = "python"):
    row = _finished_run(request, run_id)
    if row is None or row.result is None:
        return HTMLResponse("<p class='error'>run not found</p>", status_code=404)
    c = next((x for x in row.result.combos if x.combo_id == combo), None)
    if c is None:
        return HTMLResponse("<p class='error'>combo not found</p>", status_code=404)
    try:
        code = snippet(c, framework)
    except KeyError as e:
        return HTMLResponse(f"<p class='error'>{e.args[0]}</p>", status_code=400)
    return _tpl(
        request,
        "partials/snippet.html",
        {
            "run_id": run_id,
            "combo": c,
            "framework": framework,
            "frameworks": FRAMEWORKS,
            "code": code,
            "quote": quote,
        },
    )
