from __future__ import annotations

import json
import re
import secrets
from typing import Annotated

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from chunklab.core.embedders import MissingApiKeyError
from chunklab.core.models import Question, Span
from chunklab.core.questions import QUESTION_PROMPT, build_llm, generate_questions
from chunklab.core.questions.io import FORMAT_VERSION
from chunklab.server.store import Store

router = APIRouter()

#: Same shape as document ids: what ``safe_stem`` produces plus generated suffixes.
#: ``\A``/``\Z`` rather than ``^``/``$`` so a trailing newline is rejected.
_ID_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")


def _panel(
    request: Request,
    error: str | None = None,
    notice: str | None = None,
    status: int = 200,
) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(
        request,
        "partials/question_list.html",
        {
            "questions": request.app.state.store.list_questions(),
            "error": error,
            "notice": notice,
        },
        status_code=status,
    )


def _validated_span(store: Store, doc_id: str, start: int, end: int) -> Span:
    """Return a ``Span`` for an existing document, or raise ``ValueError``."""
    if not _ID_RE.match(doc_id):
        raise ValueError(f"unknown document '{doc_id}'")
    doc = store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"unknown document '{doc_id}'")
    if not (0 <= start < end <= len(doc.text)):
        raise ValueError(f"invalid span {start}:{end} for document of {len(doc.text)} chars")
    return Span(doc_id, start, end)


def _new_id() -> str:
    return f"q-{secrets.token_hex(4)}"


def _lookup(request: Request, qid: str) -> Question | None:
    if not _ID_RE.match(qid):
        return None
    return request.app.state.store.get_question(qid)


@router.get("/questions/panel", response_class=HTMLResponse)
def panel(request: Request):
    return _panel(request)


@router.post("/questions", response_class=HTMLResponse)
def add_question(
    request: Request,
    text: Annotated[str, Form()],
    doc_id: Annotated[str, Form()],
    start: Annotated[int, Form()],
    end: Annotated[int, Form()],
):
    store = request.app.state.store
    try:
        span = _validated_span(store, doc_id, start, end)
    except ValueError as e:
        return _panel(request, error=str(e), status=400)
    if not text.strip():
        return _panel(request, error="question text is empty", status=400)
    store.upsert_question(Question(_new_id(), text.strip(), (span,)))
    return _panel(request)


@router.post("/questions/{qid}/text", response_class=HTMLResponse)
def edit_text(request: Request, qid: str, text: Annotated[str, Form()]):
    q = _lookup(request, qid)
    if q is None:
        return _panel(request, error="question not found", status=404)
    if not text.strip():
        return _panel(request, error="question text is empty", status=400)
    request.app.state.store.upsert_question(Question(q.id, text.strip(), q.spans))
    return _panel(request)


@router.post("/questions/{qid}/span", response_class=HTMLResponse)
def set_span(
    request: Request,
    qid: str,
    doc_id: Annotated[str, Form()],
    start: Annotated[int, Form()],
    end: Annotated[int, Form()],
):
    store = request.app.state.store
    q = _lookup(request, qid)
    if q is None:
        return _panel(request, error="question not found", status=404)
    try:
        span = _validated_span(store, doc_id, start, end)
    except ValueError as e:
        return _panel(request, error=str(e), status=400)
    store.upsert_question(Question(q.id, q.text, (span,)))
    return _panel(request)


@router.delete("/questions/{qid}", response_class=HTMLResponse)
def delete_question(request: Request, qid: str):
    if _lookup(request, qid) is None:
        return _panel(request, error="question not found", status=404)
    request.app.state.store.delete_question(qid)
    return _panel(request)


def _llm_error(e: Exception) -> str:
    return e.args[0] if isinstance(e, KeyError) and e.args else str(e)


@router.post("/questions/generate-from-selection", response_class=HTMLResponse)
def generate_from_selection(
    request: Request,
    doc_id: Annotated[str, Form()],
    start: Annotated[int, Form()],
    end: Annotated[int, Form()],
    llm: Annotated[str, Form()] = "openai",
):
    store = request.app.state.store
    try:
        span = _validated_span(store, doc_id, start, end)
        passage = store.get_document(doc_id).text[start:end]
        reply = build_llm(llm).complete(QUESTION_PROMPT.format(passage=passage))
    except (ValueError, KeyError, MissingApiKeyError) as e:
        return _panel(request, error=_llm_error(e), status=400)
    except Exception as e:  # noqa: BLE001 - provider/network failure must not 500 away the panel
        return _panel(request, error=f"{type(e).__name__}: {e}", status=400)
    first = next((ln.strip() for ln in reply.splitlines() if ln.strip()), "")
    if not first:
        return _panel(request, error="LLM returned an empty reply", status=400)
    store.upsert_question(Question(_new_id(), first, (span,)))
    return _panel(request)


@router.post("/questions/auto-generate", response_class=HTMLResponse)
def auto_generate(
    request: Request,
    per_doc: Annotated[int, Form()] = 5,
    llm: Annotated[str, Form()] = "openai",
    min_len: Annotated[int, Form()] = 200,
    max_len: Annotated[int, Form()] = 600,
):
    store = request.app.state.store
    docs = [store.get_document(row.id) for row in store.list_documents()]
    warnings: list[str] = []
    try:
        qs = generate_questions(
            docs,
            build_llm(llm),
            per_doc=per_doc,
            min_len=min_len,
            max_len=max_len,
            warn=warnings.append,
        )
    except (KeyError, MissingApiKeyError) as e:
        return _panel(request, error=_llm_error(e), status=400)
    except Exception as e:  # noqa: BLE001 - provider/network failure must not 500 away the panel
        return _panel(request, error=f"{type(e).__name__}: {e}", status=400)
    existing = {q.id for q in store.list_questions()}
    for q in qs:
        qid = q.id if q.id not in existing else f"{q.id}-{secrets.token_hex(2)}"
        existing.add(qid)
        store.upsert_question(Question(qid, q.text, q.spans))
    message = f"generated {len(qs)} question(s)"
    if warnings:
        message += "; " + " / ".join(warnings)
    if not qs:
        # Nothing generated is a failure the user must see, matching the CLI's exit 2.
        return _panel(request, error=message, status=400)
    return _panel(request, notice=message)


@router.get("/questions/export")
def export_questions(request: Request):
    payload = {
        "version": FORMAT_VERSION,
        "questions": [
            {
                "id": q.id,
                "text": q.text,
                "spans": [{"doc_id": s.doc_id, "start": s.start, "end": s.end} for s in q.spans],
            }
            for q in request.app.state.store.list_questions()
        ],
    }
    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="questions.json"'},
    )


@router.post("/questions/import", response_class=HTMLResponse)
async def import_questions(request: Request, file: UploadFile):
    store = request.app.state.store
    try:
        data = json.loads(await file.read())
        if not isinstance(data, dict):
            raise ValueError("questions file must be a JSON object")
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported questions file version: {data.get('version')}")
    except (ValueError, TypeError) as e:  # json.JSONDecodeError is a ValueError
        return _panel(request, error=str(e), status=400)
    added = skipped = 0
    for q in data.get("questions", []):
        try:
            qid = str(q["id"])
            if not _ID_RE.match(qid):
                raise ValueError(f"invalid question id '{qid}'")
            spans = tuple(
                _validated_span(store, str(s["doc_id"]), int(s["start"]), int(s["end"]))
                for s in q["spans"]
            )
            if not spans:
                raise ValueError("no spans")
            text = str(q["text"])
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        store.upsert_question(Question(qid, text, spans))
        added += 1
    return _panel(request, notice=f"imported {added}, skipped {skipped}")
