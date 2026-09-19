from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse

from chunklab.core.text import MAX_DOCUMENT_BYTES, load_document

router = APIRouter()
_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_stem(filename: str) -> str:
    """Derive a filesystem-safe stem from a client-supplied filename.

    Only the basename is used (any directory components are discarded), the
    final extension is dropped, leading/trailing dots and underscores are
    stripped, and any remaining character outside ``[A-Za-z0-9._-]`` is
    replaced with ``_``. An empty result falls back to ``"document"``.
    """
    name = Path(filename).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    stem = stem.strip("._")
    stem = _SAFE.sub("_", stem)
    return stem or "document"


def _render(request: Request, name: str, ctx: dict, status_code: int = 200) -> HTMLResponse:
    return request.app.state.templates.TemplateResponse(request, name, ctx, status_code=status_code)


def _doc_list_ctx(request: Request, skipped: list[str] | None = None) -> dict:
    return {"documents": request.app.state.store.list_documents(), "skipped": skipped or []}


@router.get("/", response_class=HTMLResponse)
def documents_page(request: Request):
    ctx = _doc_list_ctx(request)
    ctx.update(
        title="Documents",
        active="documents",
        questions=request.app.state.store.list_questions(),
    )
    return _render(request, "documents.html", ctx)


@router.post("/documents", response_class=HTMLResponse)
async def upload_documents(request: Request, files: list[UploadFile]):
    ws: Path = request.app.state.workspace
    docs_dir = ws / "docs"
    docs_dir.mkdir(exist_ok=True)
    skipped: list[str] = []
    for f in files:
        name = Path(f.filename or "document").name
        suffix = Path(name).suffix.lower()
        target = docs_dir / f"{safe_stem(name)}{suffix}"
        data = await f.read()
        if len(data) > MAX_DOCUMENT_BYTES:
            skipped.append(f"{name}: larger than {MAX_DOCUMENT_BYTES} bytes")
            continue
        target.write_bytes(data)
        try:
            doc = load_document(target)
        except ValueError as e:  # Unsupported / Empty / TooLarge are ValueError subclasses
            skipped.append(f"{name}: {e}")
            target.unlink(missing_ok=True)
            continue
        request.app.state.store.upsert_document(doc)
    return _render(request, "partials/doc_list.html", _doc_list_ctx(request, skipped))


@router.delete("/documents/{doc_id}", response_class=HTMLResponse)
def delete_document(request: Request, doc_id: str):
    if not _ID_RE.match(doc_id):
        return _render(request, "partials/doc_list.html", _doc_list_ctx(request), status_code=404)
    doc = request.app.state.store.get_document(doc_id)
    if doc is None:
        return _render(request, "partials/doc_list.html", _doc_list_ctx(request), status_code=404)
    docs_dir = (request.app.state.workspace / "docs").resolve()
    source = Path(doc.source).resolve()
    if source.parent == docs_dir and source.is_file():
        source.unlink(missing_ok=True)
    request.app.state.store.delete_document(doc_id)
    return _render(request, "partials/doc_list.html", _doc_list_ctx(request))


@router.get("/documents/{doc_id}/view", response_class=HTMLResponse)
def view_document(request: Request, doc_id: str, q: str | None = None):
    if not _ID_RE.match(doc_id):
        return HTMLResponse("<p class='error'>document not found</p>", status_code=404)
    doc = request.app.state.store.get_document(doc_id)
    if doc is None:
        return HTMLResponse("<p class='error'>document not found</p>", status_code=404)
    return _render(request, "partials/doc_view.html", {"doc": doc, "html": None, "q": q})
