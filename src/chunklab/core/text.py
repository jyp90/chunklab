from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from chunklab.core.models import Document

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_PDF_SUFFIXES = {".pdf"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


class UnsupportedFormatError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _read_pdf(path: Path) -> str:
    import pymupdf  # lazy import

    with pymupdf.open(path) as pdf:
        return "\n\n".join(page.get_text() for page in pdf)


def load_document(path: Path) -> Document:
    size = path.stat().st_size
    if size > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(f"{path} is {size} bytes; limit is {MAX_DOCUMENT_BYTES} bytes")
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        raw = path.read_text(encoding="utf-8")
    elif suffix in _PDF_SUFFIXES:
        raw = _read_pdf(path)
    else:
        raise UnsupportedFormatError(f"unsupported file type: {path.suffix} ({path})")
    text = normalize(raw)
    if not text.strip():
        raise EmptyDocumentError(f"{path} contains no text")
    return Document(id=path.stem, text=text, source=str(path.resolve()))


def load_documents(
    paths: Iterable[Path], on_error: Callable[[Path, Exception], None] | None = None
) -> list[Document]:
    """Load every path. A per-document failure is re-raised unless `on_error` is
    given, in which case the document is skipped after reporting it. A duplicate
    document id is always a hard error."""
    docs: list[Document] = []
    seen: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if p.stem in seen:
            raise ValueError(f"duplicate document id '{p.stem}': {seen[p.stem]} and {p}")
        seen[p.stem] = p
        try:
            docs.append(load_document(p))
        except (
            UnsupportedFormatError,
            DocumentTooLargeError,
            RuntimeError,
            ValueError,
            OSError,
        ) as e:
            if on_error is None:
                raise
            on_error(p, e)
            continue
    return docs
