from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from chunklab.core.models import Document

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}
_PDF_SUFFIXES = {".pdf"}
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


class UnsupportedFormatError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _read_pdf(path: Path) -> str:
    import fitz  # pymupdf; lazy import

    with fitz.open(path) as pdf:
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
    return Document(id=path.stem, text=normalize(raw), source=str(path.resolve()))


def load_documents(paths: Iterable[Path]) -> list[Document]:
    docs: list[Document] = []
    seen: dict[str, Path] = {}
    for p in paths:
        p = Path(p)
        if p.stem in seen:
            raise ValueError(f"duplicate document id '{p.stem}': {seen[p.stem]} and {p}")
        seen[p.stem] = p
        docs.append(load_document(p))
    return docs
