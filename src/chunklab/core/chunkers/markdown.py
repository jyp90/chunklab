from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

Section = tuple[int, int, tuple[str, ...]]


def _merge_heading_only(sections: list[Section], text: str) -> list[Section]:
    """Fold a section whose body is only its heading line into the section that
    follows it, so a bare title never becomes a chunk of its own. The following
    section keeps its own heading_path (which already names the parent). A
    trailing heading-only section has nothing to merge into and stays."""
    merged: list[Section] = []
    for start, end, path in reversed(sections):
        _, _, body = text[start:end].partition("\n")
        if not body.strip() and merged:
            merged[-1] = (start, merged[-1][1], merged[-1][2])
        else:
            merged.append((start, end, path))
    merged.reverse()
    return merged


@dataclass
class MarkdownChunker:
    chunk_size: int = 1024
    overlap: int = 0
    name: str = "markdown"

    def chunk(self, doc: Document) -> list[Chunk]:
        inner = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks: list[Chunk] = []
        for s, e, path in self._sections(doc.text):
            if e - s <= self.chunk_size:
                c = make_chunk(doc, s, e, heading_path=path)
                if c is not None:
                    chunks.append(c)
                continue
            sub = Document(id=doc.id, text=doc.text[s:e], source=doc.source)
            for sc in inner.chunk(sub):
                chunks.append(
                    Chunk(doc.id, sc.start + s, sc.end + s, sc.text, {"heading_path": path})
                )
        return chunks

    @staticmethod
    def _sections(text: str) -> list[Section]:
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(0, len(text), ())]
        preamble: list[Section] = []
        if matches[0].start() > 0:
            preamble.append((0, matches[0].start(), ()))
        out: list[Section] = []
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append((m.start(), end, tuple(t for _, t in stack)))
        return preamble + _merge_heading_only(out, text)
