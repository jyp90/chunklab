from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.chunkers.recursive import RecursiveChunker
from chunklab.core.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)

Section = tuple[int, int, tuple[str, ...]]


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
        out: list[Section] = []
        if matches[0].start() > 0:
            out.append((0, matches[0].start(), ()))
        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out.append((m.start(), end, tuple(t for _, t in stack)))
        return out
