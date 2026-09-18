from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

Segment = tuple[int, int]


@dataclass
class RecursiveChunker:
    chunk_size: int = 512
    overlap: int = 0
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    name: str = "recursive"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    def chunk(self, doc: Document) -> list[Chunk]:
        segments = self._split(doc.text, 0, list(self.separators))
        merged = self._merge(segments)
        chunks: list[Chunk] = []
        prev_start = 0
        for i, (s, e) in enumerate(merged):
            start = s if i == 0 else max(s - self.overlap, prev_start)
            c = make_chunk(doc, start, e)
            if c is not None:
                chunks.append(c)
            prev_start = s
        return chunks

    def _split(self, text: str, offset: int, separators: list[str]) -> list[Segment]:
        if not text:
            return []
        if len(text) <= self.chunk_size or not separators:
            return [(offset, offset + len(text))]
        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [
                (offset + i, offset + min(i + self.chunk_size, len(text)))
                for i in range(0, len(text), self.chunk_size)
            ]
        pieces: list[Segment] = []
        pos = 0
        for m in re.finditer(re.escape(sep), text):
            pieces.append((pos, m.end()))
            pos = m.end()
        if pos < len(text):
            pieces.append((pos, len(text)))
        if len(pieces) <= 1:
            return self._split(text, offset, rest)
        out: list[Segment] = []
        for s, e in pieces:
            if e - s > self.chunk_size:
                out.extend(self._split(text[s:e], offset + s, rest))
            else:
                out.append((offset + s, offset + e))
        return out

    def _merge(self, segments: list[Segment]) -> list[Segment]:
        merged: list[Segment] = []
        cur: Segment | None = None
        for s, e in segments:
            if cur is None:
                cur = (s, e)
            elif e - cur[0] <= self.chunk_size:
                cur = (cur[0], e)
            else:
                merged.append(cur)
                cur = (s, e)
        if cur is not None:
            merged.append(cur)
        return merged
