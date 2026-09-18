from __future__ import annotations

import re
from dataclasses import dataclass

from chunklab.core.chunkers.base import make_chunk
from chunklab.core.models import Chunk, Document

_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


@dataclass
class SentenceWindowChunker:
    window: int = 1
    name: str = "sentence_window"

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be >= 0")

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = split_sentences(doc.text)
        chunks: list[Chunk] = []
        for i in range(len(sents)):
            lo = max(0, i - self.window)
            hi = min(len(sents) - 1, i + self.window)
            c = make_chunk(doc, sents[lo][0], sents[hi][1], center=i)
            if c is not None:
                chunks.append(c)
        return chunks
