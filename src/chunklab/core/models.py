from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"span end must be > start, got start={self.start} end={self.end}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: Span) -> int:
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def span(self) -> Span:
        return Span(self.doc_id, self.start, self.end)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
