from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from chunklab.core.models import Chunk, RetrievedChunk


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: Sequence[Sequence[Chunk]], rrf_k: int = 60) -> list[tuple[Chunk, float]]:
    scores: dict[Chunk, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (rrf_k + rank)
    # dict preserves first-insertion order; stable sort keeps it for ties
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridIndex:
    def __init__(self, dense: Retriever, sparse: Retriever, rrf_k: int = 60) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        cand = 2 * k
        dense = [r.chunk for r in self._dense.search(query, cand)]
        sparse = [r.chunk for r in self._sparse.search(query, cand)]
        fused = rrf_fuse([dense, sparse], self._rrf_k)[:k]
        return [RetrievedChunk(c, s, rank) for rank, (c, s) in enumerate(fused, start=1)]
