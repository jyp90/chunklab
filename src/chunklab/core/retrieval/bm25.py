from __future__ import annotations

import re

import numpy as np

from chunklab.core.models import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        k = min(k, len(self._chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(self._chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(top, start=1)
        ]
