from __future__ import annotations

import numpy as np

from chunklab.core.embedders.base import Embedder, l2_normalize
from chunklab.core.models import Chunk, RetrievedChunk


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self._chunks = chunks
        self._matrix = l2_normalize(embeddings)
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q = l2_normalize(self._embedder.embed([query]))[0]
        scores = self._matrix @ q
        k = min(k, len(self._chunks))
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(self._chunks[i], float(scores[i]), rank)
            for rank, i in enumerate(top, start=1)
        ]
