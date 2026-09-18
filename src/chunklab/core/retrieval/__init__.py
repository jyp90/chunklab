from __future__ import annotations

import numpy as np

from chunklab.core.embedders.base import Embedder
from chunklab.core.models import Chunk
from chunklab.core.retrieval.bm25 import BM25Index, tokenize
from chunklab.core.retrieval.dense import DenseIndex
from chunklab.core.retrieval.hybrid import HybridIndex, Retriever, rrf_fuse


def build_index(
    chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder, hybrid: bool
) -> Retriever:
    dense = DenseIndex(chunks, embeddings, embedder)
    if not hybrid:
        return dense
    return HybridIndex(dense, BM25Index(chunks))


__all__ = [
    "BM25Index",
    "DenseIndex",
    "HybridIndex",
    "Retriever",
    "build_index",
    "rrf_fuse",
    "tokenize",
]
