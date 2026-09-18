from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chunklab.core.embedders.base import l2_normalize


@dataclass
class LocalEmbedder:
    model: str = "all-MiniLM-L6-v2"
    name: str = field(init=False)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"local:{self.model}"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # lazy import
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install local embedding support with: pip install 'chunklab[local]'"
                ) from e
            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return l2_normalize(np.asarray(vecs, dtype=np.float32))
