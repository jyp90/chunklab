from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from chunklab.core.embedders.base import l2_normalize
from chunklab.core.embedders.openai import MissingApiKeyError


@dataclass
class GeminiEmbedder:
    model: str = "gemini-embedding-001"
    batch_size: int = 100
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"gemini:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingApiKeyError("GEMINI_API_KEY is not set")
        from google import genai  # lazy import

        client = genai.Client()
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = client.models.embed_content(
                model=self.model, contents=texts[i : i + self.batch_size]
            )
            rows.extend(e.values for e in resp.embeddings)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(rows, dtype=np.float32))
