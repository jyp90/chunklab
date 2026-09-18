from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from chunklab.core.embedders.base import l2_normalize


class MissingApiKeyError(RuntimeError):
    pass


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingApiKeyError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # lazy import

        client = OpenAI()
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            rows.extend(item.embedding for item in resp.data)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(rows, dtype=np.float32))
