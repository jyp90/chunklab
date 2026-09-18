from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.size == 0:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


@dataclass
class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder for tests and offline demos."""

    dim: int = 256
    name: str = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h, "little") % self.dim
                out[row, idx] += 1.0
        return l2_normalize(out)
