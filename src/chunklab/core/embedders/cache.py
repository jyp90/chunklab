from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from chunklab.core.embedders.base import Embedder


def cache_key(embedder_name: str, text: str) -> str:
    return hashlib.sha256(f"{embedder_name}\x00{text}".encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({marks})", batch
            ).fetchall()
            for key, dim, blob in rows:
                out[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        rows = [
            (k, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
            for k, v in items.items()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = inner.name
        self.misses = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [cache_key(self.name, t) for t in texts]
        found = self._cache.get_many(list(dict.fromkeys(keys)))
        missing_texts: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found:
                missing_texts[k] = t
        if missing_texts:
            vecs = self._inner.embed(list(missing_texts.values()))
            self.misses += len(missing_texts)
            new = dict(zip(missing_texts.keys(), vecs, strict=True))
            self._cache.put_many(new)
            found.update(new)
        if not texts:
            probe = self._inner.embed([])
            return np.zeros((0, probe.shape[1]), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)
