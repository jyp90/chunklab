from pathlib import Path

import numpy as np

from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.embedders.cache import CachedEmbedder, EmbeddingCache, cache_key


def test_cache_key_depends_on_name_and_text():
    assert cache_key("a", "t") != cache_key("b", "t")
    assert cache_key("a", "t") != cache_key("a", "u")
    assert cache_key("a", "t") == cache_key("a", "t")


def test_cache_roundtrip(tmp_path: Path):
    with EmbeddingCache(tmp_path / "c.db") as cache:
        vec = np.arange(4, dtype=np.float32)
        cache.put_many({"k1": vec})
        got = cache.get_many(["k1", "missing"])
        assert set(got) == {"k1"}
        assert np.array_equal(got["k1"], vec)


def test_cache_persists_across_instances(tmp_path: Path):
    p = tmp_path / "c.db"
    with EmbeddingCache(p) as c1:
        c1.put_many({"k": np.ones(3, dtype=np.float32)})
    with EmbeddingCache(p) as c2:
        assert "k" in c2.get_many(["k"])


def test_cached_embedder_only_calls_inner_for_misses(tmp_path: Path):
    inner = FakeEmbedder(dim=16)
    with EmbeddingCache(tmp_path / "c.db") as cache:
        ce = CachedEmbedder(inner, cache)
        assert ce.name == "fake"
        first = ce.embed(["a", "b", "c"])
        assert ce.misses == 3
        second = ce.embed(["b", "c", "d"])
        assert ce.misses == 4
        assert np.array_equal(first[1], second[0])
        assert np.array_equal(first[2], second[1])
        assert second.shape == (3, 16)


def test_cached_embedder_preserves_order_and_duplicates(tmp_path: Path):
    inner = FakeEmbedder(dim=8)
    with EmbeddingCache(tmp_path / "c.db") as cache:
        ce = CachedEmbedder(inner, cache)
        out = ce.embed(["x", "y", "x"])
        direct = inner.embed(["x", "y", "x"])
        assert np.array_equal(out, direct)
