import numpy as np

from chunklab.core.embedders.base import Embedder, FakeEmbedder


def test_fake_embedder_shape_dtype_normalized():
    e = FakeEmbedder(dim=64)
    v = e.embed(["hello world", "another text"])
    assert isinstance(e, Embedder)
    assert v.shape == (2, 64)
    assert v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_fake_embedder_deterministic():
    a = FakeEmbedder().embed(["same text"])
    b = FakeEmbedder().embed(["same text"])
    assert np.array_equal(a, b)


def test_fake_embedder_similarity_reflects_shared_words():
    e = FakeEmbedder()
    v = e.embed(["refund within 30 days", "refund policy 30 days", "shipping tracking number"])
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far


def test_fake_embedder_empty_input():
    v = FakeEmbedder(dim=8).embed([])
    assert v.shape == (0, 8)
