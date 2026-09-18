from chunklab.core.embedders.base import FakeEmbedder
from chunklab.core.models import Chunk
from chunklab.core.retrieval.dense import DenseIndex

CHUNKS = [
    Chunk("d", 0, 10, "refunds are issued within 30 days"),
    Chunk("d", 10, 20, "orders ship within 2 business days"),
    Chunk("d", 20, 30, "contact support with your order number"),
]


def test_dense_ranks_lexically_similar_chunk_first():
    emb = FakeEmbedder()
    idx = DenseIndex(CHUNKS, emb.embed([c.text for c in CHUNKS]), emb)
    res = idx.search("how long do refunds take, 30 days?", k=2)
    assert len(res) == 2
    assert res[0].chunk == CHUNKS[0]
    assert res[0].rank == 1 and res[1].rank == 2
    assert res[0].score >= res[1].score


def test_dense_k_larger_than_corpus():
    emb = FakeEmbedder()
    idx = DenseIndex(CHUNKS, emb.embed([c.text for c in CHUNKS]), emb)
    assert len(idx.search("anything", k=10)) == 3
