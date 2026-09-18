from chunklab.core.models import Chunk
from chunklab.core.retrieval.bm25 import BM25Index, tokenize

CHUNKS = [
    Chunk("d", 0, 10, "Refunds are issued within 30 days."),
    Chunk("d", 10, 20, "Orders ship within 2 business days."),
    Chunk("d", 20, 30, "Contact support with your order number."),
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Refunds, within 30 Days!") == ["refunds", "within", "30", "days"]


def test_bm25_exact_term_match_ranks_first():
    idx = BM25Index(CHUNKS)
    res = idx.search("support order number", k=3)
    assert res[0].chunk == CHUNKS[2]
    assert [r.rank for r in res] == [1, 2, 3]


def test_bm25_returns_at_most_k():
    assert len(BM25Index(CHUNKS).search("days", k=1)) == 1
