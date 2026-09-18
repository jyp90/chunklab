from chunklab.core.models import Chunk, RetrievedChunk
from chunklab.core.retrieval.hybrid import HybridIndex, rrf_fuse

A = Chunk("d", 0, 1, "a")
B = Chunk("d", 1, 2, "b")
C = Chunk("d", 2, 3, "c")


class _Fixed:
    def __init__(self, order):
        self.order = order

    def search(self, query, k):
        return [RetrievedChunk(c, 1.0 / (i + 1), i + 1) for i, c in enumerate(self.order[:k])]


def test_rrf_fuse_prefers_chunk_ranked_well_in_both():
    fused = rrf_fuse([[A, B, C], [B, A, C]], rrf_k=60)
    # A: 1/61 + 1/62 ; B: 1/62 + 1/61 -> tie broken by first appearance (A first)
    assert [c for c, _ in fused][:2] == [A, B]
    assert fused[0][1] == fused[1][1]


def test_hybrid_search_ranks_and_truncates():
    idx = HybridIndex(_Fixed([A, B, C]), _Fixed([C, A, B]))
    res = idx.search("q", k=2)
    assert len(res) == 2
    assert res[0].chunk == A  # ranks 1 and 2
    assert [r.rank for r in res] == [1, 2]
