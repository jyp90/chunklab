from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from chunklab.core.metrics.overlap import is_hit
from chunklab.core.models import Chunk, Span


@dataclass(frozen=True)
class QuestionMetrics:
    hit: bool
    reciprocal_rank: float
    ndcg: float
    precision: float
    iou: float


def hit_at_k(chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float) -> bool:
    return any(is_hit(c, golds, threshold) for c in chunks[:k])


def reciprocal_rank(
    chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float
) -> float:
    for rank, c in enumerate(chunks[:k], start=1):
        if is_hit(c, golds, threshold):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(chunks: Sequence[Chunk], golds: Sequence[Span], k: int) -> float:
    """NDCG with marginal gain: a chunk only earns credit for gold characters that
    no higher-ranked chunk already covered, so overlapping chunks cannot push the
    score above 1.0."""
    if not golds:
        return 0.0
    covered: set[tuple[str, int]] = set()
    dcg = 0.0
    for rank, c in enumerate(chunks[:k], start=1):
        chunk_pos = _positions([c.span])
        gain = 0.0
        for g in golds:
            new = (chunk_pos & _positions([g])) - covered
            gain += len(new) / g.length
            covered |= new
        dcg += gain / math.log2(rank + 1)
    ideal_slots = min(len(golds), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_slots + 1))
    return dcg / idcg if idcg else 0.0


def _positions(spans: Sequence[Span]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for s in spans:
        out.update((s.doc_id, i) for i in range(s.start, s.end))
    return out


def char_precision_iou(chunks: Sequence[Chunk], golds: Sequence[Span]) -> tuple[float, float]:
    retrieved = _positions([c.span for c in chunks])
    if not retrieved:
        return 0.0, 0.0
    gold = _positions(golds)
    inter = len(retrieved & gold)
    union = len(retrieved | gold)
    return inter / len(retrieved), (inter / union if union else 0.0)


def evaluate(
    chunks: Sequence[Chunk], golds: Sequence[Span], k: int, threshold: float
) -> QuestionMetrics:
    top = list(chunks[:k])
    precision, iou = char_precision_iou(top, golds)
    return QuestionMetrics(
        hit=hit_at_k(top, golds, k, threshold),
        reciprocal_rank=reciprocal_rank(top, golds, k, threshold),
        ndcg=ndcg_at_k(top, golds, k),
        precision=precision,
        iou=iou,
    )


def mean_metrics(items: Sequence[QuestionMetrics], k: int) -> dict[str, float]:
    n = len(items)

    def avg(values: list[float]) -> float:
        return sum(values) / n if n else 0.0

    return {
        f"hit@{k}": avg([1.0 if m.hit else 0.0 for m in items]),
        "mrr": avg([m.reciprocal_rank for m in items]),
        "ndcg": avg([m.ndcg for m in items]),
        "precision": avg([m.precision for m in items]),
        "iou": avg([m.iou for m in items]),
    }
