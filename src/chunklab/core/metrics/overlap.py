from __future__ import annotations

from collections.abc import Sequence

from chunklab.core.models import Chunk, Span


def coverage(chunk: Chunk, gold: Span) -> float:
    return chunk.span.overlap(gold) / gold.length


def best_coverage(chunk: Chunk, golds: Sequence[Span]) -> float:
    return max((coverage(chunk, g) for g in golds), default=0.0)


def is_hit(chunk: Chunk, golds: Sequence[Span], threshold: float) -> bool:
    return best_coverage(chunk, golds) >= threshold
