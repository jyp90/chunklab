from chunklab.core.metrics.overlap import best_coverage, coverage, is_hit
from chunklab.core.models import Chunk, Span


def _c(s, e, doc="d"):
    return Chunk(doc, s, e, "x" * (e - s))


def test_coverage_full():
    assert coverage(_c(0, 100), Span("d", 10, 20)) == 1.0


def test_coverage_partial_is_relative_to_gold_length():
    assert coverage(_c(0, 15), Span("d", 10, 20)) == 0.5


def test_coverage_zero_when_disjoint_or_other_doc():
    assert coverage(_c(0, 5), Span("d", 10, 20)) == 0.0
    assert coverage(_c(0, 50, doc="other"), Span("d", 10, 20)) == 0.0


def test_best_coverage_takes_max():
    golds = [Span("d", 10, 20), Span("d", 100, 110)]
    assert best_coverage(_c(0, 15), golds) == 0.5
    assert best_coverage(_c(0, 15), []) == 0.0


def test_is_hit_threshold_inclusive():
    golds = [Span("d", 10, 20)]
    assert is_hit(_c(0, 15), golds, threshold=0.5) is True
    assert is_hit(_c(0, 14), golds, threshold=0.5) is False
