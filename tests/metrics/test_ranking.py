import math

import pytest

from chunklab.core.metrics.ranking import (
    QuestionMetrics,
    char_precision_iou,
    evaluate,
    hit_at_k,
    mean_metrics,
    ndcg_at_k,
    reciprocal_rank,
)
from chunklab.core.models import Chunk, Span


def _c(s, e):
    return Chunk("d", s, e, "x" * (e - s))


GOLD = [Span("d", 100, 200)]
MISS = _c(0, 50)
HIT = _c(90, 210)
HALF = _c(150, 300)  # covers 50 of 100 gold chars


def test_hit_at_k_respects_k():
    ranked = [MISS, MISS, HIT]
    assert hit_at_k(ranked, GOLD, k=3, threshold=0.5) is True
    assert hit_at_k(ranked, GOLD, k=2, threshold=0.5) is False


def test_reciprocal_rank():
    assert reciprocal_rank([MISS, HIT], GOLD, k=5, threshold=0.5) == 0.5
    assert reciprocal_rank([HIT], GOLD, k=5, threshold=0.5) == 1.0
    assert reciprocal_rank([MISS, MISS], GOLD, k=5, threshold=0.5) == 0.0
    assert reciprocal_rank([MISS, MISS, HIT], GOLD, k=2, threshold=0.5) == 0.0


def test_ndcg_perfect_first_rank():
    assert ndcg_at_k([HIT, MISS], GOLD, k=2) == pytest.approx(1.0)


def test_ndcg_graded_by_coverage_and_position():
    # gain at rank 2 = 0.5 -> dcg = 0.5 / log2(3); idcg = 1.0
    expected = 0.5 / math.log2(3)
    assert ndcg_at_k([MISS, HALF], GOLD, k=2) == pytest.approx(expected)


def test_ndcg_two_golds_idcg_uses_two_slots():
    golds = [Span("d", 100, 200), Span("d", 500, 600)]
    ranked = [_c(100, 200), _c(0, 10), _c(500, 600)]
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, golds, k=3) == pytest.approx(dcg / idcg)


def test_char_precision_iou():
    # retrieved chars: [90,210) = 120 chars; gold [100,200) = 100 chars
    p, iou = char_precision_iou([HIT], GOLD)
    assert p == pytest.approx(100 / 120)
    assert iou == pytest.approx(100 / 120)  # union is also 120


def test_char_precision_iou_overlapping_chunks_counted_once():
    p, iou = char_precision_iou([_c(100, 150), _c(120, 200)], GOLD)
    assert p == pytest.approx(1.0)
    assert iou == pytest.approx(1.0)


def test_char_precision_iou_empty_retrieval():
    assert char_precision_iou([], GOLD) == (0.0, 0.0)


def test_evaluate_uses_top_k_only():
    m = evaluate([MISS, MISS, HIT], GOLD, k=2, threshold=0.5)
    assert m == QuestionMetrics(hit=False, reciprocal_rank=0.0, ndcg=0.0, precision=0.0, iou=0.0)


def test_mean_metrics_keys_and_values():
    a = QuestionMetrics(True, 1.0, 1.0, 0.8, 0.8)
    b = QuestionMetrics(False, 0.0, 0.0, 0.0, 0.0)
    out = mean_metrics([a, b], k=5)
    assert out == {"hit@5": 0.5, "mrr": 0.5, "ndcg": 0.5, "precision": 0.4, "iou": 0.4}


def test_mean_metrics_empty():
    assert mean_metrics([], k=3) == {
        "hit@3": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "precision": 0.0,
        "iou": 0.0,
    }


def test_ndcg_overlapping_chunks_do_not_exceed_one():
    ranked = [_c(0, 300), _c(50, 250), _c(90, 210)]
    assert ndcg_at_k(ranked, GOLD, k=3) == pytest.approx(1.0)


def test_ndcg_marginal_gain_two_partial_chunks():
    ranked = [_c(100, 150), _c(150, 200)]
    expected = 0.5 + 0.5 / math.log2(3)
    assert ndcg_at_k(ranked, GOLD, k=2) == pytest.approx(expected)
