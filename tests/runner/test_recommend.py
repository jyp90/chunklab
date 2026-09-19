from chunklab.core.runner.recommend import Recommendation, recommend
from chunklab.core.runner.run import ComboResult, RunResult


def _c(cid, hit, prec, n=10, k=5, error=None):
    return ComboResult(
        cid,
        "recursive",
        {},
        "fake",
        k,
        False,
        {f"hit@{k}": hit, "precision": prec, "mrr": 0, "ndcg": 0, "iou": 0},
        n,
        error=error,
    )


def _r(*cs):
    return RunResult("r", "t", {}, list(cs))


def test_prefers_precision_among_top_hit_within_tolerance():
    rec = recommend(_r(_c("a", 1.0, 0.10), _c("b", 0.96, 0.30), _c("c", 0.80, 0.90)))
    assert rec.combo_id == "b"
    assert "hit@5 0.960" in rec.reason and "precision 0.300" in rec.reason


def test_tie_on_precision_prefers_fewer_chunks():
    rec = recommend(_r(_c("a", 1.0, 0.3, n=20), _c("b", 1.0, 0.3, n=8)))
    assert rec.combo_id == "b"


def test_errors_excluded_and_none_when_nothing_usable():
    assert recommend(_r(_c("a", 0, 0, error="x"))) is None
    assert recommend(_r()) is None
    assert recommend(_r(_c("a", 0.0, 0.0), _c("b", 0.0, 0.0, error="x"))).combo_id == "a"


def test_mixed_k_uses_each_combos_own_hit_key():
    rec = recommend(_r(_c("a", 1.0, 0.2, k=3), _c("b", 1.0, 0.4, k=5)))
    assert isinstance(rec, Recommendation) and rec.combo_id == "b"
