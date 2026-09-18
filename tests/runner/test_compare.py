import pytest

from chunklab.core.runner import (
    ComboResult,
    RunResult,
    check_regression,
    check_thresholds,
    format_table,
    parse_thresholds,
)


def _combo(cid: str, **metrics) -> ComboResult:
    return ComboResult(cid, "recursive", {}, "fake", 5, False, metrics=dict(metrics), n_chunks=3)


def _result(*combos: ComboResult) -> RunResult:
    return RunResult("r", "2026-01-01T00:00:00+00:00", {}, list(combos))


def test_parse_thresholds():
    assert parse_thresholds(["hit@5=0.8", "mrr=0.5"]) == {"hit@5": 0.8, "mrr": 0.5}
    with pytest.raises(ValueError):
        parse_thresholds(["hit@5"])
    with pytest.raises(ValueError):
        parse_thresholds(["hit@5=high"])


def test_check_thresholds_reports_violations_and_errors():
    res = _result(
        _combo("a", **{"hit@5": 0.9, "mrr": 0.7}),
        _combo("b", **{"hit@5": 0.5, "mrr": 0.7}),
        ComboResult("c", "recursive", {}, "fake", 5, False, error="RuntimeError: x"),
    )
    v = check_thresholds(res, {"hit@5": 0.8})
    assert v == ["b: hit@5 0.500 < 0.800", "c: failed (RuntimeError: x)"]


def test_check_thresholds_raises_when_metric_absent_everywhere():
    res = _result(_combo("a", **{"hit@3": 0.1}))
    with pytest.raises(ValueError, match="hit@5"):
        check_thresholds(res, {"hit@5": 0.8})


def test_check_thresholds_skips_combo_lacking_key_when_others_have_it():
    res = _result(_combo("a", **{"hit@3": 0.1}), _combo("b", **{"hit@5": 0.5}))
    assert check_thresholds(res, {"hit@5": 0.8}) == ["b: hit@5 0.500 < 0.800"]


def test_check_regression_flags_baseline_with_no_shared_combo_ids():
    base = _result(_combo("a", **{"hit@5": 0.9}))
    cur = _result(_combo("z", **{"hit@5": 0.9}))
    assert check_regression(cur, base, max_drop=0.05) == [
        "baseline shares no combo ids with this run (grid or embedder changed?)"
    ]


def test_check_regression_flags_drops_beyond_max_drop():
    base = _result(_combo("a", **{"hit@5": 0.9, "iou": 0.5}), _combo("b", **{"hit@5": 0.8}))
    cur = _result(_combo("a", **{"hit@5": 0.8, "iou": 0.48}), _combo("z", **{"hit@5": 0.1}))
    v = check_regression(cur, base, max_drop=0.05)
    assert v == ["a: hit@5 dropped 0.900 -> 0.800 (max drop 0.050)"]


def test_format_table_contains_ids_metrics_and_error():
    res = _result(
        _combo("recursive(chunk_size=256)|fake|k=5|hybrid=False", **{"hit@5": 0.75, "mrr": 0.5}),
        ComboResult("bad|fake|k=5|hybrid=False", "bad", {}, "fake", 5, False, error="Boom: x"),
    )
    table = format_table(res)
    assert "recursive(chunk_size=256)|fake|k=5|hybrid=False" in table
    assert "0.750" in table
    assert "ERROR" in table
    assert "hit@5" in table.splitlines()[0]
