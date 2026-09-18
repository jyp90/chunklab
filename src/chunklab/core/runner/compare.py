from __future__ import annotations

from collections.abc import Sequence

from chunklab.core.runner.run import RunResult


def parse_thresholds(exprs: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for expr in exprs:
        metric, sep, value = expr.partition("=")
        if not sep or not metric:
            raise ValueError(f"expected METRIC=VALUE, got '{expr}'")
        try:
            out[metric.strip()] = float(value)
        except ValueError:
            raise ValueError(
                f"threshold value must be a number, got '{value}' in '{expr}'"
            ) from None
    return out


def check_thresholds(result: RunResult, thresholds: dict[str, float]) -> list[str]:
    violations: list[str] = []
    for c in result.combos:
        if c.error:
            violations.append(f"{c.combo_id}: failed ({c.error})")
            continue
        for metric, minimum in thresholds.items():
            if metric in c.metrics and c.metrics[metric] < minimum:
                violations.append(f"{c.combo_id}: {metric} {c.metrics[metric]:.3f} < {minimum:.3f}")
    produced = {m for c in result.combos if not c.error for m in c.metrics}
    unseen = set(thresholds) - produced
    if unseen and produced:
        raise ValueError(
            f"--fail-below metric(s) not produced by this run: {sorted(unseen)}; "
            f"available: {sorted(produced)}"
        )
    return violations


def check_regression(result: RunResult, baseline: RunResult, max_drop: float) -> list[str]:
    base_by_id = {c.combo_id: c for c in baseline.combos if not c.error}
    violations: list[str] = []
    for c in result.combos:
        b = base_by_id.get(c.combo_id)
        if b is None or c.error:
            continue
        for metric, prev in b.metrics.items():
            cur = c.metrics.get(metric)
            if cur is not None and cur < prev - max_drop:
                violations.append(
                    f"{c.combo_id}: {metric} dropped {prev:.3f} -> {cur:.3f} "
                    f"(max drop {max_drop:.3f})"
                )
    matched = [c for c in result.combos if c.combo_id in base_by_id]
    if result.combos and not matched:
        violations.append("baseline shares no combo ids with this run (grid or embedder changed?)")
    return violations


def format_table(result: RunResult) -> str:
    metric_keys: list[str] = []
    for c in result.combos:
        for k in c.metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    id_width = max([len("combo")] + [len(c.combo_id) for c in result.combos])
    header = f"{'combo':<{id_width}}  " + "  ".join(f"{k:>9}" for k in metric_keys)
    lines = [header, "-" * len(header)]
    for c in result.combos:
        if c.error:
            lines.append(f"{c.combo_id:<{id_width}}  ERROR: {c.error}")
            continue
        cells = "  ".join(f"{c.metrics.get(k, float('nan')):>9.3f}" for k in metric_keys)
        lines.append(f"{c.combo_id:<{id_width}}  {cells}")
    return "\n".join(lines)
