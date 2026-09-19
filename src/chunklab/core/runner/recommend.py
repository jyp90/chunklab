from __future__ import annotations

from dataclasses import dataclass

from chunklab.core.runner.run import ComboResult, RunResult


@dataclass(frozen=True)
class Recommendation:
    combo_id: str
    reason: str


def _hit(c: ComboResult) -> float:
    return c.metrics.get(f"hit@{c.top_k}", 0.0)


def recommend(result: RunResult, tolerance: float = 0.05) -> Recommendation | None:
    """Pick the cheapest combo that keeps recall: best hit@k within `tolerance`,
    then highest precision, then fewest chunks."""
    ok = [c for c in result.combos if not c.error and c.metrics]
    if not ok:
        return None
    best_hit = max(_hit(c) for c in ok)
    candidates = [c for c in ok if _hit(c) >= best_hit - tolerance]
    best_prec = max(c.metrics.get("precision", 0.0) for c in candidates)
    finalists = [c for c in candidates if c.metrics.get("precision", 0.0) == best_prec]
    pick = min(finalists, key=lambda c: c.n_chunks)
    reason = (
        f"hit@{pick.top_k} {_hit(pick):.3f} (best {best_hit:.3f}), "
        f"precision {best_prec:.3f} (best among candidates), {pick.n_chunks} chunks"
    )
    return Recommendation(pick.combo_id, reason)
