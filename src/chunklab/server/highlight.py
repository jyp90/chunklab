from __future__ import annotations

import html
from collections.abc import Sequence

Interval = tuple[int, int]


def _clamp(spans: Sequence[Interval], n: int) -> list[Interval]:
    out = []
    for s, e in spans:
        s, e = max(0, s), min(n, e)
        if e > s:
            out.append((s, e))
    return out


def render_highlighted(text: str, gold: Sequence[Interval], hits: Sequence[Interval] = ()) -> str:
    n = len(text)
    gold_c, hits_c = _clamp(gold, n), _clamp(hits, n)
    cuts = {0, n}
    for s, e in gold_c + hits_c:
        cuts.update((s, e))
    points = sorted(cuts)

    def cls(a: int, b: int) -> str:
        g = any(s <= a and b <= e for s, e in gold_c)
        h = any(s <= a and b <= e for s, e in hits_c)
        return "both" if g and h else "gold" if g else "hit" if h else ""

    segments: list[tuple[str, str]] = []  # (class, escaped text)
    for a, b in zip(points, points[1:], strict=False):
        c = cls(a, b)
        piece = html.escape(text[a:b], quote=False)
        if segments and segments[-1][0] == c:
            segments[-1] = (c, segments[-1][1] + piece)
        else:
            segments.append((c, piece))
    return "".join(f'<mark class="{c}">{t}</mark>' if c else t for c, t in segments)
