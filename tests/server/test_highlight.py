import html
import re

from chunklab.server.highlight import render_highlighted


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"</?mark[^>]*>", "", s))


def test_plain_when_no_spans():
    assert render_highlighted("a<b\n", [], []) == "a&lt;b\n"


def test_gold_and_hit_and_overlap_classes():
    out = render_highlighted("0123456789", gold=[(2, 6)], hits=[(4, 8)])
    assert (
        out
        == '01<mark class="gold">23</mark><mark class="both">45</mark><mark class="hit">67</mark>89'
    )


def test_text_content_is_preserved():
    text = "Refunds <30 days> & more.\nSecond line."
    out = render_highlighted(text, gold=[(0, 7), (10, 14)], hits=[(5, 12), (26, 32)])
    assert _strip(out) == text


def test_adjacent_same_class_merged_and_bounds_clamped():
    out = render_highlighted("abcdef", gold=[(0, 2), (2, 4)], hits=[(10, 20)])
    assert out == '<mark class="gold">abcd</mark>ef'
