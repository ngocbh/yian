"""Tests for `condition_reads` and `rewrite_condition_reads`.

`condition_reads` lists each reference occurrence of a parsed condition with the outermost
`:-` / `:?` / `|` sub-expression around it and whether that sub-expression co-skips as a
binding would. `rewrite_condition_reads` renames reference leaves by occurrence, so the same
path read plainly and inside an escape can get different names.
"""

import pytest

from yian.expr import (
    condition_reads,
    parse_condition,
    rewrite_condition_reads,
    rewrite_condition_refs,
)


def _reads(when: str) -> list[tuple]:
    return [(r.path, r.escape, r.co_skips) for r in condition_reads(parse_condition(when))]


@pytest.mark.parametrize(
    "when, expected",
    [
        ("a.output >= 1", [("a.output", None, False)]),
        ("(a.output :- 0) >= 1", [("a.output", 0, False)]),
        ("(a.output | 0) >= 1", [("a.output", 0, False)]),
        ('(a.output :? "m") >= 1', [("a.output", 0, False)]),
        ("(a.output | b.output) == 'x'", [("a.output", 0, True), ("b.output", 0, True)]),
        (
            "(upper(a.output) :- b.output) == 'X'",
            [("a.output", 0, True), ("b.output", 0, True)],
        ),
        ("(a.output :? b.output) > 1", [("a.output", 0, False), ("b.output", 0, False)]),
        ("((a.output :- b.output) | 0) > 1", [("a.output", 0, False), ("b.output", 0, False)]),
        (
            "a.output > 5 or (a.output :- 0) < 1",
            [("a.output", None, False), ("a.output", 0, False)],
        ),
        (
            "(a.output :- 0) > 1 and (b.output | c.output) == 2",
            [("a.output", 0, False), ("b.output", 1, True), ("c.output", 1, True)],
        ),
        (
            "upper(a.output :- '') == 'X' and b.output",
            [("a.output", 0, False), ("b.output", None, False)],
        ),
        (
            "isinstance(d.output, Buy) and (d.output.qty :- 0) > 5",
            [("d.output", None, False), ("d.output.qty", 0, False)],
        ),
        ("x.output and a.output :- false", [("x.output", 0, False), ("a.output", 0, False)]),
        ("${(a.output :- 0) >= 1}", [("a.output", 0, False)]),
        ("(1 :- 2) > a.output", [("a.output", None, False)]),
    ],
)
def test_condition_reads(when, expected):
    assert _reads(when) == expected


def _by_index(names: dict):
    return lambda i, read: names.get(i)


def test_rewrite_renames_each_occurrence():
    when = "a.output > 5 or (a.output :- 0) < 1"
    renamed = rewrite_condition_reads(when, _by_index({0: "__r0", 1: "__r1"}))
    assert renamed == "__r0 > 5 or (__r1 :- 0) < 1"


def test_rewrite_keeps_whole_span_braces():
    renamed = rewrite_condition_reads("${(a.output :- 0) >= 1}", _by_index({0: "__r0"}))
    assert renamed == "${(__r0 :- 0) >= 1}"


def test_rewrite_leaves_string_literals_alone():
    renamed = rewrite_condition_reads("(a.output :- '}') == '}'", _by_index({0: "__r0"}))
    assert renamed == "(__r0 :- '}') == '}'"


def test_rewrite_none_keeps_leaf():
    when = "a.output > 5 or (a.output :- 0) < 1"
    renamed = rewrite_condition_reads(when, _by_index({1: "__r1"}))
    assert renamed == "a.output > 5 or (__r1 :- 0) < 1"


def test_rewrite_passes_the_read():
    seen = []
    rewrite_condition_reads(
        "(b.output | c.output) == 2", lambda i, read: seen.append((i, read.path, read.escape))
    )
    assert seen == [(0, "b.output", 0), (1, "c.output", 0)]


@pytest.mark.parametrize(
    "when",
    [
        "score.output > 0 and input.weight * 100 <= 5",
        "${score.output >= 0.5}",
        "isinstance(d.output, Buy) and d.output.qty > 5",
    ],
)
def test_rewrite_by_path_matches_rewrite_condition_refs(when):
    paths = {"score.output": "__s", "input.weight": "__w", "d.output": "__d", "d.output.qty": "__q"}
    by_read = rewrite_condition_reads(when, lambda i, read: paths.get(read.path))
    assert by_read == rewrite_condition_refs(when, paths.get)
