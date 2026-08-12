from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange


def test_loaderror_carries_range():
    r = SourceRange(2, 1, 2, 6)
    e = LoadError("bad", range=r)
    assert e.range is r


def test_node_reader_field_and_missing():
    from agent_composer.compose.helpers.spanloader import load_with_spans
    from agent_composer.compose.helpers.reader import NodeReader
    import pytest

    value, spans, root = load_with_spans("a: 1\nb: two\n", return_root=True)
    r = NodeReader(value, spans, root)
    assert r.str("b") == "two"
    assert r.range_of("b").start_line == 2
    with pytest.raises(LoadError):
        r.str("missing")            # located LoadError


def test_item_range_list_positional_preserves_duplicates():
    """Positional twin of item_ranges: one entry per item, IN ORDER, DUPLICATES KEPT —
    the property #88's asserts-by-position fix rests on (item_ranges would collapse dups)."""
    from agent_composer.compose.helpers.spanloader import load_with_spans
    from agent_composer.compose.helpers.reader import NodeReader

    text = "asserts:\n  - ${output} > 0\n  - ${output} > 0\n"  # two IDENTICAL items
    value, spans, root = load_with_spans(text, return_root=True)
    r = NodeReader(value, spans, root)

    ranges = r.item_range_list("asserts")
    assert len(ranges) == 2                          # aligned 1:1 with the list, not collapsed
    assert ranges[0].start_line == 2 and ranges[1].start_line == 3  # distinct lines
    assert ranges[0] != ranges[1]
    # item_ranges (text-keyed) collapses the two identical items to ONE entry:
    assert len(r.item_ranges("asserts")) == 1


def test_item_range_list_absent_and_spanless_are_empty():
    from agent_composer.compose.helpers.spanloader import load_with_spans
    from agent_composer.compose.helpers.reader import NodeReader

    value, spans, root = load_with_spans("a: 1\n", return_root=True)
    assert NodeReader(value, spans, root).item_range_list("asserts") == []   # field absent
    assert NodeReader.spanless({"asserts": ["x > 0"]}).item_range_list("asserts") == []  # no spans


def test_sequence_field_ranges_positional_per_item_field():
    """Per-item VALUE range of a field in a SEQUENCE of MAPPINGS, positional 1:1 — the basis
    for a CASE's `cases[N].when` / `cases[N].then` caret (no unique text key to key on)."""
    from agent_composer.compose.helpers.spanloader import load_with_spans
    from agent_composer.compose.helpers.reader import NodeReader

    text = (
        "cases:\n"
        '  - when: "a >= 1"\n'   # L2
        "    then: x\n"          # L3
        '  - when: "b >= 2"\n'   # L4
        "    then: y\n"          # L5
    )
    value, spans, root = load_with_spans(text, return_root=True)
    r = NodeReader(value, spans, root)

    whens = r.sequence_field_ranges("cases", "when")
    assert len(whens) == 2                                    # one per case, IN ORDER
    assert whens[0].start_line == 2 and whens[1].start_line == 4
    assert [t.start_line for t in r.sequence_field_ranges("cases", "then")] == [3, 5]


def test_sequence_field_ranges_absent_spanless_and_nonmapping():
    from agent_composer.compose.helpers.spanloader import load_with_spans
    from agent_composer.compose.helpers.reader import NodeReader

    value, spans, root = load_with_spans("a: 1\n", return_root=True)
    assert NodeReader(value, spans, root).sequence_field_ranges("cases", "when") == []  # field absent
    assert NodeReader.spanless(
        {"cases": [{"when": "x"}]}
    ).sequence_field_ranges("cases", "when") == []                                       # no spans
    v2, s2, r2 = load_with_spans("cases:\n  - x\n  - y\n", return_root=True)             # scalar items
    assert NodeReader(v2, s2, r2).sequence_field_ranges("cases", "when") == [None, None]

