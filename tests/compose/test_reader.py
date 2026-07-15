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

