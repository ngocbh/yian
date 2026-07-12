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

