"""Classifying `nodes:` entries: a node has `kind:`, a value has a YAML tag.

`parse_nodes` hands every tagged entry to its `read_value` callback, in source order
among the nodes, and rejects an untagged entry that has no `kind:` with a located error
saying to add `kind:` for a node or a tag for a value. The error for an untagged scalar
or list shows the tag to write, except under a secret name, where the value is never
echoed. Reserved node ids apply to value entries too.
"""

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument

_NODE_OR_VALUE = (
    "node 'x': an entry in `nodes:` is a node or a value; add `kind:` for a node, "
    "or a tag (`!object`, `!Buy`, ...) for a value"
)


def _flow(entry: str) -> str:
    """Return a flow whose `nodes:` holds `entry` (already indented) and a CODE node."""
    return (
        "id: f\n"
        "name: f\n"
        "nodes:\n"
        f"{entry}"
        "  step:\n"
        "    kind: code\n"
        "    code: |\n"
        "      return 1\n"
        "output: ${step.output}\n"
    )


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _nodes_cursor(text: str):
    """Return the `nodes:` cursor of a parsed document."""
    return SourceDocument.parse(text).root.child("nodes")


def test_untagged_mapping_without_kind_says_add_kind_or_a_tag():
    text = _flow("  x:\n    prompt: hi\n")

    with pytest.raises(LoadError, match=re.escape(_NODE_OR_VALUE)) as info:
        load_flow(text)

    assert _error_line(info.value) == _line_of(text, "prompt: hi")


@pytest.mark.parametrize(
    ("entry", "example"),
    [
        ("  x: 5\n", "`x: !int 5`"),
        ("  x: hello\n", "`x: !str hello`"),
        ("  x: [1, 2]\n", "`x: !list(int) [1, 2]`"),
        ("  x:\n", "`x: !Optional(str) null`"),
    ],
)
def test_untagged_scalar_or_list_shows_the_tag_to_write(entry, example):
    text = _flow(entry)

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert str(info.value.diagnostic.message) == f"{_NODE_OR_VALUE}, e.g. {example}"
    assert _error_line(info.value) == _line_of(text, "  x:")


def test_untagged_secret_literal_is_rejected_without_echoing_it():
    text = _flow("  api_key: sk-live-123\n")

    with pytest.raises(LoadError) as info:
        load_flow(text)

    message = info.value.diagnostic.message
    assert "value entry 'api_key' is a secret name" in message
    assert "${input.api_key}" in message
    assert "sk-live-123" not in str(info.value)
    assert _error_line(info.value) == _line_of(text, "api_key:")


@pytest.mark.parametrize("entry", ["  input: !int 5\n", "  input:\n    kind: code\n"])
def test_reserved_ids_apply_to_a_tagged_entry(entry):
    with pytest.raises(LoadError, match=re.escape("node id 'input' is reserved")):
        load_flow(_flow(entry))


def test_tagged_entries_are_read_by_the_callback_in_source_order():
    cursor = _nodes_cursor(
        "nodes:\n"
        "  a:\n"
        "    kind: code\n"
        "    code: return 1\n"
        "  x: !int 5\n"
        "  d: !Buy {a: 1}\n"
        "  b:\n"
        "    kind: code\n"
        "    code: return 2\n"
    )
    calls = []

    def read_value(node_id, entry):
        calls.append((node_id, entry.tag, entry.value))
        return f"value {node_id}"

    descriptors = parse_nodes(cursor, read_value=read_value)

    assert list(descriptors) == ["a", "x", "d", "b"]
    assert calls == [("x", "int", 5), ("d", "Buy", {"a": 1})]
    assert descriptors["x"] == "value x"
    assert descriptors["d"] == "value d"
    assert isinstance(descriptors["a"], CodeDescriptor)
    assert isinstance(descriptors["b"], CodeDescriptor)


def test_a_tagged_mapping_with_kind_is_still_a_value():
    cursor = _nodes_cursor("nodes:\n  v: !object {kind: code, code: return 1}\n")

    descriptors = parse_nodes(cursor, read_value=lambda node_id, entry: "value")

    assert descriptors == {"v": "value"}


def test_a_tagged_entry_without_a_reader_is_a_type_error():
    with pytest.raises(TypeError, match="read_value"):
        parse_nodes(_nodes_cursor("nodes:\n  x: !int 5\n"))
