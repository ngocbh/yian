"""Classifying `nodes:` entries: a node has `kind:`, a value has a YAML tag.

`parse_nodes` hands every tagged entry to its `read_value` callback, in source order
among the nodes, and rejects an untagged entry that has no `kind:` with a located error
saying to add `kind:` for a node or a tag for a value. The error for an untagged scalar,
list or date shows the tag to write (with `...` for a value written over several
lines), and the example loads when pasted in. There is no example for a value no tag
can hold (one that contains itself or holds a number that is not finite). An untagged
`${...}` or `call(...)` value is not a literal, so it gets no example. Reserved node ids
apply to value entries too.
"""

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument
from yian.run import run_flow

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

    with pytest.raises(LoadError, match=f"^{re.escape(_NODE_OR_VALUE)}$") as info:
        load_flow(text)

    assert _error_line(info.value) == _line_of(text, "prompt: hi")


@pytest.mark.parametrize(
    ("entry", "example"),
    [
        ("  x: 5\n", "`x: !int 5`"),
        ("  x: hello\n", "`x: !str hello`"),
        ('  x: "hi"\n', '`x: !str "hi"`'),
        ("  x: [1, 2]\n", "`x: !list(int) [1, 2]`"),
        ("  x: [[1, 2], [3]]\n", "`x: !list(list(int)) [[1, 2], [3]]`"),
        ("  x: [null, 1]\n", "`x: !list(Optional(int)) [null, 1]`"),
        ("  x: [1, 2.5]\n", "`x: !list(float) [1, 2.5]`"),
        ("  x: [[1], [null]]\n", "`x: !list(list(Optional(int))) [[1], [null]]`"),
        ("  x: [[], [1]]\n", "`x: !list(list(int)) [[], [1]]`"),
        ("  x: [1, a]\n", "`x: !list(Any) [1, a]`"),
        ("  x: [[1], [a]]\n", "`x: !list(Any) [[1], [a]]`"),
        ("  x: [{a: 1}, 2]\n", "`x: !list(Any) [{a: 1}, 2]`"),
        ("  x: [[1, null], a]\n", "`x: !list(Any) [[1, null], a]`"),
        ("  x: [null, 1, a]\n", "`x: !list(Optional(Any)) [null, 1, a]`"),
        ("  x: [[1, a], [b, 2]]\n", "`x: !list(list(Any)) [[1, a], [b, 2]]`"),
        (
            "  x: [1, 2024-01-01]\n",
            "`x: !<type> [1, 2024-01-01]` (quote its dates to tag it `list(Any)`)",
        ),
        (
            "  x: [null, [2024-01-01], a]\n",
            "`x: !<type> [null, [2024-01-01], a]` (quote its dates to tag it "
            "`list(Optional(Any))`)",
        ),
        ("  x:\n", "`x: !Optional(str) null`"),
        ("  x: 2024-01-01\n", '`x: !date "2024-01-01"`'),
        ("  x: 2024-01-01T10:00:00\n", '`x: !datetime "2024-01-01T10:00:00"`'),
    ],
)
def test_untagged_scalar_or_list_shows_the_tag_to_write(entry, example):
    text = _flow(entry)

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert str(info.value.diagnostic.message) == f"{_NODE_OR_VALUE}, e.g. {example}"
    assert _error_line(info.value) == _line_of(text, "  x:")


@pytest.mark.parametrize(
    ("entry", "value"),
    [
        ("  x: [null, 1]\n", [None, 1]),
        ("  x: [1, 2.5]\n", [1.0, 2.5]),
        ("  x: [[1], [null]]\n", [[1], [None]]),
        ("  x: [[], [1]]\n", [[], [1]]),
        ("  x: [null, [1]]\n", [None, [1]]),
        ("  x: [1, a]\n", [1, "a"]),
        ("  x: [[1], [a]]\n", [[1], ["a"]]),
        ("  x: [{a: 1}, 2]\n", [{"a": 1}, 2]),
        ("  x: [null, 1, a]\n", [None, 1, "a"]),
        ("  x: [[1, a], [b, 2]]\n", [[1, "a"], ["b", 2]]),
        ("  x: !!str 3\n", "3"),
        ("  x: !!str true\n", "true"),
    ],
)
def test_the_example_entry_loads(entry, value):
    with pytest.raises(LoadError) as info:
        load_flow(_flow(entry))
    found = re.search(r", e\.g\. `([^`]+)`$", str(info.value.diagnostic.message))
    assert found is not None, str(info.value)

    text = f"id: f\nname: f\nnodes:\n  {found[1]}\noutput: ${{x.output}}\n"

    assert run_flow(load_flow(text), {}).output == value


@pytest.mark.parametrize(
    "entry",
    ["  x: &R [*R]\n", "  x: &R [{a: *R}]\n", "  x: .inf\n", "  x: [{a: .nan}]\n"],
)
def test_an_untagged_value_no_tag_can_hold_gets_no_example(entry):
    text = _flow(entry)

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert str(info.value.diagnostic.message) == _NODE_OR_VALUE
    assert _error_line(info.value) == _line_of(text, "  x:")


def test_untagged_block_list_shows_the_tag_with_a_placeholder_value():
    text = _flow("  x:\n    - 1\n    - 2\n")

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert str(info.value.diagnostic.message) == (
        f"{_NODE_OR_VALUE}, e.g. `x: !list(int) ...`"
    )
    assert _error_line(info.value) == _line_of(text, "- 1")


def test_an_untagged_credential_named_literal_gets_the_example():
    text = _flow("  api_key: sk-live-123\n")

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert info.value.diagnostic.message == (
        _NODE_OR_VALUE.replace("node 'x'", "node 'api_key'")
        + ", e.g. `api_key: !str sk-live-123`"
    )
    assert _error_line(info.value) == _line_of(text, "api_key:")


@pytest.mark.parametrize(
    ("entry", "node_id"),
    [
        ("  x: call(f)\n", "x"),
        ("  api_key: call(f)\n", "api_key"),
        ("  x: ${step.output}\n", "x"),
    ],
)
def test_an_untagged_flow_call_or_span_gets_no_example(entry, node_id):
    text = _flow(entry)

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert info.value.diagnostic.message == _NODE_OR_VALUE.replace(
        "node 'x'", f"node {node_id!r}"
    )
    assert _error_line(info.value) == _line_of(text, f"  {node_id}:")


@pytest.mark.parametrize("entry", ["  input: !int 5\n", "  input:\n    kind: code\n"])
def test_reserved_ids_apply_to_a_tagged_entry(entry):
    message = (
        "node id 'input' is reserved (used as `${input.X}` as a resolver head — pick "
        "a different node id)"
    )

    with pytest.raises(LoadError, match=f"^{re.escape(message)}$"):
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
    message = "node 'x' is a tagged value entry; parse_nodes needs read_value to read it"

    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        parse_nodes(_nodes_cursor("nodes:\n  x: !int 5\n"))
