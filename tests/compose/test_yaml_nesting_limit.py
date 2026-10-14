"""YAML nested past the loader's limit is a located error, never a RecursionError."""

from __future__ import annotations

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.source import SourceDocument, SourceSyntaxError, parse_yaml_literal
from yian.compose.source.scalars import MAX_YAML_NESTING
from yian.diagnostics import SourceRange
from yian.typesys.values import MAX_VALUE_NESTING

_MESSAGE = f"nests deeper than {MAX_YAML_NESTING} levels, the limit for a YAML document"


def _flow(topic: str) -> str:
    """Build a one-node flow whose `topic` input is written out as `topic`."""

    return (
        "id: f\nname: f\nnodes:\n  use:\n    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n      topic: " + topic + "\n    output: str\noutput: ${use.output}\n"
    )


def test_limit_is_twice_the_value_limit_and_the_message_names_both() -> None:
    assert MAX_YAML_NESTING == 2 * MAX_VALUE_NESTING
    depth = MAX_YAML_NESTING + 1
    with pytest.raises(SourceSyntaxError) as caught:
        SourceDocument.parse("[" * depth + "]" * depth)

    assert f"this list nests deeper than {MAX_YAML_NESTING} levels" in str(caught.value)
    assert f"a yian value nests at most {MAX_VALUE_NESTING})" in str(caught.value)


def test_flow_nested_600_levels_is_a_located_load_error() -> None:
    """A 600-level flow fails at the first list past the limit, not with RecursionError."""

    with pytest.raises(LoadError, match=_MESSAGE) as caught:
        load_flow(_flow("[" * 600 + "1" + "]" * 600))

    frame_range = caught.value.diagnostic.frames[-1].range
    # `topic: ` ends at column 13; the root, nodes, use, and input mappings are levels
    # 1-4, so the 197th `[` (column 14 + 196) is level 201.
    assert (frame_range.start_line, frame_range.start_col) == (8, 210)


@pytest.mark.parametrize(
    "topic",
    ["[" * 196 + "1" + "]" * 196, "{a: " * 196 + "1" + "}" * 196],
    ids=["list", "mapping"],
)
def test_flow_with_a_value_at_the_document_limit_parses(topic: str) -> None:
    """A value reaching document level 200 parses; it then fails the value limit."""
    with pytest.raises(LoadError, match="the limit for a yian value") as caught:
        load_flow(_flow(topic))

    assert "invalid YAML" not in str(caught.value)


@pytest.mark.parametrize(
    ("opener", "closer", "kind"),
    [("[", "]", "list"), ("{a: ", "}", "mapping")],
    ids=["flow-list", "flow-mapping"],
)
def test_flow_collections_past_the_limit_are_located(opener: str, closer: str, kind: str) -> None:
    depth = MAX_YAML_NESTING + 1
    with pytest.raises(SourceSyntaxError, match=f"this {kind} {_MESSAGE}") as caught:
        SourceDocument.parse(opener * depth + "1" + closer * depth)

    column = 1 + len(opener) * MAX_YAML_NESTING
    assert caught.value.range == SourceRange(1, column, 1, column + 1)


@pytest.mark.parametrize(
    ("line", "last", "kind"),
    [("a:", "b: 1", "mapping"), ("-", "- 1", "list")],
    ids=["block-mapping", "block-list"],
)
def test_block_collections_past_the_limit_are_located(line: str, last: str, kind: str) -> None:
    depth = MAX_YAML_NESTING
    text = "".join("  " * i + line + "\n" for i in range(depth)) + "  " * depth + last + "\n"

    with pytest.raises(SourceSyntaxError, match=f"this {kind} {_MESSAGE}") as caught:
        SourceDocument.parse(text)

    assert caught.value.range == SourceRange(depth + 1, 2 * depth + 1, depth + 1, 2 * depth + 2)


def test_document_at_the_limit_parses() -> None:
    depth = MAX_YAML_NESTING
    value = SourceDocument.parse("[" * depth + "]" * depth).root.value
    for _ in range(depth - 1):
        (value,) = value
    assert value == []


def test_sibling_collections_do_not_add_up() -> None:
    deep = "[" * (MAX_YAML_NESTING - 1) + "]" * (MAX_YAML_NESTING - 1)
    SourceDocument.parse("[" + ", ".join([deep] * 3) + "]")
    SourceDocument.parse("[" + ", ".join(["[1]"] * 5000) + "]")


def test_isolated_literal_past_the_limit_is_a_syntax_error() -> None:
    with pytest.raises(SourceSyntaxError, match=_MESSAGE):
        parse_yaml_literal("[" * 600 + "]" * 600)
