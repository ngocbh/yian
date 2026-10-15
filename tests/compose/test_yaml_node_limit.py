"""A YAML document that expands past ``MAX_YAML_NODES`` nodes is a located load error.

Every mapping, list, and scalar node (mapping keys included) counts once per place it
appears, so an alias, and a ``<<:`` merge source, counts every node it names again.
The loader counts the composed document before anything constructs or walks it and
reports the first node, in post-order, whose expanded count exceeds the limit: a
document whose aliases double at each level fails fast instead of taking exponential
time, while a recursive alias keeps its own error.
"""

from __future__ import annotations

import time

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.source import SourceDocument, SourceSyntaxError, parse_yaml_literal
from yian.compose.source.literals import read_plain_scalar
from yian.defaults import Defaults

MAX_YAML_NODES = Defaults().max_yaml_nodes


def _message(limit: int) -> str:
    """Return the start of the node-limit message for ``limit``."""

    return f"holds more than {limit} YAML nodes"


def _flow(topic: str, header: str = "") -> str:
    """Return a one-node flow whose `topic` input is `topic`, after a top-level `header`."""

    return (
        f"id: f\nname: f\n{header}nodes:\n  use:\n    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        f"    input:\n      topic: {topic}\n    output: str\noutput: ${{use.output}}\n"
    )


def _doubling_anchors(levels: int) -> str:
    """Return an `x-anchors:` block list, one anchor per line, each twice the last.

    Line 4 holds `a0`, so `a<i>` is on line ``4 + i`` of a flow built by `_flow`.
    """

    lines = ["x-anchors:", "  - &a0 [1]"]
    lines += [f"  - &a{i} [*a{i - 1}, *a{i - 1}]" for i in range(1, levels)]
    return "\n".join(lines) + "\n"


def test_doubling_aliases_fail_fast_at_the_first_anchor_over_the_limit() -> None:
    # a<i> holds 3 * 2**i - 1 nodes: a15 (98,303) fits and a16 (196,607) does not.
    assert 3 * 2**15 - 1 <= MAX_YAML_NODES < 3 * 2**16 - 1
    text = _flow("*a29", _doubling_anchors(30))
    assert text.splitlines()[3 + 16] == "  - &a16 [*a15, *a15]"

    started = time.monotonic()
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert f"this list {_message(MAX_YAML_NODES)}" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == 4 + 16
    cause = caught.value.__cause__
    while cause is not None:
        assert not isinstance(cause, RecursionError)
        cause = cause.__cause__


def test_message_names_the_variable_and_what_a_node_is(set_defaults) -> None:
    set_defaults(MAX_YAML_NODES=50)

    with pytest.raises(SourceSyntaxError) as caught:
        SourceDocument.parse("{a: [" + ", ".join(["1"] * 50) + "]}")

    assert (
        "this list holds more than 50 YAML nodes, counting each alias as a copy of what "
        "it names; the limit for a YAML document is 50 (YIAN_MAX_YAML_NODES; each "
        "mapping, list, and scalar is a node)"
    ) in str(caught.value)


def test_exact_boundary(set_defaults) -> None:
    set_defaults(MAX_YAML_NODES=50)

    # One list and 49 scalars are 50 nodes.
    assert SourceDocument.parse("[" + ", ".join(["1"] * 49) + "]").root.value == [1] * 49
    with pytest.raises(SourceSyntaxError, match=_message(50)):
        SourceDocument.parse("[" + ", ".join(["1"] * 50) + "]")


def test_mapping_keys_count(set_defaults) -> None:
    set_defaults(MAX_YAML_NODES=50)

    # One mapping, 24 keys, and 24 values are 49 nodes; 25 pairs are 51.
    SourceDocument.parse("{" + ", ".join(f"k{i}: 1" for i in range(24)) + "}")
    with pytest.raises(SourceSyntaxError, match=f"this mapping {_message(50)}"):
        SourceDocument.parse("{" + ", ".join(f"k{i}: 1" for i in range(25)) + "}")


def test_each_alias_counts_as_a_copy(set_defaults) -> None:
    set_defaults(MAX_YAML_NODES=70)
    ten = "[" + ", ".join(["1"] * 10) + "]"

    def aliased(copies: int) -> str:
        return f"x: &t {ten}\ny: [" + ", ".join(["*t"] * copies) + "]\n"

    # The root, x, the 11-node list, y, and y's list are 15 nodes; each alias adds 11.
    SourceDocument.parse(aliased(0))
    SourceDocument.parse(aliased(5))
    with pytest.raises(SourceSyntaxError, match=_message(70)):
        SourceDocument.parse(aliased(6))


def test_a_merge_source_counts_in_full(set_defaults) -> None:
    set_defaults(MAX_YAML_NODES=1000)
    lines = ["m0: &m0 {k0: 1}"]
    lines += [f"m{i}: &m{i} {{<<: [*m{i - 1}, *m{i - 1}], k{i}: 1}}" for i in range(1, 8)]

    # m<i> holds 2 * m<i-1> + 5 nodes, so m6 holds 507 and m7's merge list
    # [*m6, *m6] holds 1015, although merged m7 has only 8 keys.
    with pytest.raises(SourceSyntaxError, match=f"this list {_message(1000)}") as caught:
        SourceDocument.parse("\n".join(lines) + "\n")
    assert caught.value.range.start_line == 8
    merged = SourceDocument.parse("\n".join(lines[:5]) + "\n").root.value["m4"]
    assert merged == {f"k{i}": 1 for i in range(5)}


def test_an_isolated_literal_past_the_limit_is_a_syntax_error() -> None:
    items = ", ".join(f"&a{i} [*a{i - 1}, *a{i - 1}]" if i else "&a0 [1]" for i in range(30))

    with pytest.raises(SourceSyntaxError, match=_message(MAX_YAML_NODES)):
        parse_yaml_literal(f"[{items}]")


def _linear_anchors(count: int) -> str:
    """Return an `x-anchors:` line whose anchor `a<i>` names a list nesting i+1 levels."""

    items = ["&a0 [1]"] + [f"&a{i} [*a{i - 1}]" for i in range(1, count)]
    return f"x-anchors: [{', '.join(items)}]\n"


def test_a_listed_1000_link_chain_meets_the_node_limit_first() -> None:
    # a<i> holds i + 2 nodes, so the listed chain holds about 500,000.
    with pytest.raises(LoadError, match=_message(MAX_YAML_NODES)) as caught:
        load_flow(_flow("[*a998]", _linear_anchors(999)))

    assert caught.value.diagnostic.frames[-1].range.start_line == 3


def test_a_listed_1000_link_chain_under_a_raised_limit_meets_the_value_limit(
    set_defaults,
) -> None:
    set_defaults(MAX_YAML_NODES=10_000_000)

    with pytest.raises(LoadError, match="value nests deeper than 100 levels"):
        load_flow(_flow("[*a998]", _linear_anchors(999)))


def test_a_recursive_alias_keeps_its_error() -> None:
    with pytest.raises(LoadError, match="cannot descend through a recursive alias occurrence"):
        load_flow(_flow("&x [*x]"))


def test_a_plain_20000_element_list_loads() -> None:
    value = SourceDocument.parse("[" + ", ".join(["1"] * 20_000) + "]").root.value

    assert len(value) == 20_000


def test_reading_a_plain_scalar_is_unaffected(set_defaults) -> None:
    set_defaults.unchecked(max_yaml_nodes=0)

    assert read_plain_scalar("1e3") == 1000.0
    assert read_plain_scalar("~") is None
