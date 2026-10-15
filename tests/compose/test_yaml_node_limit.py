"""A YAML document that expands past ``MAX_YAML_NODES`` nodes is a located load error.

Every mapping, list, and scalar node (mapping keys included) counts once per place it
appears, so an alias, and a ``<<:`` merge source, counts every node it names again.
The loader counts the composed document before anything constructs or walks it and
reports the first node, in post-order, whose expanded count exceeds the limit: a
document whose aliases double at each level fails fast instead of taking exponential
time, while a recursive alias keeps its own error.
"""

from __future__ import annotations

import random
import time

import pytest
import yaml

from yian.compose import LoadError, load_flow
from yian.compose.source import SourceDocument, SourceSyntaxError, parse_yaml_literal
from yian.compose.source.literals import read_plain_scalar
from yian.compose.source.scalars import _FlowLoader
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


def _assert_no_recursion_error(error: BaseException) -> None:
    """Assert that no exception in `error`'s cause chain is a `RecursionError`."""

    cause = error.__cause__
    while cause is not None:
        assert not isinstance(cause, RecursionError)
        cause = cause.__cause__


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
    _assert_no_recursion_error(caught.value)


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


def test_a_count_a_recursive_alias_cut_short_is_not_reused_outside_the_cycle() -> None:
    # Inside r, s's aliases of r are recursive and count 1 each, but t0 reaches s from
    # outside r, where each *r expands r's 12,287-node a12 again: t0 holds over 98,000
    # nodes. With the anchors before it, the count passes the limit inside t0, so the
    # error names the deepest node on the path already over it, the x-anchors list.
    lines = [
        "  - &r [*a12, &s [*r, *r, *r, *r]]",
        "  - &t0 [*s, *s]",
        *(f"  - &t{j} [*t{j - 1}, *t{j - 1}]" for j in range(1, 30)),
    ]
    text = _flow("*t29", _doubling_anchors(13) + "\n".join(lines) + "\n")
    assert text.splitlines()[3] == "  - &a0 [1]"

    started = time.monotonic()
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert f"this list {_message(MAX_YAML_NODES)}" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == 4
    _assert_no_recursion_error(caught.value)


def test_a_node_on_its_own_cycle_is_counted_again_inside_it() -> None:
    # M is on its own cycle, so each *M inside D counts 1 even after M's first visit:
    # the block holds about 1,200 nodes, not 400 copies of M's 402.
    anchors = "x-anchors:\n  - &M [&D [" + ", ".join(["*M"] * 400) + "]]\n  - [*D]\n"

    SourceDocument.parse(anchors)
    load_flow(_flow("hi", anchors))


@pytest.mark.parametrize(
    ("items", "where"),
    [
        # Each s<i> depends on r, so the walk cannot memoize it; s<i> doubles s<i-1>.
        (
            ["&s0 [*r, *r]", *(f"&s{i} [*s{i - 1}, *s{i - 1}]" for i in range(1, 30))],
            ("mapping", 1),
        ),
        # u's thousand recursive aliases each count, so recounting u stays bounded.
        (
            [
                "&u [" + ", ".join(["*r"] * 1000) + "]",
                "&s1 [*u, *u]",
                *(f"&s{i} [*s{i - 1}, *s{i - 1}]" for i in range(2, 30)),
            ],
            ("list", 3),
        ),
        # s15 depends on r and holds about 65,000 nodes; r lists it 300 more times,
        # bare and wrapped, so a walk that recounted each would take minutes.
        (
            [
                "&s0 [*r]",
                *(f"&s{i} [*s{i - 1}, *s{i - 1}]" for i in range(1, 16)),
                *(["*s15", "[*s15]"] * 300),
            ],
            ("mapping", 1),
        ),
    ],
    ids=["doubling", "wide", "repeated"],
)
def test_exponential_nodes_inside_a_cycle_fail_fast(
    items: list[str], where: tuple[str, int]
) -> None:
    # Once the document has counted past the limit, a node a recursive alias made
    # path-dependent is not counted again: the error names the deepest node on the
    # path already over the limit, which here is often the document itself.
    kind, line = where
    text = _flow("*r", f"x-anchors: [&r [{', '.join(items)}]]\n")

    started = time.monotonic()
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert f"this {kind} {_message(MAX_YAML_NODES)}" in str(caught.value)
    assert caught.value.diagnostic.frames[-1].range.start_line == line
    _assert_no_recursion_error(caught.value)


@pytest.mark.parametrize("value", ["&x [*x, *x, [*x]]", "&m {<<: *m}"])
def test_a_pure_cycle_ends(value: str) -> None:
    started = time.monotonic()
    try:
        load_flow(_flow(value))
    except LoadError as error:
        assert "cannot descend through a recursive alias occurrence" in str(error)

    assert time.monotonic() - started < 5


def _expanded_count(root: yaml.Node) -> int:
    """Return how many nodes `root` expands to, counting a recursive alias as 1."""

    def count(node: yaml.Node, path: set[int]) -> int:
        if id(node) in path:
            return 1
        path.add(id(node))
        if isinstance(node, yaml.SequenceNode):
            children = node.value
        elif isinstance(node, yaml.MappingNode):
            children = [child for pair in node.value for child in pair]
        else:
            children = []
        total = 1 + sum(count(child, path) for child in children)
        path.discard(id(node))
        return total

    return count(root, set())


def _random_alias_graph(rng: random.Random, recursive: bool) -> str:
    """Return a flow list of nested anchored lists and aliases, recursive if asked."""

    open_names: list[str] = []
    done_names: list[str] = []

    def value(depth: int) -> str:
        roll = rng.random()
        pool = done_names + (open_names if recursive else [])
        if pool and roll < 0.45:
            return "*" + rng.choice(pool)
        if depth > 4 or roll < 0.6:
            return "1"
        name = f"a{len(open_names) + len(done_names)}"
        open_names.append(name)
        items = [value(depth + 1) for _ in range(rng.randint(1, 4))]
        open_names.remove(name)
        done_names.append(name)
        return f"&{name} [{', '.join(items)}]"

    return "[" + ", ".join(value(0) for _ in range(rng.randint(2, 6))) + "]"


def _over_node_limit(text: str) -> bool:
    """Return whether composing `text` meets the node limit."""

    loader = _FlowLoader(text)
    try:
        loader.get_single_node()
    except yaml.composer.ComposerError as error:
        assert "YAML nodes" in str(error)
        return True
    finally:
        loader.dispose()
    return False


def test_the_count_matches_a_full_expansion(set_defaults) -> None:
    # Below the limit the count is exact, recursive aliases or not: a document loads
    # at a limit of its full expansion and fails one below it.
    rng = random.Random(646)
    for index in range(300):
        text = _random_alias_graph(rng, recursive=index % 2 == 1)
        expanded = _expanded_count(yaml.compose(text, Loader=yaml.SafeLoader))
        if expanded > 5000:
            continue

        set_defaults.unchecked(max_yaml_nodes=expanded - 1)
        assert _over_node_limit(text), text
        set_defaults.unchecked(max_yaml_nodes=expanded)
        assert not _over_node_limit(text), text


def test_a_plain_20000_element_list_loads() -> None:
    value = SourceDocument.parse("[" + ", ".join(["1"] * 20_000) + "]").root.value

    assert len(value) == 20_000


def test_reading_a_plain_scalar_is_unaffected(set_defaults) -> None:
    set_defaults.unchecked(max_yaml_nodes=0)

    assert read_plain_scalar("1e3") == 1000.0
    assert read_plain_scalar("~") is None
