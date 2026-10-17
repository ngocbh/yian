"""Null checks in CASE conditions, end to end: what a branch, a param and an assert read.

A searched CASE branch under `x != null` sees `x` as not null, so `x` fills a strict input
of a CALL or a MAP argument. These tests cover which conditions narrow a branch (a negated
`== null`, but not an `or` or the `else:`), joins of two cases, an `on:` CASE over an
`Optional` union, null checks meeting `isinstance` checks within and across nodes, the views
of a searched CASE's params, and the guards of reads in a node assert.
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.condition_guards import NOTHING, Holds, NotNullCheck
from yian.compose.helpers.narrowing_facts import (
    Exact,
    checked_refs,
    guard_env,
    guarded_read_types,
)
from yian.compose.helpers.types import read_type
from yian.compose.nodes.base import BuildContext, NodeDescriptor
from yian.compose.nodes.case import CaseDescriptor
from yian.compose.validate.asserts import _narrows, _root_base_type
from yian.compose.validate.narrowing import compute_narrowing
from yian.expr import parse_condition
from yian.nodes.contracts import ValueContract
from yian.typesys.types import read_typedefs

_ECHO = "tests.engine._compose_codefns:echo"
_CHILD_EXPECTS = "child expects 'str', source is 'Optional\\(str\\)'"


def _skip(node_id: str) -> str:
    """A CODE node `node_id` that outputs `0`, for a branch that passes nothing on."""
    return (
        f"  {node_id}:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: 0\n    output: int\n"
    )


def _sink(target: str) -> str:
    """The `use` node: a CALL of `take`, or a MAP of `take_each`, passing `${x.output}`."""
    if target == "call":
        return "  use:\n    kind: call\n    call: take\n    input:\n      v: ${x.output}\n"
    return (
        "  use:\n    kind: map\n    call: take_each\n    over: ${input.items}\n"
        "    input:\n      i: ${item}\n      v: ${x.output}\n"
    )


def _producer_flow(cases: list[tuple[str, str]], target: str = "call") -> str:
    """A flow whose searched CASE `gate` routes on `x`, a node with an `Optional(str)` output.

    Args:
        cases (`list[tuple[str, str]]`):
            `(when, then)` per case; `else:` goes to `skip`.
        target (`str`, *optional*, defaults to `"call"`):
            `"call"` passes `${x.output}` to a def's `str` input; `"map"` passes it as a
            MAP's per-item argument.
    """
    whens = "".join(f'      - when: "{when}"\n        then: {then}\n' for when, then in cases)
    return (
        "id: p\nname: p\n"
        "input:\n  s: !Optional(str)\n  items: !list(int)\n  y: !bool\n"
        "nodes:\n"
        f"  x:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: ${input.s}\n    output: Optional(str)\n"
        "  gate:\n    kind: case\n    cases:\n"
        f"{whens}"
        "    else: skip\n"
        f"{_sink(target)}"
        f"  skip:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: skipped\n    output: str\n"
        "output: ${use.output :- skip.output}\n"
        "defs:\n"
        "  take:\n    input:\n      v: !str\n    nodes:\n"
        f"      d:\n        kind: code\n        code: {_ECHO}\n"
        "        input:\n          topic: ${input.v}\n        output: str\n"
        "    output: ${d.output}\n"
        "  take_each:\n    input:\n      i: !int\n      v: !str\n    nodes:\n"
        f"      d:\n        kind: code\n        code: {_ECHO}\n"
        "        input:\n          topic: ${input.v}\n        output: str\n"
        "    output: ${d.output}\n"
    )


_TYPEDEFS = (
    "typedefs:\n"
    "  Item:\n    qty: int\n"
    "  Wide:\n    qty: int\n    note: Optional(str)\n"
    "  A:\n    b: Optional(str)\n"
    "  MaybeItem: Optional(Item)\n"
)


def _input_flow(when: str, x_type: str, sink: str = "Item", source: str = "${input.x}") -> str:
    """A flow whose searched CASE passes `source` to a def's `!<sink>` input under `when`.

    `Item` is `{qty: int}`; `Wide` is `{qty: int, note: Optional(str)}`, which `Item` fits;
    `A` is `{b: Optional(str)}`; `MaybeItem` is `Optional(Item)`.
    """
    return (
        f"id: p\nname: p\n{_TYPEDEFS}"
        f"input:\n  x: !{x_type}\n"
        "nodes:\n"
        "  gate:\n    kind: case\n    cases:\n"
        f'      - when: "{when}"\n        then: use\n'
        "    else: skip\n"
        f"  use:\n    kind: call\n    call: take\n    input:\n      v: {source}\n"
        f"{_skip('skip')}"
        "output: ${use.output :- skip.output}\n"
        "defs:\n"
        f"  take:\n    input:\n      v: !{sink}\n    nodes:\n"
        f"      d:\n        kind: code\n        code: {_ECHO}\n"
        "        input:\n          topic: ${input.v}\n        output: Any\n"
        "    output: ${d.output}\n"
    )


def _cross_node_flow(outer: str, inner: str, x_type: str) -> str:
    """A flow where CASE `outer` routes to CASE `inner`, which routes `${input.x}` to `!Item`."""
    return (
        f"id: p\nname: p\n{_TYPEDEFS}"
        f"input:\n  x: !{x_type}\n"
        "nodes:\n"
        "  outer:\n    kind: case\n    cases:\n"
        f'      - when: "{outer}"\n        then: inner\n'
        "    else: skip\n"
        "  inner:\n    kind: case\n    cases:\n"
        f'      - when: "{inner}"\n        then: use\n'
        "    else: skip2\n"
        "  use:\n    kind: call\n    call: take\n    input:\n      v: ${input.x}\n"
        f"{_skip('skip')}"
        f"{_skip('skip2')}"
        "output: ${use.output :- skip.output}\n"
        "defs:\n"
        "  take:\n    input:\n      v: !Item\n    nodes:\n"
        f"      d:\n        kind: code\n        code: {_ECHO}\n"
        "        input:\n          topic: ${input.v.qty}\n        output: int\n"
        "    output: ${d.output}\n"
    )


# --------------------------------------------------------------------------- #
# A branch under a null check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", ["call", "map"])
def test_a_branch_under_a_null_check_passes_the_value_as_not_null(target):
    load_flow(_producer_flow([("x.output != null", "use")], target))


@pytest.mark.parametrize(
    ("when", "x_type", "sink", "source"),
    [
        ("input.x != null", "Optional(Item)", "Item", "${input.x}"),
        ("input.x.b != null", "A", "str", "${input.x.b}"),
        ("input.x.b != null", "Optional(A)", "str", "${input.x.b}"),
        ("input.x != null", "Optional(Any)", "Any", "${input.x}"),
    ],
)
def test_a_branch_under_a_null_check_on_an_input_passes_it_as_not_null(
    when, x_type, sink, source
):
    load_flow(_input_flow(when, x_type, sink, source))


def test_a_negated_is_null_check_narrows_the_branch():
    load_flow(_producer_flow([("not (x.output == null)", "use")]))


def test_a_null_check_that_may_not_hold_does_not_narrow_the_branch():
    with pytest.raises(LoadError, match=_CHILD_EXPECTS):
        load_flow(_producer_flow([("x.output != null or input.y", "use")]))


def test_the_else_of_a_null_check_is_not_narrowed():
    text = _producer_flow([("x.output != null", "skip")]).replace("else: skip", "else: use")
    with pytest.raises(LoadError, match=_CHILD_EXPECTS):
        load_flow(text)


def test_a_read_after_an_is_null_check_in_an_or_is_guarded():
    registry = read_typedefs({})
    tree = parse_condition("x.output == null or x.output > 0")
    declared = read_type("Optional(int)", registry)

    def base_type_of(base):
        return declared if base == "x.output" else None

    (_, first), (_, second) = checked_refs(tree, base_type_of, registry)
    assert second == Holds(NotNullCheck("x.output"), True)
    assert first == NOTHING
    env = guard_env(second, base_type_of, registry)
    assert env == {"x.output": Exact(read_type("int", registry))}


# --------------------------------------------------------------------------- #
# Joins and `on:` cases
# --------------------------------------------------------------------------- #


def test_a_node_two_null_checking_cases_route_is_narrowed():
    load_flow(
        _producer_flow([("x.output != null and input.y", "use"), ("x.output != null", "use")])
    )


def test_a_node_that_only_one_of_two_cases_proves_not_null_is_not_narrowed():
    with pytest.raises(LoadError, match=_CHILD_EXPECTS):
        load_flow(_producer_flow([("x.output != null", "use"), ("input.y", "use")]))


def _on_flow(sink: str) -> str:
    """A flow whose `on:` CASE over an `Optional(Union[A, B])` tag passes the value to `!sink`."""
    return (
        "id: p\nname: p\n"
        "typedefs:\n"
        "  A:\n    kind: Literal(a)\n    n: int\n"
        "  B:\n    kind: Literal(b)\n    s: str\n"
        "  AB: Union(A, B)\n"
        "input:\n  v: !Optional(AB)\n"
        "nodes:\n"
        "  gate:\n    kind: case\n    on: ${input.v.kind}\n    cases:\n"
        "      - when: a\n        then: use\n"
        "    else: skip\n"
        "  use:\n    kind: call\n    call: take\n    input:\n      v: ${input.v}\n"
        f"{_skip('skip')}"
        "output: ${use.output :- skip.output}\n"
        "defs:\n"
        f"  take:\n    input:\n      v: !{sink}\n    nodes:\n"
        f"      d:\n        kind: code\n        code: {_ECHO}\n"
        "        input:\n          topic: 1\n        output: int\n"
        "    output: ${d.output}\n"
    )


@pytest.mark.parametrize("sink", ["A", "Optional(A)", "AB"])
def test_an_on_case_over_an_optional_union_narrows_its_branch_as_before(sink):
    load_flow(_on_flow(sink))


# --------------------------------------------------------------------------- #
# Null checks meet isinstance checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("x_type", ["Optional(Any)", "Optional(Wide)"])
@pytest.mark.parametrize(
    "when",
    [
        "input.x != null and isinstance(input.x, Item)",
        "isinstance(input.x, Item) and input.x != null",
    ],
)
def test_a_null_check_keeps_an_isinstance_narrowing_of_the_branch(when, x_type):
    load_flow(_input_flow(when, x_type))


@pytest.mark.parametrize("sink", ["Optional(Item)", "Item"])
@pytest.mark.parametrize("x_type", ["Optional(Any)", "Optional(Wide)"])
@pytest.mark.parametrize(
    "when",
    [
        "input.x != null and isinstance(input.x, MaybeItem)",
        "isinstance(input.x, MaybeItem) and input.x != null",
    ],
)
def test_a_null_check_keeps_a_nullable_isinstance_narrowing_of_the_branch(when, x_type, sink):
    load_flow(_input_flow(when, x_type, sink))


@pytest.mark.parametrize("x_type", ["Optional(Any)", "Optional(Wide)"])
def test_a_null_check_case_under_an_isinstance_branch_keeps_its_narrowing(x_type):
    load_flow(_cross_node_flow("isinstance(input.x, Item)", "input.x != null", x_type))


def test_a_null_check_alone_does_not_prove_a_type():
    with pytest.raises(LoadError, match="child expects '\\{\"qty\":int\\}', source is 'Any'"):
        load_flow(_cross_node_flow("input.x != null", "input.x != null", "Optional(Any)"))


# --------------------------------------------------------------------------- #
# Param views
# --------------------------------------------------------------------------- #


_PARAM_REGISTRY = read_typedefs({"Rec": {"a": "str", "b": "str"}})


def test_a_case_param_read_after_a_null_check_sees_the_value_as_not_null():
    rec = read_type("Optional(Rec)", _PARAM_REGISTRY)
    gate = CaseDescriptor(
        id="gate",
        on=None,
        cases=[
            {
                "when": "x.output.a == 'p' and x.output != null and x.output.b == 'q'",
                "then": "use",
            }
        ],
        else_=None,
    )
    descriptors = {"x": NodeDescriptor(id="x"), "gate": gate, "use": NodeDescriptor(id="use")}
    edges = list(gate.build(BuildContext({})).edges)
    narrowing = compute_narrowing(
        descriptors, edges, {"x": ValueContract(rec)}, {}, registry=_PARAM_REGISTRY
    )
    # The params follow the reads: `x.output.a`, `x.output`, `x.output.b`.
    before, after = "__r0", "__r2"
    assert narrowing.for_param("gate", after)[0]["x"].type.nullable is False
    assert narrowing.for_param("gate", before)[0]["x"].type == rec
    not_null = Exact(read_type("Rec", _PARAM_REGISTRY))
    assert narrowing.param_env("gate", after) == {"x.output": not_null}
    assert narrowing.param_env("gate", before) == {}


# --------------------------------------------------------------------------- #
# Node asserts
# --------------------------------------------------------------------------- #


def test_an_assert_read_after_a_null_check_is_guarded_by_it():
    registry = read_typedefs({"Item": {"qty": "int"}})
    contract = ValueContract(read_type("Optional(Item)", registry))
    check = NotNullCheck("output")
    assert _narrows(check, "output", (contract,))
    assert not _narrows(check, "output", (ValueContract(read_type("Item", registry)),))
    # A root that may be skipped is nullable whatever its contract says.
    assert _narrows(
        check, "output", (ValueContract(read_type("Item", registry)),), nullable=True
    )

    base_type_of = _root_base_type({"output": contract})
    tree = parse_condition("output.qty > 0 or output != null and output.qty > 1")
    (unguarded, _), _check, (guarded, guard) = checked_refs(tree, base_type_of, registry)
    assert unguarded == guarded == "output.qty"
    assert guard == Holds(check, True)
    ((narrowed, fields, base),) = guarded_read_types(guarded, guard, base_type_of, registry)
    assert narrowed == read_type("Item", registry)
    assert (fields, base) == (("qty",), "output")


def test_a_node_assert_read_after_a_null_check_names_the_non_null_type():
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Item:\n    qty: int\n"
        "input:\n  s: !Optional(str)\n"
        "nodes:\n"
        f"  x:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: ${input.s}\n    output: Optional(Item)\n"
        '    asserts:\n      - "output != null and output.nosuch > 0"\n'
        "output: ${x.output}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "node 'x' assert: reference ${output.nosuch} reads field 'nosuch', which Item doesn't "
        "have; output is Item here, by the checks before this read"
    ) in str(caught.value)


def test_a_flow_assert_null_check_narrows_a_skippable_producer_of_a_non_null_type():
    # `d` declares `Item`, but it may be skipped, so the assert can read it as null.
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Item:\n    qty: int\n"
        "input:\n  s: !str\n"
        "nodes:\n"
        "  route:\n    kind: case\n    cases:\n"
        "      - when: \"input.s == 'go'\"\n        then: d\n"
        "    else: e\n"
        f"  d:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: ${input.s}\n    output: Item\n"
        f"  e:\n    kind: code\n    code: {_ECHO}\n"
        "    input:\n      topic: ${input.s}\n    output: str\n"
        "output: ${input.s}\n"
        'asserts:\n  - "d.output != null and d.output.nosuch > 0"\n'
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert (
        "assert 'd.output != null and d.output.nosuch > 0': reference ${d.output.nosuch} reads "
        "field 'nosuch', which Item doesn't have; d.output is Item here, by the checks before "
        "this read"
    ) in str(caught.value)
