"""Union narrowing from CASE branches: the fact domain, propagation, and node views."""

from dataclasses import replace

import pytest

from yian.compile.model import Edge
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.types import read_type
from yian.compose.nodes.base import BuildContext, NodeDescriptor
from yian.compose.nodes.case import (
    CaseDescriptor,
    isinstance_conjuncts,
    searched_param_guards,
)
from yian.compose.validate.narrowing import (
    Exact,
    Members,
    compute_narrowing,
    intersect_facts,
    isinstance_fact,
    narrow_contract_at,
    union_facts,
)
from yian.diagnostics import OriginRef, SourceRange
from yian.nodes.contracts import ContractOrigins, PartialRecordContract, ValueContract
from yian.typesys import Type, ValueKind
from yian.typesys.types import read_typedefs

_REGISTRY = read_typedefs(
    {
        "Limit": {"type": "Literal[limit]", "price": "float"},
        "Market": {"type": "Literal[market]"},
        "Leg": "Union[Limit, Market]",
        "Buy": {"action": "Literal[buy]", "qty": "int", "leg": "Leg"},
        "Hold": {"action": "Literal[hold]"},
        "Sell": {"action": "Literal[sell]", "qty": "int", "price": "float"},
        "Decision": "Union[Buy, Hold, Sell]",
        "BuyOrSell": "Union[Buy, Sell]",
        "Text": {"text": "str"},
    }
)


def _t(name: str) -> Type:
    """Resolve one type expression against the test typedefs."""
    return read_type(name, _REGISTRY)


BUY, HOLD, SELL = _t("Buy"), _t("Hold"), _t("Sell")
DECISION = _t("Decision")
BUY_OR_SELL = _t("BuyOrSell")
LIMIT = _t("Limit")
TEXT = _t("Text")

_ROOT = OriginRef("/s:root/s:node/s:d/s:field/s:output")
_QTY = OriginRef("/s:root/s:node/s:d/s:field/s:output/s:qty")


def _node(node_id: str, *, depends_on=(), runs_after=()) -> NodeDescriptor:
    """A plain node descriptor with optional ordering lists."""
    return NodeDescriptor(
        id=node_id, depends_on=list(depends_on), runs_after=list(runs_after)
    )


def _case(
    case_id: str,
    on: str | None,
    cases: list[tuple[str, str]],
    else_: str | None = None,
    *,
    when_ranges=(),
) -> tuple[CaseDescriptor, list[Edge]]:
    """A CASE descriptor and the data and control edges its own build returns."""
    desc = CaseDescriptor(
        id=case_id,
        on=on,
        cases=[{"when": when, "then": then} for when, then in cases],
        else_=else_,
        when_ranges=list(when_ranges),
    )
    return desc, list(desc.build(BuildContext({})).edges)


def _data(src: str, dst: str, group: str = "x", *, optional: bool = False) -> Edge:
    """One data edge from `src`'s output into `dst`'s input group."""
    return Edge(
        id=f"{src}->{dst}:{group}",
        from_=src,
        to=dst,
        input_group=group,
        optional=optional,
    )


def _compute(nodes, edges, producers, inputs=None, *, diagnostics=None):
    """Run the narrowing pass over hand-built nodes and edges."""
    descriptors = {}
    all_edges = list(edges)
    for item in nodes:
        if isinstance(item, tuple):
            desc, case_edges = item
            all_edges.extend(case_edges)
        else:
            desc = item
        descriptors[desc.id] = desc
    return compute_narrowing(
        descriptors,
        all_edges,
        producers,
        inputs or {},
        registry=_REGISTRY,
        diagnostics=diagnostics,
    )


def _decision_contract(typ: Type = DECISION) -> ValueContract:
    """An exact output contract for `d` with a root and a nested origin."""
    return ValueContract(
        type=typ,
        origins=ContractOrigins(default=_ROOT, entries=((("qty",), _QTY),)),
    )


def _view_type(narrowing, node_id: str, producer: str = "d") -> Type:
    """The type node `node_id` sees for `producer`'s output."""
    producers, _inputs = narrowing.for_node(node_id)
    return producers[producer].type


def _branches():
    """`d` (a Decision) routed by `c` on its tag: buy -> b, sell -> s, else -> h."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "s")], "h"),
        _node("b"),
        _node("s"),
        _node("h"),
    ]
    return nodes


# --------------------------------------------------------------------------- #
# The fact domain
# --------------------------------------------------------------------------- #


def test_union_of_members_is_the_set_union_and_or_of_nullability():
    """Two member sets join to their union; either nullable makes it nullable."""
    joined = union_facts(Members((BUY,)), Members((SELL, BUY), nullable=True))
    assert joined == Members((BUY, SELL), nullable=True)


def test_union_of_exact_facts_keeps_only_an_equal_type():
    """Equal exact types survive a union; anything else loses the fact."""
    assert union_facts(Exact(TEXT), Exact(TEXT)) == Exact(TEXT)
    assert union_facts(Exact(TEXT), Exact(BUY)) is None
    assert union_facts(Exact(BUY), Members((BUY,))) is None
    assert union_facts(Members((BUY,)), None) is None


def test_intersection_of_members_is_the_set_intersection():
    """Two member sets meet at their common members; both nullable stays nullable."""
    met = intersect_facts(
        Members((BUY, SELL), nullable=True), Members((SELL, HOLD), nullable=True)
    )
    assert met == Members((SELL,), nullable=True)
    assert intersect_facts(Members((BUY,)), Members((SELL,))) == Members(())
    assert intersect_facts(Members((BUY,)), None) == Members((BUY,))


def test_intersection_of_exact_with_members():
    """An exact member picks that member; an unrelated exact type stays exact."""
    assert intersect_facts(Exact(BUY), Members((BUY, SELL))) == Members((BUY,))
    assert intersect_facts(Members((BUY, SELL)), Exact(TEXT)) == Exact(TEXT)
    assert intersect_facts(Exact(TEXT), Exact(TEXT)) == Exact(TEXT)
    assert intersect_facts(Exact(TEXT), Exact(BUY)) is None


def test_isinstance_fact_against_the_base_union():
    """A member or a sub-union gives members; any other type gives an exact fact."""
    assert isinstance_fact(DECISION, BUY) == Members((BUY,))
    assert isinstance_fact(DECISION, BUY_OR_SELL) == Members((BUY, SELL))
    assert isinstance_fact(DECISION, TEXT) == Exact(TEXT)
    assert isinstance_fact(TEXT, TEXT) is None
    assert isinstance_fact(DECISION, Type.scalar(ValueKind.OBJECT)) is None
    assert isinstance_fact(DECISION, DECISION) is None
    assert isinstance_fact(replace(DECISION, nullable=True), BUY) == Members((BUY,))
    assert isinstance_fact(Type.scalar(ValueKind.OBJECT), TEXT) == Exact(TEXT)
    assert isinstance_fact(DECISION, replace(BUY, nullable=True)) == Members(
        (BUY,), nullable=True
    )


# --------------------------------------------------------------------------- #
# isinstance checks in a searched CASE
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("when", "checks"),
    [
        ("isinstance(d.output, Buy)", [("d.output", "Buy")]),
        ("x > 1 and isinstance(${d.output.leg}, Limit)", [("d.output.leg", "Limit")]),
        ("${isinstance(input.d, Buy) and isinstance(e.output, Text)}",
         [("input.d", "Buy"), ("e.output", "Text")]),
        ("isinstance(d.output, Buy) or x", []),
        ("not isinstance(d.output, Buy)", []),
        ("isinstance(upper(d.output), Buy)", []),
        ("isinstance(d.output.items[0], Buy)", []),
        ("isinstance(", []),
        (3, []),
    ],
)
def test_isinstance_conjuncts_reads_top_level_and_operands(when, checks):
    assert isinstance_conjuncts(when) == checks


def test_searched_param_guards_lists_the_checks_before_each_read():
    desc, _ = _case(
        "c",
        None,
        [
            ("isinstance(d.output, Buy) and d.output.qty > 5", "b"),
            ("d.output.qty > 1 and isinstance(d.output, Sell)", "s"),
        ],
    )
    assert searched_param_guards(desc) == {
        "__r0": [(0, ()), (1, ())],
        "__r1": [(0, (("d.output", "Buy"),)), (1, ())],
    }
    on_desc, _ = _case("o", "${d.output.action}", [("buy", "b")])
    assert searched_param_guards(on_desc) == {}


def _searched_view(cases, else_=None, typ: Type = DECISION, target: str = "b") -> Type:
    """The type `target` sees for `d` under a searched CASE `c` over `cases`."""
    nodes = [_node("d"), _case("c", None, cases, else_), _node("b"), _node("s")]
    narrowing = _compute(nodes, [], {"d": ValueContract(typ)})
    return _view_type(narrowing, target)


def test_an_isinstance_case_narrows_its_target_to_the_member_or_sub_union():
    assert _searched_view([("isinstance(d.output, Buy)", "b")]) == BUY
    assert _searched_view([("isinstance(d.output, BuyOrSell) and x", "b")]) == BUY_OR_SELL


def test_an_isinstance_case_on_a_non_union_base_gives_the_exact_type():
    obj = Type.scalar(ValueKind.OBJECT)
    nodes = [_node("d"), _case("c", None, [("isinstance(d.output, Text)", "b")]), _node("b")]
    narrowing = _compute(nodes, [], {"d": ValueContract(obj)})
    assert narrowing.environment("b") == {"d.output": Exact(TEXT)}
    assert _view_type(narrowing, "b") == TEXT


def test_isinstance_facts_use_the_original_base_union():
    """Under an outer `buy` branch, an inner `isinstance(d.output, Buy)` still names a member."""
    nodes = [
        _node("d"),
        _case("o", "${d.output.action}", [("buy", "c")], "h"),
        _case("c", None, [("isinstance(d.output, Buy)", "b")]),
        _node("b"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_an_isinstance_of_a_type_the_base_fits_gives_no_fact():
    nodes = [
        _node("d"),
        _case("c", None, [("isinstance(d.output, Decision) and isinstance(d.output, Buy)", "b")]),
        _node("b"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_two_isinstance_cases_into_one_target_join_by_union():
    cases = [("isinstance(d.output, Buy)", "b"), ("isinstance(d.output, Sell)", "b")]
    assert _searched_view(cases) == BUY_OR_SELL


def test_two_isinstance_cases_on_unrelated_types_into_one_target_drop_the_fact():
    obj = ValueContract(Type.scalar(ValueKind.OBJECT))
    cases = [("isinstance(d.output, Text)", "b"), ("isinstance(d.output, Buy)", "b")]
    nodes = [_node("d"), _case("c", None, cases), _node("b")]
    narrowing = _compute(nodes, [], {"d": obj})
    assert narrowing.environment("b") == {}
    assert narrowing.for_node("b")[0]["d"] is obj


def test_a_plain_case_or_else_into_the_target_drops_the_fact():
    nodes = [
        _node("d"),
        _case("c", None, [("isinstance(d.output, Buy)", "b"), ("x > 1", "b")], "s"),
        _node("b"),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {}
    assert narrowing.environment("s") == {}


def test_an_unknown_isinstance_type_gives_no_fact():
    nodes = [_node("d"), _case("c", None, [("isinstance(d.output, Nope)", "b")]), _node("b")]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {}


def test_a_guarded_case_param_gets_the_checked_view():
    nodes = [
        _node("d"),
        _case(
            "c",
            None,
            [
                ("isinstance(d.output, Buy) and d.output.qty > 5", "b"),
                ("isinstance(d.output, Sell) and d.output.qty > 1", "s"),
                ("d.output.action == 'hold' and isinstance(d.output, Hold)", "s"),
            ],
        ),
        _node("b"),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.for_param("c", "__r1")[0]["d"].type == BUY_OR_SELL
    # `d.output` and `d.output.action` are read unguarded: they see the CASE's own view.
    assert narrowing.for_param("c", "__r0")[0]["d"].type == DECISION
    assert narrowing.for_param("c", "__r2")[0]["d"].type == DECISION


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


def test_on_case_narrows_each_branch_and_else_to_the_unlisted_members():
    """Each `when:` branch sees its member; `else:` sees the rest."""
    producers = {"d": _decision_contract()}
    narrowing = _compute(_branches(), [], producers)

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "s") == SELL
    assert _view_type(narrowing, "h") == HOLD
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_views_keep_origins_and_unnarrowed_nodes_get_the_same_maps():
    """A narrowed view keeps the producer's origins; other nodes see the inputs as-is."""
    producers = {"d": _decision_contract()}
    inputs = {"s": ValueContract(Type.scalar(ValueKind.STRING))}
    narrowing = _compute(_branches(), [], producers, inputs)

    view, view_inputs = narrowing.for_node("b")
    assert view["d"].origins == producers["d"].origins
    assert view_inputs is inputs
    plain_producers, plain_inputs = narrowing.for_node("d")
    assert plain_producers is producers and plain_inputs is inputs
    assert narrowing.for_node("c") == (producers, inputs)


def test_else_keeps_the_base_nullability_and_when_branches_drop_it():
    """A nullable base stays nullable under `else:` only."""
    producers = {"d": _decision_contract(replace(DECISION, nullable=True))}
    narrowing = _compute(_branches(), [], producers)

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "h") == replace(HOLD, nullable=True)


def test_a_when_branch_and_else_into_one_target_join_to_every_member():
    """Routing `buy` and `else:` to one node leaves every member possible."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "x")], "x"),
        _node("x"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("x") == {"d.output": Members((BUY, HOLD, SELL))}
    assert narrowing.for_node("x")[0]["d"].type == DECISION


def test_a_then_target_named_default_carries_no_fact():
    """A `then: default` node shares the else handle, so it is not narrowed."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "default")]),
        _node("b"),
        _node("default"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert narrowing.environment("default") == {}


def test_a_then_target_named_default_and_an_else_carry_no_fact():
    """With `then: default` the else target shares the handle and is not narrowed either."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "default")], "h"),
        _node("b"),
        _node("default"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert narrowing.environment("default") == {}
    assert narrowing.environment("h") == {}


def test_else_keeps_nullability_from_a_nullable_path_segment():
    """A nullable record on the way to the base makes the `else:` members nullable."""
    holder = Type(
        kind=ValueKind.OBJECT,
        fields={"dec": DECISION},
        required=frozenset({"dec"}),
        nullable=True,
    )
    nodes = [
        _node("d"),
        _case("c", "${d.output.dec.action}", [("buy", "b")], "h"),
        _node("b"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(holder)})

    assert narrowing.environment("h") == {
        "d.output.dec": Members((HOLD, SELL), nullable=True)
    }
    assert _view_type(narrowing, "h").fields["dec"] == Type(
        kind=ValueKind.OBJECT, variants=(HOLD, SELL), nullable=True
    )
    assert _view_type(narrowing, "b").fields["dec"] == BUY


def test_control_edges_join_by_union_and_an_edge_without_facts_unconstrains():
    """Two branches into one node give both members; a fact-free route removes the fact."""
    nodes = [
        _node("d"),
        _node("z"),
        _case("c", "${d.output.action}", [("buy", "x"), ("sell", "x"), ("hold", "y")]),
        _case("g", None, [("${z.output} > 0", "y")]),
        _node("x"),
        _node("y"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "x") == BUY_OR_SELL
    assert narrowing.environment("y") == {}
    assert narrowing.for_node("y")[0]["d"].type == DECISION


def test_a_required_data_group_joins_its_producers_by_union():
    """A coalesce of the buy and sell branches sees both members."""
    edges = [
        _data("b", "m", "v"),
        _data("s", "m", "v"),
        _data("b", "o", "v", optional=True),
    ]
    nodes = _branches() + [_node("m"), _node("o")]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "m") == BUY_OR_SELL
    assert narrowing.environment("o") == {}


def test_depends_on_carries_the_source_environment_and_runs_after_does_not():
    """A co-skipping ordering edge narrows; a pure-order edge does not."""
    nodes = _branches() + [_node("n", depends_on=["b"]), _node("r", runs_after=["b"])]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "n") == BUY
    assert narrowing.environment("r") == {}


def test_contributions_intersect():
    """A data group from both branches and a dependency on one give that one."""
    edges = [_data("b", "m", "v"), _data("s", "m", "v")]
    nodes = _branches() + [_node("m", depends_on=["s"])]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "m") == SELL


def test_two_required_groups_from_disjoint_branches_meet_empty():
    """An empty member set on a non-nullable base leaves the node unnarrowed."""
    edges = [_data("b", "m", "u"), _data("s", "m", "v")]
    nodes = _branches() + [_node("m")]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert narrowing.environment("m") == {"d.output": Members(())}
    assert _view_type(narrowing, "m") == DECISION


def test_nested_case_composes_with_its_own_environment():
    """A CASE inside a branch narrows within that branch's members."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2"), ("sell", "c2")], "h"),
        _case("c2", "${d.output.action}", [("buy", "p")], "q"),
        _node("p"),
        _node("q"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "c2") == BUY_OR_SELL
    assert _view_type(narrowing, "p") == BUY
    assert _view_type(narrowing, "q") == SELL


def test_a_contradictory_nested_case_branch_is_empty():
    """A sell branch inside the buy branch never runs, so it stays unnarrowed."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.action}", [("sell", "z")]),
        _node("z"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("z") == {"d.output": Members(())}
    assert _view_type(narrowing, "z") == DECISION


def test_outer_bases_narrow_before_inner_ones():
    """Narrowing `d.output` to Buy lets a nested case narrow `d.output.leg`."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.leg.type}", [("limit", "l")]),
        _node("l"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    seen = _view_type(narrowing, "l")
    assert seen.fields["leg"] == LIMIT
    assert seen == replace(BUY, fields={**BUY.fields, "leg": LIMIT})
    assert narrowing.for_node("l")[0]["d"].origins == _decision_contract().origins


def test_flow_input_bases_narrow_the_input_view():
    """An `on:` over `input.<name>.<tag>` narrows the flow-input contract."""
    inputs = {"order": ValueContract(DECISION)}
    nodes = [_case("c", "${input.order.action}", [("hold", "w")]), _node("w")]
    narrowing = _compute(nodes, [], {}, inputs)

    producers, view = narrowing.for_node("w")
    assert producers == {}
    assert view["order"].type == HOLD


def test_partial_record_producer_narrows_a_proven_field():
    """A proven union field of a partial record narrows inside the record."""
    contract = PartialRecordContract(
        fields=(("dec", ValueContract(DECISION)), ("raw", None))
    )
    nodes = [
        _node("d"),
        _case("c", "${d.output.dec.action}", [("sell", "s")]),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": contract})

    view = narrowing.for_node("s")[0]["d"]
    assert view == PartialRecordContract(
        fields=(("dec", ValueContract(SELL)), ("raw", None))
    )


def test_bases_that_are_not_tagged_unions_give_no_facts():
    """A non-union base, a non-tag field, or a computed `on:` narrows nothing."""
    producers = {
        "d": _decision_contract(),
        "t": ValueContract(TEXT),
    }
    nodes = [
        _node("d"),
        _node("t"),
        _case("c1", "${t.output.text}", [("hi", "a")]),
        _case("c2", "${d.output.qty}", [("1", "b")]),
        _case("c3", "${d.output.action:-hold}", [("buy", "e")]),
        _node("a"),
        _node("b"),
        _node("e"),
    ]
    narrowing = _compute(nodes, [], producers)

    for node_id in ("a", "b", "e"):
        assert narrowing.environment(node_id) == {}


# --------------------------------------------------------------------------- #
# Empty sets
# --------------------------------------------------------------------------- #


def test_defensive_else_after_every_tag_is_empty_and_unnarrowed():
    """`else:` after listing every tag never runs; a non-nullable base keeps its type."""
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("hold", "b"), ("sell", "b")],
            "e",
        ),
        _node("b"),
        _node("e"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("e") == {"d.output": Members(())}
    assert _view_type(narrowing, "e") == DECISION
    assert _view_type(narrowing, "b") == DECISION


def test_defensive_else_over_a_nullable_base_sees_none():
    """`else:` after every tag on a nullable base can only see `None`."""
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("hold", "b"), ("sell", "b")],
            "e",
        ),
        _node("b"),
        _node("e"),
    ]
    producers = {"d": _decision_contract(replace(DECISION, nullable=True))}
    narrowing = _compute(nodes, [], producers)

    assert narrowing.environment("e") == {"d.output": Members((), nullable=True)}
    assert _view_type(narrowing, "e").kind is ValueKind.NONE


def test_a_when_value_that_is_not_a_tag_is_a_load_error():
    """A typo'd tag on a union base is rejected with the union's tags."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buyy", "b")], "h"),
        _node("b"),
        _node("h"),
    ]
    with pytest.raises(LoadError) as caught:
        _compute(nodes, [], {"d": _decision_contract()})
    assert str(caught.value).startswith(
        "when: buyy is not a tag of the union at ${d.output.action} "
        "(tags: buy, hold, sell)"
    )


def test_the_tag_error_points_at_the_when_value():
    """The error is located at the offending `when:` value's range."""
    ranges = [SourceRange(5, 15, 5, 18), SourceRange(7, 15, 7, 19)]
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("sel", "s")],
            when_ranges=ranges,
        ),
        _node("b"),
        _node("s"),
    ]
    diagnostics = LoadDiagnostics(label="flow.yaml", text="line\n" * 10)
    with pytest.raises(LoadError) as caught:
        _compute(nodes, [], {"d": _decision_contract()}, diagnostics=diagnostics)
    assert caught.value.diagnostic.frames[-1].range == ranges[1]


def test_a_tag_error_on_a_base_only_the_branch_view_has():
    """A member-only union field is checked against the union the branch sees."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.leg.type}", [("limt", "l")]),
        _node("l"),
        _node("h"),
    ]
    with pytest.raises(LoadError) as caught:
        _compute(nodes, [], {"d": _decision_contract()})
    assert str(caught.value).startswith(
        "when: limt is not a tag of the union at ${d.output.leg.type} "
        "(tags: limit, market)"
    )


# --------------------------------------------------------------------------- #
# Cycles and unknown ids
# --------------------------------------------------------------------------- #


def test_nodes_kahn_cannot_order_stay_unnarrowed():
    """A branch target on a cycle, and what follows it, are reported unordered."""
    edges = [_data("a", "k", "v"), _data("k", "a", "w"), _data("k", "after", "v")]
    nodes = _branches() + [
        _node("a", depends_on=["b"]),
        _node("k"),
        _node("after"),
    ]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert narrowing.unordered == frozenset({"a", "k", "after"})
    assert _view_type(narrowing, "a") == DECISION
    assert _view_type(narrowing, "b") == BUY


def test_unknown_ids_are_ignored():
    """Edges and dependencies naming unknown nodes neither fail nor contribute."""
    edges = [_data("ghost", "b", "v"), _data("b", "nowhere", "v")]
    nodes = _branches() + [_node("n", depends_on=["b", "ghost"])]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "n") == BUY
    assert narrowing.unordered == frozenset()


# --------------------------------------------------------------------------- #
# narrow_contract_at
# --------------------------------------------------------------------------- #


def test_narrow_contract_at_the_root_and_a_nested_record_field():
    """The type at the path is replaced and every origin is kept."""
    contract = _decision_contract()
    assert narrow_contract_at(contract, (), BUY) == ValueContract(
        BUY, contract.origins
    )

    buy = ValueContract(BUY, contract.origins)
    narrowed = narrow_contract_at(buy, ("leg",), LIMIT)
    assert narrowed.type == replace(BUY, fields={**BUY.fields, "leg": LIMIT})
    assert narrowed.origins == contract.origins


def test_narrow_contract_at_does_not_cross_unions_or_missing_fields():
    """A path through a union, an absent field, or a scalar cannot be narrowed."""
    contract = _decision_contract()
    assert narrow_contract_at(contract, ("leg",), LIMIT) is None
    assert narrow_contract_at(ValueContract(BUY), ("nope",), LIMIT) is None
    assert narrow_contract_at(ValueContract(BUY), ("qty", "x"), LIMIT) is None


def test_narrow_contract_at_inside_a_partial_record():
    """A proven field narrows in place; an opaque one cannot be narrowed by a path."""
    inner = ValueContract(DECISION, ContractOrigins(default=_ROOT))
    contract = PartialRecordContract(fields=(("dec", inner), ("raw", None)))

    narrowed = narrow_contract_at(contract, ("dec",), BUY)
    assert narrowed == PartialRecordContract(
        fields=(("dec", ValueContract(BUY, inner.origins)), ("raw", None))
    )
    assert narrow_contract_at(contract, ("raw", "x"), BUY) is None
    assert narrow_contract_at(contract, ("missing",), BUY) is None


def test_narrow_contract_at_the_root_of_a_partial_record_rebases_field_origins():
    """Replacing a whole partial record keeps each field's origins under its name."""
    inner = ValueContract(
        DECISION,
        ContractOrigins(default=_ROOT, entries=((("qty",), _QTY),)),
    )
    contract = PartialRecordContract(fields=(("dec", inner), ("raw", None)))
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"dec": BUY, "raw": Type.scalar(ValueKind.STRING)},
        required=frozenset({"dec", "raw"}),
    )

    narrowed = narrow_contract_at(contract, (), record)
    assert narrowed == ValueContract(
        record,
        ContractOrigins(entries=((("dec",), _ROOT), (("dec", "qty"), _QTY))),
    )
