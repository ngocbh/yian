"""`host_control_edges`: a lifted node takes its host's CASE routes, ordering, and data gates."""

import pytest

from yian.compile.model import Edge
from yian.compose.edges.hosts import host_control_edges
from yian.compose.parse.file import CodeDescriptor
from yian.nodes.case import ELSE_HANDLE


def _code(node_id, **kwargs):
    """A CODE descriptor with the given common fields."""
    return CodeDescriptor(id=node_id, code="return 1", **kwargs)


def _nodes(*descriptors):
    """Descriptors keyed by id, in the given order."""
    return {d.id: d for d in descriptors}


def _data(src, dst, group, optional=False, i=0):
    """A data edge from ``src`` into ``dst``'s input ``group``."""
    return Edge(id=f"{src}->{dst}#{i}", from_=src, to=dst, input_group=group, optional=optional)


def _route(source, target, handle, index=0):
    """A CASE route edge from ``source`` to ``target``."""
    return Edge(id=f"{source}->{target}#{index}", from_=source, to=target, source_handle=handle)


def test_route_into_the_host_is_copied():
    nodes = _nodes(_code("c"), _code("h"), _code("L", host="h"))
    assert host_control_edges(nodes, [_route("c", "h", "h")]) == [
        Edge(id="c->L#0", from_="c", to="L", source_handle="h")
    ]


def test_else_route_keeps_its_handle():
    nodes = _nodes(_code("c"), _code("h"), _code("L", host="h"))
    [edge] = host_control_edges(nodes, [_route("c", "h", ELSE_HANDLE)])
    assert edge.source_handle == ELSE_HANDLE


def test_each_route_into_the_host_gets_a_copy():
    nodes = _nodes(_code("c"), _code("d"), _code("h"), _code("L", host="h"))
    edges = [
        _route("c", "h", "h", 0),
        _route("c", "h", ELSE_HANDLE, 1),
        _route("d", "h", "h", 0),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="c->L#0", from_="c", to="L", source_handle="h"),
        Edge(id="c->L#1", from_="c", to="L", source_handle=ELSE_HANDLE),
        Edge(id="d->L#0", from_="d", to="L", source_handle="h"),
    ]


def test_depends_on_and_runs_after_become_ordering_edges():
    nodes = _nodes(
        _code("p"),
        _code("q"),
        _code("h", depends_on=["p"], runs_after=["q"]),
        _code("L", host="h"),
    )
    assert host_control_edges(nodes, []) == [
        Edge(id="p~>L#0", from_="p", to="L", ordering=True, optional=False),
        Edge(id="q~>L#0", from_="q", to="L", ordering=True, optional=True),
    ]


def test_routes_come_before_depends_on_then_runs_after():
    nodes = _nodes(
        _code("c"),
        _code("p"),
        _code("q"),
        _code("h", depends_on=["p"], runs_after=["q"]),
        _code("L", host="h"),
    )
    edges = host_control_edges(nodes, [_route("c", "h", "h")])
    assert [e.id for e in edges] == ["c->L#0", "p~>L#0", "q~>L#0"]


def test_unknown_or_self_sources_are_skipped():
    # The host reports an unknown source itself; a host ordering its own lifted node would
    # make a self-loop.
    nodes = _nodes(
        _code("h", depends_on=["missing"], runs_after=["L"]),
        _code("L", host="h"),
    )
    assert host_control_edges(nodes, []) == []


def test_host_chains_are_followed():
    nodes = _nodes(
        _code("c"),
        _code("p"),
        _code("h", depends_on=["p"]),
        _code("L1", host="h"),
        _code("L2", host="L1"),
    )
    edges = host_control_edges(nodes, [_route("c", "h", "h")])
    assert [(e.from_, e.to, e.source_handle, e.ordering) for e in edges] == [
        ("c", "L1", "h", False),
        ("p", "L1", None, True),
        ("c", "L2", "h", False),
        ("p", "L2", None, True),
    ]


def test_ids_skip_ones_already_used():
    nodes = _nodes(_code("c"), _code("h"), _code("L", host="h"))
    data = Edge(id="c->L#0", from_="c", to="L", input_group="x")
    [edge] = host_control_edges(nodes, [data, _route("c", "h", "h")])
    assert edge.id == "c->L#1"


def test_ordering_edges_are_not_copied_as_routes_and_data_edges_become_gates():
    nodes = _nodes(_code("a"), _code("h"), _code("L", host="h"))
    edges = [
        Edge(id="a->h#0", from_="a", to="h", input_group="x"),
        Edge(id="a~>h#0", from_="a", to="h", ordering=True),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="a->L#0", from_="a", to="L", input_group="h.x")
    ]


def test_no_host_or_ungated_host_gives_no_edges():
    nodes = _nodes(_code("c"), _code("h"), _code("free"), _code("L", host="free"))
    assert host_control_edges(nodes, [_route("c", "h", "h")]) == []


@pytest.mark.parametrize(
    "lifted, edges",
    [
        (_code("L", host="h"), [_route("c", "L", "L")]),
        (_code("L", host="h", depends_on=["c"]), []),
        (_code("L", host="h", runs_after=["c"]), []),
    ],
)
def test_a_lifted_node_with_its_own_control_is_a_compiler_bug(lifted, edges):
    nodes = _nodes(_code("c"), _code("h"), lifted)
    with pytest.raises(ValueError, match="'L'"):
        host_control_edges(nodes, edges)


def test_missing_host_or_host_cycle_is_a_compiler_bug():
    with pytest.raises(ValueError, match="'nowhere'"):
        host_control_edges(_nodes(_code("L", host="nowhere")), [])
    with pytest.raises(ValueError, match="cycle"):
        host_control_edges(_nodes(_code("A", host="B"), _code("B", host="A")), [])
    # A host's data producer whose own hosts cycle is met before that producer's own turn.
    nodes = _nodes(_code("h"), _code("L", host="h"), _code("A", host="B"), _code("B", host="A"))
    with pytest.raises(ValueError, match="cycle"):
        host_control_edges(nodes, [_data("A", "h", "x")])


def test_host_data_input_becomes_a_gate():
    nodes = _nodes(_code("m"), _code("h"), _code("L", host="h"))
    edges = [_data("L", "h", "draft"), _data("m", "h", "notes")]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L#0", from_="m", to="L", input_group="h.notes")
    ]


def test_defaulted_host_input_gives_an_optional_gate():
    nodes = _nodes(_code("m"), _code("h"), _code("L", host="h"))
    edges = [_data("L", "h", "draft"), _data("m", "h", "notes", optional=True)]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L#0", from_="m", to="L", input_group="h.notes", optional=True)
    ]


def test_coalesce_producers_share_one_gate_group():
    nodes = _nodes(_code("a"), _code("b"), _code("h"), _code("L", host="h"))
    edges = [_data("L", "h", "draft"), _data("a", "h", "n"), _data("b", "h", "n")]
    assert host_control_edges(nodes, edges) == [
        Edge(id="a->L#0", from_="a", to="L", input_group="h.n"),
        Edge(id="b->L#0", from_="b", to="L", input_group="h.n"),
    ]


def test_siblings_do_not_wait_on_each_other():
    nodes = _nodes(_code("m"), _code("h"), _code("L0", host="h"), _code("L1", host="h"))
    edges = [_data("L0", "h", "a"), _data("L1", "h", "b"), _data("m", "h", "notes")]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L0#0", from_="m", to="L0", input_group="h.notes"),
        Edge(id="m->L1#0", from_="m", to="L1", input_group="h.notes"),
    ]


def test_a_sibling_read_by_the_host_is_looked_through():
    nodes = _nodes(
        _code("m"),
        _code("m2"),
        _code("h"),
        _code("outer", host="h"),
        _code("inner", host="h"),
    )
    edges = [
        _data("outer", "h", "draft"),
        _data("m", "h", "notes"),
        _data("inner", "outer", "x"),
        _data("m2", "outer", "y"),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->outer#0", from_="m", to="outer", input_group="h.notes"),
        Edge(id="m->inner#0", from_="m", to="inner", input_group="h.notes"),
        Edge(id="m2->inner#0", from_="m2", to="inner", input_group="outer.y"),
    ]


def test_case_target_inner_call_waits_on_the_targets_other_inputs():
    nodes = _nodes(_code("d"), _code("outer"), _code("inner", host="outer"))
    edges = [_data("inner", "outer", "x"), _data("d", "outer", "y")]
    assert host_control_edges(nodes, edges) == [
        Edge(id="d->inner#0", from_="d", to="inner", input_group="outer.y")
    ]


def test_transitive_lifts_gate_on_every_ancestor():
    nodes = _nodes(
        _code("m"),
        _code("n"),
        _code("h"),
        _code("L1", host="h"),
        _code("L2", host="L1"),
    )
    edges = [
        _data("L1", "h", "a"),
        _data("m", "h", "b"),
        _data("L2", "L1", "x"),
        _data("n", "L1", "y"),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L1#0", from_="m", to="L1", input_group="h.b"),
        Edge(id="n->L2#0", from_="n", to="L2", input_group="L1.y"),
        Edge(id="m->L2#0", from_="m", to="L2", input_group="h.b"),
    ]


def test_adaptive_gate_shape():
    nodes = _nodes(
        _code("p"),
        _code("m"),
        _code("gate"),
        _code("call", host="gate"),
        _code("comp", host="gate"),
    )
    edges = [
        _data("call", "gate", "ctx"),
        _data("m", "gate", "notes"),
        _data("comp", "gate", "__questions"),
        _data("call", "comp", "ctx"),
        _data("m", "comp", "notes"),
        _data("p", "call", "tag"),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->call#0", from_="m", to="call", input_group="gate.notes")
    ]


def test_a_source_the_lifted_node_already_requires_gives_no_gate():
    nodes = _nodes(_code("m"), _code("h"), _code("L", host="h"))
    edges = [_data("m", "L", "v"), _data("L", "h", "draft"), _data("m", "h", "notes")]
    assert host_control_edges(nodes, edges) == []


def test_a_source_the_lifted_node_reads_optionally_still_gates():
    nodes = _nodes(_code("m"), _code("h"), _code("L", host="h"))
    edges = [
        _data("m", "L", "v", optional=True),
        _data("L", "h", "draft"),
        _data("m", "h", "notes"),
    ]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L#1", from_="m", to="L", input_group="h.notes")
    ]


def test_a_group_mixing_a_sibling_and_another_source_only_waits():
    nodes = _nodes(_code("t"), _code("h"), _code("L0", host="h"), _code("L1", host="h"))
    edges = [_data("L0", "h", "a"), _data("L1", "h", "n"), _data("t", "h", "n")]
    assert host_control_edges(nodes, edges) == [
        Edge(id="t->L0#0", from_="t", to="L0", input_group="h.n", optional=True),
        Edge(id="t->L1#0", from_="t", to="L1", input_group="h.n", optional=True),
    ]


def test_an_unknown_producer_gives_no_gate():
    nodes = _nodes(_code("h"), _code("L", host="h"))
    edges = [_data("L", "h", "draft"), _data("nope", "h", "x")]
    assert host_control_edges(nodes, edges) == []


def test_route_copies_and_data_gates_share_ids():
    nodes = _nodes(_code("m"), _code("h"), _code("L", host="h"))
    edges = [_route("m", "h", "h"), _data("m", "h", "notes", i=1)]
    assert host_control_edges(nodes, edges) == [
        Edge(id="m->L#0", from_="m", to="L", source_handle="h"),
        Edge(id="m->L#1", from_="m", to="L", input_group="h.notes"),
    ]


def test_routes_then_ordering_then_data_gates():
    nodes = _nodes(
        _code("c"),
        _code("q"),
        _code("m"),
        _code("h", runs_after=["q"]),
        _code("L", host="h"),
    )
    edges = host_control_edges(nodes, [_route("c", "h", "h"), _data("m", "h", "notes")])
    assert [e.id for e in edges] == ["c->L#0", "q~>L#0", "m->L#0"]
