"""`host_control_edges`: a lifted node takes its host's CASE routes and ordering edges."""

import pytest

from yian.compile.model import Edge
from yian.compose.edges.hosts import host_control_edges
from yian.compose.parse.file import CodeDescriptor
from yian.nodes.case import ELSE_HANDLE


def _code(node_id, **kwargs):
    return CodeDescriptor(id=node_id, code="return 1", **kwargs)


def _nodes(*descriptors):
    return {d.id: d for d in descriptors}


def _route(source, target, handle, index=0):
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


def test_data_and_ordering_edges_into_the_host_are_not_routes():
    nodes = _nodes(_code("a"), _code("h"), _code("L", host="h"))
    edges = [
        Edge(id="a->h#0", from_="a", to="h", input_group="x"),
        Edge(id="a~>h#0", from_="a", to="h", ordering=True),
    ]
    assert host_control_edges(nodes, edges) == []


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
