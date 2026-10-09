from yian.compile.model import CompiledFlow, Edge, START_ID, END_ID
from yian.nodes.contracts import Binding
from tests.engine._fakes import FuncNode


def _base() -> CompiledFlow:
    a = FuncNode("a", lambda i: 1)
    nodes = {"a": a}
    edges = [Edge(id="__start__->a", from_=START_ID, to="a"), Edge(id="a->__end__#0", from_="a", to=END_ID)]
    return CompiledFlow.from_parts(nodes=nodes, edges=edges)


def test_add_subgraph_extends_nodes_edges_wiring():
    flow = _base()
    b = FuncNode("ns/b", lambda i: 2)
    flow.add_subgraph(
        nodes={"ns/b": b},
        edges=[Edge(id="a->ns/b#0", from_="a", to="ns/b", input_group="x")],
        wiring={"ns/b": {"x": "${a.output}"}},
    )
    assert "ns/b" in flow.nodes
    assert flow.wiring["ns/b"] == {"x": Binding("${a.output}")}
    # adjacency updated both directions
    assert any(e.to == "ns/b" for e in flow.incoming("ns/b"))
    assert any(e.to == "ns/b" for e in flow.outgoing("a"))


def test_remove_subgraph_drops_nodes_edges_wiring_and_adjacency():
    # Two sibling namespaces spawned onto the base graph; remove ONE and assert the other survives.
    flow = _base()
    for ns in ("iter0", "iter1"):
        node = FuncNode(f"{ns}/b", lambda i: 2)
        flow.add_subgraph(
            nodes={f"{ns}/b": node},
            edges=[
                Edge(id=f"a->{ns}/b", from_="a", to=f"{ns}/b", input_group="x"),
                Edge(id=f"{ns}/b->__end__", from_=f"{ns}/b", to=END_ID),
            ],
            wiring={f"{ns}/b": {"x": "${a.output}"}},
        )

    flow.remove_subgraph({"iter0/b"})

    # removed node is gone from nodes + wiring
    assert "iter0/b" not in flow.nodes
    assert "iter0/b" not in flow.wiring
    # no edge references the removed id (as producer or consumer)
    assert all(e.from_ != "iter0/b" and e.to != "iter0/b" for e in flow.edges)
    # adjacency: no key for the removed id, and no surviving value edge references it
    assert "iter0/b" not in flow._outgoing
    assert "iter0/b" not in flow._incoming
    for edges in flow._outgoing.values():
        assert all(e.to != "iter0/b" for e in edges)
    for edges in flow._incoming.values():
        assert all(e.from_ != "iter0/b" for e in edges)

    # SURVIVING sibling namespace + base node untouched
    assert "iter1/b" in flow.nodes
    assert flow.wiring["iter1/b"] == {"x": Binding("${a.output}")}
    assert any(e.to == "iter1/b" for e in flow.incoming("iter1/b"))
    assert any(e.to == "iter1/b" for e in flow.outgoing("a"))
    assert "a" in flow.nodes
    # 'a' no longer routes to the removed node, still routes to the survivor
    assert all(e.to != "iter0/b" for e in flow.outgoing("a"))
