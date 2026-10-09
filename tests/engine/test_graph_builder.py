from yian.compile.model import END_ID, START_ID
from yian.nodes.base import NodeKind

from tests.engine._graph_builder import _graph


def test_graph_injects_real_start_and_end_nodes():
    from yian.nodes.code.node import CodeNode

    a = CodeNode("a", code="tests.engine._compose_codefns:echo")
    flow = _graph([a], [(START_ID, "a"), ("a", END_ID)])  # nodes as a LIST (test_engine convention)
    # The graph helper INJECTS the real boundary NODES (not bare edge sentinels). Roots remain the
    # TARGETS of START_ID edges, so do NOT assert roots == [START_ID]; assert the boundary NODES
    # exist + their kinds.
    assert START_ID in flow.nodes and flow.nodes[START_ID].kind is NodeKind.START
    assert END_ID in flow.nodes and flow.nodes[END_ID].kind is NodeKind.END
    # ordinary edges START_ID -> a -> END_ID exist as real adjacencies.
    pairs = {(e.from_, e.to) for e in flow.edges}
    assert (START_ID, "a") in pairs and ("a", END_ID) in pairs


def test_graph_passes_through_extra_nodes_and_handles():
    from yian.nodes.code.node import CodeNode

    nodes = [CodeNode(n, code="tests.engine._compose_codefns:echo")
             for n in ("a", "b", "c")]
    flow = _graph(nodes, [(START_ID, "a"), ("a", "b"), ("a", "c"),
                          ("b", END_ID), ("c", END_ID)])
    # boundary nodes injected (the helper injects the START_ID/END_ID nodes into the graph).
    assert START_ID in flow.nodes and flow.nodes[START_ID].kind is NodeKind.START
    assert END_ID in flow.nodes and flow.nodes[END_ID].kind is NodeKind.END
    assert set(flow.nodes) >= {"a", "b", "c", START_ID, END_ID}
