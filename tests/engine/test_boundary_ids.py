from yian.compile.model import END_ID, START_ID
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.typesys.pool import VariablePool


def test_start_end_ids_are_owned_by_the_node_classes():
    # The boundary NODE ids are owned canonically by the node classes (StartNode.ID /
    # EndNode.ID); compile.model re-exports them as START_ID/END_ID.
    assert StartNode.ID == "__start__" == START_ID
    assert EndNode.ID == "__end__" == END_ID


def test_pool_start_id_default_mirrors_startnode_id():
    # state/pool.py keeps a literal default (it is a leaf below nodes and cannot import the
    # node class); this pins it to StartNode.ID so the duplicated literal can never diverge.
    assert VariablePool().start_id == StartNode.ID


def test_compiledflow_start_end_id_accessors():
    from yian.compile.model import CompiledFlow, Edge, FlowOutput
    from yian.nodes.code.node import CodeNode

    n = CodeNode("a", code="tests.engine._compose_codefns:echo")
    flow = CompiledFlow.from_parts({"a": n}, [Edge("s", START_ID, "a"), Edge("t", "a", END_ID)])
    assert flow.start_id == START_ID
    assert flow.end_id == END_ID
