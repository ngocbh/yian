from agent_composer.nodes.start.node import StartNode


def test_node_range_defaults_none():
    n = StartNode("s")
    assert n.range is None and n.field_ranges == {}
