from agent_composer.runtime.common import NodeState


def test_expanded_member_value():
    assert NodeState.EXPANDED == "expanded"
    assert NodeState.EXPANDED.value == "expanded"


def test_expanded_is_distinct_member():
    assert NodeState.EXPANDED not in (NodeState.UNKNOWN, NodeState.TAKEN, NodeState.SKIPPED)
