from yian.nodes.base import NodeKind


def test_start_end_call_member_values():
    assert NodeKind.START == "start"
    assert NodeKind.END == "end"
    assert NodeKind.CALL == "call"
    assert NodeKind.START.value == "start"


def test_new_members_are_distinct():
    assert len({NodeKind.START, NodeKind.END, NodeKind.CALL, NodeKind.MAP}) == 4
    # The retired REF member stays absent; CALL and MAP are distinct current drivers.
    assert not hasattr(NodeKind, "REF") and hasattr(NodeKind, "MAP")
