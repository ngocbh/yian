"""The `Route` outcome + the `NodeRouted` event — routing as a first-class outcome arm.

A router (CASE) returns `Route(handle)` (no value); the engine's node seam turns it into a
`NodeRouted` terminal, and the engine dispatches routing on that event (not on node.kind).
"""

from yian.events import NodeRouted
from yian.nodes.base import NodeResult, Route


def test_route_carries_handle():
    assert Route("x").handle == "x"
    assert Route(handle="case_a").handle == "case_a"


def test_route_is_in_node_result_union():
    assert Route in NodeResult.__args__


def test_node_routed_event_fields():
    ev = NodeRouted("n", "x")
    assert ev.node_id == "n"
    assert ev.handle == "x"
