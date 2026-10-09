"""The seed-first Grow outcome and the compile-owned Flow value type."""
from yian.compile.model import Flow
from yian.nodes.base import Grow


def test_flow_holds_fragment_and_start_id():
    sg = Flow(nodes={}, edges=[], wiring={}, start_id="r", end_id="r")
    assert sg.start_id == "r" and sg.nodes == {} and sg.edges == [] and sg.wiring == {}


def test_grow_defaults_to_empty_prune():
    g = Grow(seed=None)
    assert g.prune == frozenset()
    assert g.seed is None
    assert not hasattr(g, "subgraph")


def test_grow_carries_seed():
    g = Grow(seed={"x": 1})
    assert g.seed == {"x": 1}


def test_grow_is_a_noderesult_member():
    from yian.nodes.base import NodeResult  # Union includes Grow
    import typing
    assert Grow in typing.get_args(NodeResult)
