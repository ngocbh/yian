from types import SimpleNamespace

import pytest

from agent_composer.nodes.base import Grow, NodeKind
from agent_composer.nodes.call import CallNode
from agent_composer.nodes.map import MapNode


def _child():
    # A stub baked child: run only reads self.child for the not-baked guard; compile owns
    # construction from the returned seed.
    return SimpleNamespace(nodes={}, edges=[], wiring={}, outputs=[])


def test_call_node_has_no_map_fields():
    call = CallNode("c", flow_id="child", child=_child())
    assert call.kind == NodeKind.CALL
    assert not hasattr(call, "over")
    assert not hasattr(call, "parallel")


def test_map_node_kind_and_parallel_fields():
    mp = MapNode("m", flow_id="child", child=_child(), parallel=True)
    assert mp.kind == NodeKind.MAP
    assert mp.parallel is True
    assert not hasattr(mp, "over")                    # the SOURCE rides flow.wiring, not the node
    assert MapNode("d", flow_id="child", child=_child()).parallel is False


def test_call_returns_grow():
    # CALL returns only the raw call-arg record; compile owns fragment construction.
    from tests.engine.test_expand import _child_flow

    node = CallNode("c", flow_id="child", child=_child_flow(), child_inputs=[])
    out = node.run({"topic": "ACME"})
    assert isinstance(out, Grow)
    assert out.seed == {"topic": "ACME"}
    assert out.prune == frozenset()
    assert not hasattr(out, "subgraph")


def test_map_returns_grow():
    # MAP returns only the raw per-element records; compile owns fan-in construction.
    from tests.engine.test_expand import _child_flow

    node = MapNode("m", flow_id="child", child=_child_flow(), child_inputs=[])
    out = node.run({"over": ["ACME", "BETA"]}, bind_item=lambda el, index: {"x": el})
    assert isinstance(out, Grow)
    assert out.seed == [{"x": "ACME"}, {"x": "BETA"}]
    assert out.prune == frozenset()
    assert not hasattr(out, "subgraph")


def test_map_empty_returns_grow_with_lone_list_end():
    from tests.engine.test_expand import _child_flow

    node = MapNode("m", flow_id="child", child=_child_flow(), child_inputs=[])
    out = node.run({"over": []}, bind_item=lambda el, index: {"x": el})
    assert isinstance(out, Grow)
    assert out.seed == []
    assert out.prune == frozenset()
    assert not hasattr(out, "subgraph")


def test_call_unbaked_child_raises():
    node = CallNode("c", flow_id="child", child=None)
    with pytest.raises(RuntimeError, match="not baked"):
        node.run({"topic": "ACME"})


def test_map_unbaked_child_raises():
    node = MapNode("m", flow_id="child", child=None)
    with pytest.raises(RuntimeError, match="not baked"):
        node.run({"over": ["ACME"]}, bind_item=lambda el, index: {"topic": el})
