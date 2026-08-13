"""`NodeKind.MAP` + `MapNode`: the re-split MAP driver (no caller wired yet).

`MapNode` discriminates by KIND (`NodeKind.MAP`), not an `over` flag — it carries NO `over`
attribute and no `${...}` source on the node; the `over` SOURCE rides `flow.wiring[id]["over"]`
(the engine pre-resolves it into `inputs["over"]` before `run`). `run` returns one seed-first
`Grow`; the compile-owned builder constructs the MAP fan-in.
"""

import importlib

from agent_composer.nodes.base import Grow, NodeKind
from agent_composer.nodes.map import MapNode


def test_map_node_kind_value():
    assert NodeKind.MAP == "map"
    assert NodeKind.MAP.value == "map"


def test_map_node_carries_no_over_attr():
    n = MapNode("m", flow_id="child", child=object(), parallel=True)
    assert n.parallel is True
    assert n.child is not None
    assert not hasattr(n, "over")          # discriminator is the KIND, not an `over` flag
    assert MapNode("d", flow_id="child", child=object()).parallel is False


def test_map_node_run_builds_grow_fan_in_per_element():
    from tests.engine.test_expand import _child_flow

    child = _child_flow()
    n = MapNode("m", flow_id="child", child=child,
                child_inputs=[])
    grow = n.run({"over": [1, 2, 3]}, bind_item=lambda el: {"x": el})
    assert isinstance(grow, Grow)
    assert grow.seed == [{"x": 1}, {"x": 2}, {"x": 3}]
    assert grow.prune == frozenset()
    assert not hasattr(grow, "subgraph")


def test_map_node_run_empty_over_builds_lone_list_end():
    from tests.engine.test_expand import _child_flow

    n = MapNode("m", flow_id="child", child=_child_flow(), child_inputs=[])
    grow = n.run({"over": []}, bind_item=lambda el: {"x": el})
    assert isinstance(grow, Grow) and grow.seed == []
    assert grow.prune == frozenset()
    assert not hasattr(grow, "subgraph")


def test_map_node_kind_exists_and_package_imports():
    assert hasattr(NodeKind, "MAP")
    importlib.import_module("agent_composer.nodes.map")
