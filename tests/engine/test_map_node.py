"""`NodeKind.MAP` + `MapNode`: the re-split MAP driver (no caller wired yet).

`MapNode` discriminates by KIND (`NodeKind.MAP`), not an `over` flag — it carries NO `over`
attribute and no `${...}` source on the node; the `over` SOURCE rides `flow.wiring[id]["over"]`
(the engine pre-resolves it into `inputs["over"]` before `run`). `run` returns one `Grow(Flow)`
— it builds the whole MAP fan-in (N child clones + a synthesized list END); an empty `over` -> a
subgraph whose sole node is the list END.
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
    from agent_composer.compile.expand import map_callsite, ns

    from tests.engine.test_expand import _child_flow

    child = _child_flow()
    n = MapNode("m", flow_id="child", child=child,
                child_inputs=[])
    grow = n.run({"over": [1, 2, 3]}, bind_item=lambda el: {"x": el})
    assert isinstance(grow, Grow)
    assert grow.seed == [{"x": 1}, {"x": 2}, {"x": 3}]
    # The single entry is the synthetic map#/__start__; each element START is present in nodes and
    # reached from the synthetic start by an ordering edge.
    from agent_composer.compile.model import START_ID
    assert grow.subgraph.start_id == ns("m", START_ID)
    for i in range(3):
        assert ns(map_callsite("m", i), child.start_id) in grow.subgraph.nodes


def test_map_node_run_empty_over_builds_lone_list_end():
    from agent_composer.compile.expand import ns
    from agent_composer.compile.model import END_ID, START_ID

    from tests.engine.test_expand import _child_flow

    n = MapNode("m", flow_id="child", child=_child_flow(), child_inputs=[])
    grow = n.run({"over": []}, bind_item=lambda el: {"x": el})
    assert isinstance(grow, Grow) and grow.seed == []
    # N=0: the body is the synthetic start + the list END.
    assert set(grow.subgraph.nodes) == {ns("m", START_ID), ns("m", END_ID)}


def test_map_node_kind_exists_and_package_imports():
    assert hasattr(NodeKind, "MAP")
    importlib.import_module("agent_composer.nodes.map")
