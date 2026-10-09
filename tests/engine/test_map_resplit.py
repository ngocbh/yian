"""CALL and MAP are distinct typed drivers.

`kind: call` builds `CallNode` (call once); `kind: map` builds `MapNode` (map over a list).
The retired ref kind/package stays absent; `CallNode` carries no `over`/`parallel`, while
`MapNode` has its own kind and keeps the `over` source in flow wiring.
"""

import importlib

import pytest

from yian.nodes.base import NodeKind


def test_retired_ref_kind_is_absent_and_call_map_are_distinct():
    assert not hasattr(NodeKind, "REF")
    assert NodeKind.CALL.value == "call"
    assert NodeKind.MAP.value == "map"               # MAP re-split into its own kind
    assert NodeKind.CALL is not NodeKind.MAP


def test_retired_ref_package_is_absent():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yian.nodes.ref")
    importlib.import_module("yian.nodes.call")
    importlib.import_module("yian.nodes.map")


def test_call_node_excludes_map_fields():
    from yian.nodes.call import CallNode
    from yian.nodes.map import MapNode
    call = CallNode("c", flow_id="child", child=object())
    assert call.kind is NodeKind.CALL
    assert not hasattr(call, "over") and not hasattr(call, "parallel")
    mp = MapNode("m", flow_id="child", child=object(), parallel=True)
    assert mp.kind is NodeKind.MAP and mp.parallel is True
    assert not hasattr(mp, "over")                    # the SOURCE rides flow.wiring, not the node
