"""`NodeBase.reserved_wiring_keys()` — the static author-wiring keys a node owns
beyond its declared `params` (MAP `over`, WAIT `until`). Load-time only (no pool);
`check_wiring_parity` + MAP validation read it instead of dispatching on `node.kind`.
"""

from agent_composer.nodes.code.node import CodeNode
from agent_composer.nodes.map.node import MapNode
from agent_composer.nodes.wait.node import WaitNode


def test_reserved_wiring_keys_by_kind():
    assert MapNode("m", flow_id="c", child=None, child_inputs=[]).reserved_wiring_keys() == {"over"}
    assert WaitNode("w").reserved_wiring_keys() == {"until"}
    # a plain leaf reserves nothing (CodeNode.__init__ is (node_id, *, code: str) — a ref needs a ':')
    assert CodeNode("c", code="mod:fn").reserved_wiring_keys() == set()
