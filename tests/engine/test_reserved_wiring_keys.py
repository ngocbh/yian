"""`NodeBase.reserved_wiring_keys()` — the static author-wiring keys a node owns
beyond its declared `params` (MAP `over`, WAIT `until`). Load-time only (no pool);
`check_wiring_parity` + MAP validation read it instead of dispatching on `node.kind`.
"""

from yian.nodes.agent import AgentNode
from yian.nodes.code.node import CodeNode
from yian.nodes.map.node import MapNode
from yian.nodes.wait.node import WaitNode


def test_reserved_wiring_keys_by_kind():
    assert MapNode("m", flow_id="c", child=None, child_inputs=[]).reserved_wiring_keys() == {"over"}
    assert WaitNode("w").reserved_wiring_keys() == {"until"}
    # a plain leaf reserves nothing (CodeNode.__init__ is (node_id, *, code: str) — a ref needs a ':')
    assert CodeNode("c", code="mod:fn").reserved_wiring_keys() == set()
    # an AGENT reserves `__llm_config` only once Compose wires values in its layers
    agent = AgentNode("a", prompt="p")
    assert agent.reserved_wiring_keys() == set()
    agent.wires_llm_config = True
    assert agent.reserved_wiring_keys() == {"__llm_config"}
