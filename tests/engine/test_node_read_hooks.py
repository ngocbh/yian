"""Node-owned read hooks — `bind_reserved` / `binds_per_item`.

The read boundary's reserved-input resolution (timed WAIT `until`, MAP `over`) lives on the
node, not in a `node.kind == ...` branch in `eval_node`. These tests pin the hook contract per
kind: a timed WAIT resolves `until`; MAP declares per-item binding and resolves `over` to a list;
an ordinary node reserves nothing and binds up front.
"""

from agent_composer.nodes.base import Node, NodeKind
from agent_composer.nodes.map import MapNode
from agent_composer.nodes.wait.node import WaitNode
from agent_composer.typesys.pool import VariablePool


class _PlainNode(Node):
    """A minimal ordinary node (no reserved keys, binds up front) for the default-hook check."""

    kind = NodeKind.CODE

    def _run_impl(self, inputs, **caps):  # pragma: no cover — never run here
        return None


def test_timed_wait_bind_reserved_resolves_until():
    # A timed WAIT pre-resolves its `until` source (from node_wiring) into a concrete ISO ts.
    pool = VariablePool()
    pool.set("deadline", "2026-01-01T00:00:00")
    node = WaitNode("w", is_timed=True)
    got = node.bind_reserved({"until": "${deadline.output}"}, pool)
    assert got == {"until": "2026-01-01T00:00:00"}


def test_event_wait_bind_reserved_is_empty():
    # An event-mode WAIT reserves nothing.
    assert WaitNode("w", is_timed=False).bind_reserved({}, VariablePool()) == {}


def test_map_binds_per_item_and_resolves_over():
    # MAP declares per-element binding and resolves `over` to the list run() maps over.
    assert MapNode.binds_per_item is True
    pool = VariablePool()
    pool.set("items", [1, 2, 3])
    node = MapNode("m", flow_id="child")
    assert node.bind_reserved({"over": "${items.output}"}, pool) == {"over": [1, 2, 3]}


def test_map_over_not_a_list_raises():
    # A non-list `over` source is a loud RuntimeError (surfaced as NodeFailed by the read seam).
    pool = VariablePool()
    pool.set("scalar", 7)
    node = MapNode("m", flow_id="child")
    try:
        node.bind_reserved({"over": "${scalar.output}"}, pool)
    except RuntimeError as exc:
        assert "did not resolve to a list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for a non-list `over`")


def test_plain_node_defaults():
    # An ordinary node reserves no keys and binds up front (not per item).
    node = _PlainNode("n")
    assert node.binds_per_item is False
    assert node.bind_reserved({}, VariablePool()) == {}
