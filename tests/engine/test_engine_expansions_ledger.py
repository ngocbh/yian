"""Tests for the dispatcher-side expansion ledger.

The dispatcher appends one descriptor per `_apply_grow` call; nested expansions ride
under their parent. The ledger is the source of truth for snapshot/restore; these tests
verify the ledger shape itself.

Fixtures here are reused by the snapshot/restore and durable-resume tests — keep them stable.
"""

from typing import Any

from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, START_ID, FlowOutput
from agent_composer.nodes.call.node import CallNode
from agent_composer.nodes.map.node import MapNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.suspension.pause import HumanInputRequired
from tests.engine._fakes import FuncNode, PauseNode, stamp_reads
from tests.engine._graph_builder import _graph


def _inner_pause_child():
    """A child flow whose body contains a test-only generic Pause leaf
    and an `unwrap` step that lifts the answer scalar out of the record. The
    unwrap keeps the flow output a bare string so downstream e2e
    tests can assert `evs[-1].output == "approve"` cleanly."""
    unwrap = FuncNode("unwrap", lambda p: {"output": p["v"]})
    stamp_reads(unwrap, {"v": "${ask.output}"})
    return _graph(
        [
            FuncNode("seed", lambda p: {"output": p.get("payload", "x")}),
            PauseNode(
                "ask", reason=HumanInputRequired(prompt="Approve?", node_id="ask")
            ),
            unwrap,
        ],
        [(START_ID, "seed"), ("seed", "ask"), ("ask", "unwrap"), ("unwrap", END_ID)],
        outputs=[FlowOutput(name="answer", from_="${unwrap.output.output}")],
    )


def call_with_inner_pause():
    """Parent: ${input.payload} -> bridge(CALL inner_pause_child) -> after(echo) -> END_ID.
    pass `flow_id="inner_pause_child"` to CallNode (required keyword-only)."""
    child = _inner_pause_child()
    bridge = CallNode(
        "bridge",
        flow_id="inner_pause_child",
        child=child,
        child_inputs=child.nodes[child.start_id].params,
    )
    stamp_reads(bridge, {"payload": "${input.payload}"})
    after = FuncNode("after", lambda p: {"output": p["v"]})
    stamp_reads(after, {"v": "${bridge.output.answer}"})  # read .answer field
    return _graph(
        [bridge, after],
        [(START_ID, "bridge"), ("bridge", "after"), ("after", END_ID)],
        outputs=[FlowOutput(name="r", from_="${after.output}")],
    )


def map_with_inner_pause():
    """Parent: MAP over ${input.items} -> each clone of inner_pause_child -> END_ID-list."""
    child = _inner_pause_child()
    each = MapNode(
        "each",
        flow_id="inner_pause_child",
        child=child,
        child_inputs=child.nodes[child.start_id].params,
    )
    stamp_reads(each, {"over": "${input.items}", "payload": "${item}"})
    return _graph(
        [each],
        [(START_ID, "each"), ("each", END_ID)],
        outputs=[FlowOutput(name="results", from_="${each.output}")],
    )


def test_ledger_attributes_exist():
    """FlowEngine declares the empty ledger + parent-pointer fields."""
    g = _graph(
        [FuncNode("a", lambda p: {"output": 1})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="r", from_="${a.output}")],
    )
    engine = FlowEngine(g)
    assert engine.expansions == []
    assert engine._spawner_expansion == {}


def test_ledger_empty_for_static_flow():
    """A flow with no CALL/MAP/LOOP leaves `expansions` empty after a complete run.
    A static flow has no spawners to fire."""
    g = _graph(
        [FuncNode("a", lambda p: {"output": 1})],
        [(START_ID, "a"), ("a", END_ID)],
        outputs=[FlowOutput(name="r", from_="${a.output}")],
    )
    engine = FlowEngine(g)
    list(engine.run())
    assert engine.expansions == []


def test_ledger_records_call_expansion():
    """A CALL spawner adds ONE GrowRecord to the top-level ledger keyed by the spawner id;
    its `seed` is the call-arg record. The cloned child's pause leaves `bridge` EXPANDED but
    the ledger entry is independent."""
    g = call_with_inner_pause()
    engine = FlowEngine(g, run_inputs={"payload": "go"}, grow_builder=build_grow)
    list(engine.run())  # parks at the inner ask
    calls = [e for e in engine.expansions if e.spawner_id == "bridge"]
    assert len(calls) == 1
    assert calls[0].seed == {"payload": "go"}


def test_ledger_records_map_expansion_with_records():
    """A MAP spawner adds ONE GrowRecord whose `seed` is the per-element records list."""
    g = map_with_inner_pause()
    engine = FlowEngine(g, run_inputs={"items": ["a", "b"]}, grow_builder=build_grow)
    list(engine.run())  # parks at both element leaves
    maps = [e for e in engine.expansions if e.spawner_id == "each"]
    assert len(maps) == 1
    assert len(maps[0].seed) == 2
    # seed carries the per-element binding (payload from ${item}):
    payloads = sorted(r.get("payload") for r in maps[0].seed)
    assert payloads == ["a", "b"]


def test_nested_call_inside_map_is_descriptor_child():
    """A MAP whose child contains a CALL: the inner CALL GrowRecord rides under the map
    record's flat `children`, NOT in engine.expansions top-level."""
    # Build a MAP whose child has its own CALL inside:
    inner_child = _inner_pause_child()
    nested_call_node = CallNode("inner_bridge", flow_id="inner_pause_child",
                                child=inner_child,
                                child_inputs=inner_child.nodes[inner_child.start_id].params)
    stamp_reads(nested_call_node, {"payload": "${input.payload}"})  # parent body input
    parent_child = _graph(
        [nested_call_node],
        [(START_ID, "inner_bridge"), ("inner_bridge", END_ID)],
        outputs=[FlowOutput(name="result", from_="${inner_bridge.output.answer}")],
    )
    each = MapNode("each", flow_id="parent_child", child=parent_child,
                   child_inputs=parent_child.nodes[parent_child.start_id].params)
    stamp_reads(each, {"over": "${input.items}", "payload": "${item}"})
    parent = _graph(
        [each],
        [(START_ID, "each"), ("each", END_ID)],
        outputs=[FlowOutput(name="r", from_="${each.output}")],
    )
    engine = FlowEngine(parent, run_inputs={"items": ["a"]}, grow_builder=build_grow)
    list(engine.run())  # parks deep inside element 0
    # Top-level: exactly ONE record, the MAP. No top-level inner CALL record.
    assert len(engine.expansions) == 1
    map_rec = engine.expansions[0]
    assert map_rec.spawner_id == "each"
    # The map record's children carry the inner CALL GrowRecord, namespaced under each#0:
    kids = map_rec.children
    assert len(kids) == 1
    assert kids[0].spawner_id.endswith("/inner_bridge")  # namespaced under each#0
