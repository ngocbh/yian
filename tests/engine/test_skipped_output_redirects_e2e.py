"""Real growth-path coverage for skipped logical output redirects."""

from __future__ import annotations

from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, NodeState
from agent_composer.compose import load_flow
from agent_composer.events import RunFailed
from agent_composer.runtime.log import NodeExpandedRecord, NodeRoutedRecord
from tests.engine._graph_builder import _engine


_CALL = """
id: skipped-call
name: skipped_call
defs:
  broken:
    input:
      topic: str
    nodes:
      gate:
        kind: case
        cases:
          - when: "false"
            then: dead
        else: live
      dead:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
      live:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${dead.output}
nodes:
  invoke:
    kind: call
    call: broken
    input:
      topic: value
output: ${invoke.output}
"""


_MAP = """
id: skipped-map
name: skipped_map
defs:
  broken:
    input:
      topic: str
    nodes:
      gate:
        kind: case
        cases:
          - when: "false"
            then: dead
        else: live
      dead:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
      live:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${dead.output}
nodes:
  each:
    kind: map
    call: broken
    over: [value]
    input:
      topic: ${item}
output: ${each.output}
"""


_LOOP = """
id: skipped-loop
name: skipped_loop
defs:
  broken_turn:
    input:
      n: int
    nodes:
      gate:
        kind: case
        cases:
          - when: "false"
            then: dead
        else: live
      dead:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
      live:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
    output: ${dead.output}
nodes:
  repeat:
    kind: loop
    call: broken_turn
    input:
      n: 1
    times: 1
output: ${repeat.output}
"""


_NESTED_LOOP = """
id: skipped-nested-loop
name: skipped_nested_loop
defs:
  broken_turn:
    input:
      n: int
    nodes:
      gate:
        kind: case
        cases:
          - when: "false"
            then: dead
        else: live
      dead:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
      live:
        kind: code
        code: tests.engine._compose_codefns:loop_countdown
        input:
          n: ${input.n}
        output:
          n: int
    output: ${dead.output}
  wrapped:
    input:
      n: int
    nodes:
      repeat:
        kind: loop
        call: broken_turn
        input:
          n: ${input.n}
        times: 1
    output: ${repeat.output}
nodes:
  invoke:
    kind: call
    call: wrapped
    input:
      n: 1
output: ${invoke.output}
"""


def _run_failed_growth(text: str):
    """Drive one provider-free flow through the real injected growth builder."""
    engine = _engine(load_flow(text).compiled, grow_builder=build_grow)
    events = list(engine.run())
    assert isinstance(events[-1], RunFailed)
    records = tuple(envelope.record for envelope in engine.log)
    assert any(isinstance(record, NodeExpandedRecord) for record in records)
    assert any(isinstance(record, NodeRoutedRecord) for record in records)
    return engine, records


def _assert_skipped_edge(engine, from_id: str, to_id: str) -> None:
    """Assert the one exact logical output edge is skipped."""
    edges = [
        edge
        for edge in engine.state.flow.edges
        if edge.from_ == from_id and edge.to == to_id
    ]
    assert len(edges) == 1
    assert engine.state.edge_state[edges[0].id] is NodeState.SKIPPED


def test_call_child_end_redirects_through_expanded_anchor() -> None:
    engine, records = _run_failed_growth(_CALL)

    assert sum(isinstance(record, NodeExpandedRecord) for record in records) == 1
    assert engine.state.node_state["invoke"] is NodeState.EXPANDED
    assert engine.state.node_state["invoke/__end__"] is NodeState.SKIPPED
    _assert_skipped_edge(engine, "invoke", END_ID)
    assert engine.state.node_state[END_ID] is NodeState.SKIPPED


def test_map_collector_redirects_through_expanded_anchor() -> None:
    engine, records = _run_failed_growth(_MAP)

    assert sum(isinstance(record, NodeExpandedRecord) for record in records) == 1
    assert engine.state.node_state["each"] is NodeState.EXPANDED
    assert engine.state.node_state["each#0/__end__"] is NodeState.SKIPPED
    assert engine.state.node_state["each/__end__"] is NodeState.SKIPPED
    _assert_skipped_edge(engine, "each", END_ID)
    assert engine.state.node_state[END_ID] is NodeState.SKIPPED


def test_respawned_loop_driver_redirects_through_expanded_origin() -> None:
    engine, records = _run_failed_growth(_LOOP)

    assert sum(isinstance(record, NodeExpandedRecord) for record in records) == 1
    assert engine.state.node_state["repeat"] is NodeState.EXPANDED
    assert engine.state.node_state["repeat#0/__end__"] is NodeState.SKIPPED
    assert engine.state.node_state["repeat~1"] is NodeState.SKIPPED
    _assert_skipped_edge(engine, "repeat", END_ID)
    assert engine.state.node_state[END_ID] is NodeState.SKIPPED


def test_cloned_loop_redirects_through_nested_expanded_anchors() -> None:
    engine, records = _run_failed_growth(_NESTED_LOOP)

    assert sum(isinstance(record, NodeExpandedRecord) for record in records) == 2
    assert engine.state.node_state["invoke"] is NodeState.EXPANDED
    assert engine.state.node_state["invoke/repeat"] is NodeState.EXPANDED
    assert engine.state.node_state["invoke/repeat#0/__end__"] is NodeState.SKIPPED
    assert engine.state.node_state["invoke/repeat~1"] is NodeState.SKIPPED
    assert engine.state.node_state["invoke/__end__"] is NodeState.SKIPPED
    _assert_skipped_edge(engine, "invoke/repeat", "invoke/__end__")
    _assert_skipped_edge(engine, "invoke", END_ID)
    assert engine.state.node_state[END_ID] is NodeState.SKIPPED
