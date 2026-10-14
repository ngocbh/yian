"""The incrementally kept growth index and adjacency match a rebuild after every growth."""

from __future__ import annotations

import pytest

import yian.runtime.state as state_module
from yian import load_flow, run_flow

# The body output reads only `bump`, so later drivers prune iterations whose `slow`
# side chain is still running; the reducer defers and later retires them.
_DEFERRED_SIDE = """
id: deferred-side
name: deferred-side
defs:
  echo:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"]
    output: ${step.output}
  body:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      slow:
        kind: code
        input:
          n: ${input.n}
        output: int
        code: |
          import time
          time.sleep(0.05)
          return inputs["n"]
      note:
        kind: call
        call: echo
        input:
          n: ${slow.output}
      side:
        kind: code
        input:
          n: ${note.output}
          m: ${bump.output}
        output: int
        code: |
          return inputs["n"] + inputs["m"]
    output:
      n: ${bump.output}
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 0
    times: 6
output: ${loop.output}
"""

_MAP_OVER_LOOP_OVER_CALL = """
id: map-over-loop-over-call
name: map-over-loop-over-call
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: ${input.xs}
    input:
      x: ${item}
    nodes:
      repeat:
        kind: loop
        times: 3
        input:
          n: ${input.x}
        nodes:
          inner:
            kind: call
            input:
              n: ${input.n}
            nodes:
              a:
                kind: code
                input: {n: "${input.n}"}
                output: int
                code: |
                  return inputs["n"] + 1
            output: ${a.output}
        output:
          n: ${inner.output}
    output: ${repeat.output}
output: ${each.output}
"""


def _require_index_matches_rebuild(state) -> None:
    """Assert the derived growth lookups equal ones rebuilt from the live state."""
    flow = state.flow
    kept = state.growth_index
    state.growth_index = None
    try:
        fresh = state_module._growth_index(state)
    finally:
        state.growth_index = kept
    assert kept.names == fresh.names
    assert {k: set(v) for k, v in kept.referrers.items()} == {
        k: set(v) for k, v in fresh.referrers.items()
    }
    assert {k: set(v) for k, v in kept.grown.items() if v} == {
        k: set(v) for k, v in fresh.grown.items()
    }

    outgoing: dict[str, list] = {}
    incoming: dict[str, list] = {}
    for edge in flow.edges:
        assert edge.from_ in flow.nodes and edge.to in flow.nodes, edge.id
        outgoing.setdefault(edge.from_, []).append(edge)
        incoming.setdefault(edge.to, []).append(edge)
    assert flow._outgoing == outgoing
    assert flow._incoming == incoming
    assert state.edge_state.keys() == {edge.id for edge in flow.edges}
    assert state.node_state.keys() == flow.nodes.keys()
    live = flow.nodes.keys()
    assert flow.wiring.keys() <= live
    assert state.depth.keys() <= live
    assert state.growth_owner.keys() <= live
    assert set(state.growth_owner.values()) <= live
    assert state.deferred_prune <= live
    assert set(state.ready) <= live
    assert state.in_flight.keys() <= live


@pytest.mark.parametrize(
    ("source", "inputs", "num_workers", "defers"),
    [
        (_DEFERRED_SIDE, {}, 0, True),
        (_DEFERRED_SIDE, {}, 4, True),
        (_MAP_OVER_LOOP_OVER_CALL, {"xs": [1, 2, 3, 4]}, 0, False),
        (_MAP_OVER_LOOP_OVER_CALL, {"xs": [1, 2, 3, 4]}, 4, False),
    ],
    ids=["deferred-serial", "deferred-pooled", "nested-serial", "nested-pooled"],
)
def test_growth_index_matches_a_rebuild_after_every_growth(
    monkeypatch, source, inputs, num_workers, defers
):
    apply_growth = state_module._apply_growth
    checked: list[str] = []
    deferred: list[frozenset[str]] = []

    def checked_apply(state, delta, topology, wiring):
        apply_growth(state, delta, topology, wiring)
        _require_index_matches_rebuild(state)
        checked.append(delta.spawner_id)
        deferred.append(state.deferred_prune)

    monkeypatch.setattr(state_module, "_apply_growth", checked_apply)
    result = run_flow(load_flow(source), inputs, num_workers=num_workers)
    assert result.status == "succeeded", result.failure
    assert len(checked) > 6
    # The deferred cases keep unfinished pruned work live across growths.
    assert any(deferred) is defers
