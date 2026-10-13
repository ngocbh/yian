"""Growth cost scales with the fragment and its prunes, not the run's graph so far."""

from __future__ import annotations

import time

from yian import load_flow, run_flow

_MAP_OVER_CALL = """
id: map-over-call
name: map-over-call
input:
  xs: list[int]
nodes:
  each:
    kind: map
    over: ${input.xs}
    input:
      x: ${item}
    nodes:
      inner:
        kind: call
        input:
          x: ${input.x}
        nodes:
          a:
            kind: code
            input: {x: "${input.x}"}
            output: int
            code: |
              return 1
        output: ${a.output}
    output: ${inner.output}
output: ${each.output}
"""

_LOOP_AFTER_MAP = """
id: loop-after-map
name: loop-after-map
input:
  xs: list[int]
nodes:
  each:
    kind: map
    over: ${input.xs}
    input:
      x: ${item}
    nodes:
      a:
        kind: code
        input: {x: "${input.x}"}
        output: int
        code: |
          return 1
    output: ${a.output}
  repeat:
    kind: loop
    depends_on: [each]
    times: TIMES
    input:
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${bump.output}
output: ${repeat.output}
"""


def _seconds(loaded, inputs) -> float:
    """Return the faster of two serial runs, to damp one-off pauses."""
    best = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        result = run_flow(loaded, inputs, num_workers=0)
        best = min(best, time.perf_counter() - started)
        assert result.status == "succeeded"
    return best


def test_map_over_a_call_child_costs_about_the_same_per_item_at_any_size():
    loaded = load_flow(_MAP_OVER_CALL)
    small, large = 100, 400

    per_item_small = _seconds(loaded, {"xs": list(range(small))}) / small
    per_item_large = _seconds(loaded, {"xs": list(range(large))}) / large

    # Each item's CALL is one growth; a growth that copied or rechecked the whole
    # live graph made the per-item cost grow with the item count.
    assert per_item_large < 2 * per_item_small, (per_item_small, per_item_large)


def test_a_loop_iteration_costs_about_the_same_after_a_large_map():
    short = load_flow(_LOOP_AFTER_MAP.replace("TIMES", "40"))
    long = load_flow(_LOOP_AFTER_MAP.replace("TIMES", "240"))

    def per_iteration(items: int) -> float:
        inputs = {"xs": list(range(items))}
        return (_seconds(long, inputs) - _seconds(short, inputs)) / 200

    after_empty_map = per_iteration(0)
    after_large_map = per_iteration(800)

    assert after_large_map < 2 * after_empty_map, (after_empty_map, after_large_map)
