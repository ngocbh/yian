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
    times: 60
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

_MAP_OVER_LOOP = """
id: map-over-loop
name: map-over-loop
input:
  xs: list[int]
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
output: ${each.output}
"""


def _per_item_seconds(loaded, items: int) -> float:
    """Return the faster of two serial runs over ``items`` items, per item."""
    best = float("inf")
    for _ in range(2):
        started = time.perf_counter()
        result = run_flow(loaded, {"xs": list(range(items))}, num_workers=0)
        best = min(best, time.perf_counter() - started)
        assert result.status == "succeeded"
    return best / items


def test_map_over_a_call_child_costs_about_the_same_per_item_at_any_size():
    loaded = load_flow(_MAP_OVER_CALL)

    small = _per_item_seconds(loaded, 40)
    large = _per_item_seconds(loaded, 160)

    # Each item's CALL is one growth; a growth that copied or rechecked the whole
    # live graph made the per-item cost grow with the item count.
    assert large < 2 * small, (small, large)


def test_map_over_a_loop_child_costs_about_the_same_per_item_at_any_size():
    loaded = load_flow(_MAP_OVER_LOOP)

    small = _per_item_seconds(loaded, 25)
    large = _per_item_seconds(loaded, 100)

    # Each LOOP iteration prunes the last one while the other items' nodes are
    # still unfinished; checking every unfinished node per prune was quadratic.
    assert large < 2 * small, (small, large)


def test_a_loop_iteration_costs_about_the_same_after_a_large_map():
    loaded = load_flow(_LOOP_AFTER_MAP)

    def per_iteration(items: int) -> float:
        """Return the faster of two runs' LOOP seconds per iteration."""
        best = float("inf")
        for _ in range(2):
            loop_started: list[float] = []

            def on_event(event) -> None:
                if getattr(event, "node_id", None) == "repeat" and not loop_started:
                    loop_started.append(time.perf_counter())

            result = run_flow(
                loaded,
                {"xs": list(range(items))},
                num_workers=0,
                on_event=on_event,
            )
            best = min(best, time.perf_counter() - loop_started[0])
            assert result.status == "succeeded"
        return best / 60

    after_empty_map = per_iteration(0)
    after_large_map = per_iteration(300)

    assert after_large_map < 2 * after_empty_map, (after_empty_map, after_large_map)
