/**
 * The incrementally kept growth index and adjacency match a rebuild after every growth: a port of
 * `tests/engine/test_growth_index.py`.
 *
 * Ported: `test_growth_index_matches_a_rebuild_after_every_growth` (all four rows).
 *
 * Changed: Python runs `run_flow` with `_apply_growth` patched to check the state after each
 * growth; a TS module binding can't be patched, so each row runs through the `engine()` fixture,
 * which exposes `state`, and `onEvent` runs `expectIndexMatchesRebuild` on each `NodeExpanded`,
 * which the engine publishes once the reducer has applied the growth. A failed check stops the run
 * and `run()` rejects with it. Integers are bigints (decision 4a), and `slow`'s `time.sleep(0.05)`
 * is an awaited 50 ms `setTimeout`.
 */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { NodeExpanded, type RunEvent, RunSucceeded } from "../../src/events.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { engine, expectIndexMatchesRebuild } from "./run_state_fixtures.ts";

// The body output reads only `bump`, so later drivers prune iterations whose `slow`
// side chain is still running; the reducer defers and later retires them.
const DEFERRED_SIDE = `
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
          n: \${input.n}
        output: int
        code: |
          return inputs["n"]
    output: \${step.output}
  body:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
      slow:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          await new Promise((resolve) => setTimeout(resolve, 50))
          return inputs["n"]
      note:
        kind: call
        call: echo
        input:
          n: \${slow.output}
      side:
        kind: code
        input:
          n: \${note.output}
          m: \${bump.output}
        output: int
        code: |
          return inputs["n"] + inputs["m"]
    output:
      n: \${bump.output}
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 0
    times: 6
output: \${loop.output}
`;

const MAP_OVER_LOOP_OVER_CALL = `
id: map-over-loop-over-call
name: map-over-loop-over-call
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: \${input.xs}
    input:
      x: \${item}
    nodes:
      repeat:
        kind: loop
        times: 3
        input:
          n: \${input.x}
        nodes:
          inner:
            kind: call
            input:
              n: \${input.n}
            nodes:
              a:
                kind: code
                input: {n: "\${input.n}"}
                output: int
                code: |
                  return inputs["n"] + 1
            output: \${a.output}
        output:
          n: \${inner.output}
    output: \${repeat.output}
output: \${each.output}
`;

const ROWS: [string, string, Record<string, unknown>, number, boolean][] = [
  ["deferred-serial", DEFERRED_SIDE, {}, 0, true],
  ["deferred-pooled", DEFERRED_SIDE, {}, 4, true],
  ["nested-serial", MAP_OVER_LOOP_OVER_CALL, { xs: [1n, 2n, 3n, 4n] }, 0, false],
  ["nested-pooled", MAP_OVER_LOOP_OVER_CALL, { xs: [1n, 2n, 3n, 4n] }, 4, false],
];

for (const [id, source, inputs, numWorkers, defers] of ROWS) {
  test(`growth index matches a rebuild after every growth: ${id}`, async () => {
    const checked: string[] = [];
    const deferred: ReadonlySet<string>[] = [];
    const eng = engine((await loadConformanceFlow(source)).compiled, {
      inputs,
      numWorkers,
      growBuilder: buildGrow,
      onEvent: (event: RunEvent) => {
        if (!(event instanceof NodeExpanded)) return;
        expectIndexMatchesRebuild(eng.state);
        checked.push(event.nodeId);
        deferred.push(new Set(eng.state.deferredPrune));
      },
    });
    expect((await eng.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(checked.length).toBeGreaterThan(6);
    // The deferred cases keep unfinished pruned work live across growths.
    expect(deferred.some((ids) => ids.size > 0)).toBe(defers);
  });
}
