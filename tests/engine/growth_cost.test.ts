/**
 * Growth cost scales with the fragment and its prunes, not the run's graph so far. Each test makes
 * one unmeasured warm-up run first, so the first timed size doesn't pay the JIT's warm-up.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { RunEvent } from "../../src/events.ts";
import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";

const MAP_OVER_CALL = `
id: map-over-call
name: map-over-call
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: \${input.xs}
    input:
      x: \${item}
    nodes:
      inner:
        kind: call
        input:
          x: \${input.x}
        nodes:
          a:
            kind: code
            input: {x: "\${input.x}"}
            output: int
            code: |
              return 1
        output: \${a.output}
    output: \${inner.output}
output: \${each.output}
`;

const LOOP_AFTER_MAP = `
id: loop-after-map
name: loop-after-map
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: \${input.xs}
    input:
      x: \${item}
    nodes:
      a:
        kind: code
        input: {x: "\${input.x}"}
        output: int
        code: |
          return 1
    output: \${a.output}
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
          n: \${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: \${bump.output}
output: \${repeat.output}
`;

const MAP_OVER_LOOP = `
id: map-over-loop
name: map-over-loop
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
          bump:
            kind: code
            input:
              n: \${input.n}
            output:
              n: int
            code: |
              return {"n": inputs["n"] + 1}
        output: \${bump.output}
    output: \${repeat.output}
output: \${each.output}
`;

/** `[0n, ..., items - 1n]`. */
function range(items: number): bigint[] {
  return Array.from({ length: items }, (_, i) => BigInt(i));
}

/** Return the faster of two serial runs over `items` items, in seconds per item. */
async function perItemSeconds(loaded: ComposedFlow, items: number): Promise<number> {
  let best = Infinity;
  for (let i = 0; i < 2; i++) {
    const started = performance.now();
    const result = await runFlow(loaded, { xs: range(items) }, { numWorkers: 0 });
    best = Math.min(best, (performance.now() - started) / 1000);
    expect(result.status).toBe("succeeded");
  }
  return best / items;
}

test("map over a call child costs about the same per item at any size", async () => {
  const loaded = await loadConformanceFlow(MAP_OVER_CALL);
  await perItemSeconds(loaded, 40);

  const small = await perItemSeconds(loaded, 40);
  const large = await perItemSeconds(loaded, 160);

  // Each item's CALL is one growth; a growth that copied or rechecked the whole
  // live graph made the per-item cost grow with the item count.
  expect(large).toBeLessThan(2 * small);
});

test("map over a loop child costs about the same per item at any size", async () => {
  const loaded = await loadConformanceFlow(MAP_OVER_LOOP);
  await perItemSeconds(loaded, 25);

  const small = await perItemSeconds(loaded, 25);
  const large = await perItemSeconds(loaded, 100);

  // Each LOOP iteration prunes the last one while the other items' nodes are
  // still unfinished; checking every unfinished node per prune was quadratic.
  expect(large).toBeLessThan(2 * small);
});

test("a loop iteration costs about the same after a large map", async () => {
  const loaded = await loadConformanceFlow(LOOP_AFTER_MAP);

  /** Return the faster of two runs' LOOP seconds per iteration. */
  const perIteration = async (items: number): Promise<number> => {
    let best = Infinity;
    for (let i = 0; i < 2; i++) {
      const loopStarted: number[] = [];
      const onEvent = (event: RunEvent) => {
        if ((event as { nodeId?: string }).nodeId === "repeat" && loopStarted.length === 0) {
          loopStarted.push(performance.now());
        }
      };
      const result = await runFlow(loaded, { xs: range(items) }, { numWorkers: 0, onEvent });
      best = Math.min(best, (performance.now() - loopStarted[0]!) / 1000);
      expect(result.status).toBe("succeeded");
    }
    return best / 60;
  };
  await perIteration(0);

  const afterEmptyMap = await perIteration(0);
  const afterLargeMap = await perIteration(300);

  expect(afterLargeMap).toBeLessThan(2 * afterEmptyMap);
});
