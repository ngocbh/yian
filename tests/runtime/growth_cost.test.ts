/**
 * Growth cost scales with the fragment and its prunes, not the run's graph so far, and a wide
 * map's records cost about the same each at any width. Each timed test makes one unmeasured
 * warm-up run first, so the first timed size doesn't pay the JIT's warm-up.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, spyOn, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { RunEvent } from "../../src/events.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { MemoryRunLog, readRunLog, type RecordEnvelope } from "../../src/runtime/log.ts";
import type { RunState } from "../../src/runtime/state.ts";
import * as json from "../../src/typesys/json.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-growth-cost-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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
            output: !int
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
        output: !int
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
          n: !int
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
              n: !int
            code: |
              return {"n": inputs["n"] + 1}
        output: \${bump.output}
    output: \${repeat.output}
output: \${each.output}
`;

const MAP_OF_TEXT = `
id: map-of-text
name: map-of-text
input:
  values: !list(str)
nodes:
  each:
    kind: map
    over: \${input.values}
    input:
      value: \${item}
    nodes:
      work:
        kind: code
        input: {v: "\${input.value}"}
        output: !int
        code: |
          return inputs.v.length
    output: \${work.output}
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

/** The records of a serial run of `loaded` over `values`. */
async function runRecords(
  loaded: ComposedFlow,
  values: readonly string[],
): Promise<readonly RecordEnvelope[]> {
  const path = join(mkdtempSync(join(dir, "run-")), "log.jsonl");
  const result = await runFlow(loaded, { values }, { numWorkers: 0, log: path });
  expect(result.status).toBe("succeeded");
  return readRunLog(path);
}

/** The per-node inventories of `state` that grow with a map's width. */
function inventories(state: RunState): (Map<unknown, unknown> | Set<unknown>)[] {
  return [
    state.acceptedStarts,
    state.inFlight,
    state.nodeStarts,
    state.nodeStartSeqs,
    state.nodeSteps,
    state.nodeStepSeqs,
    state.pendingCommits,
    state.pendingMessages,
    state.crashedAttempts,
    state.waits.waiting,
  ];
}

/** Count the entries a replay of `records` iterates over the run state's per-node inventories. */
function inventoryReads(loaded: ComposedFlow, records: readonly RecordEnvelope[]): number {
  let reads = 0;
  const counted = new WeakSet<object>();
  /** Make each iterator over `container` count the entries it yields; the container stays a `Map` or `Set`. */
  const count = (container: Map<unknown, unknown> | Set<unknown>) => {
    if (counted.has(container)) return;
    counted.add(container);
    for (const name of ["keys", "values", "entries", Symbol.iterator] as const) {
      const iterate = (container[name] as () => Iterator<unknown>).bind(container);
      Object.defineProperty(container, name, {
        value: () => {
          const iterator = iterate();
          return {
            next: () => {
              reads += 1;
              return iterator.next();
            },
            [Symbol.iterator]() {
              return this;
            },
          };
        },
      });
    }
  };
  replayRunLog(loaded, MemoryRunLog.fromRecords(records), {
    onPrepared: (_prepared, state) => inventories(state).forEach(count),
  });
  return reads;
}

test("a wide map's records don't each read every unsettled node's inventory", async () => {
  const loaded = await loadConformanceFlow(MAP_OF_TEXT);
  const values = (items: number) => Array.from({ length: items }, (_, i) => `v${i}`);

  const small = inventoryReads(loaded, await runRecords(loaded, values(50)));
  const large = inventoryReads(loaded, await runRecords(loaded, values(200)));

  // A map's items stay unsettled across its records, so a check that scanned them on each record
  // read about 16x as many entries for 4x the items; reads that each record makes for itself, 4x.
  expect(large / small).toBeLessThan(6);
});

// Under Bun only: the Node runner can't spy on a module's export.
test.if(process.versions.bun !== undefined)(
  "a wide map's replay parses about the same text per item at any width",
  async () => {
    const loaded = await loadConformanceFlow(MAP_OF_TEXT);
    const values = (items: number) => Array.from({ length: items }, (_, i) => "x".repeat(200) + i);
    /** The characters a replay of a run over `items` items parses. */
    const parsed = async (items: number): Promise<number> => {
      const records = await runRecords(loaded, values(items));
      const parse = spyOn(json, "parseJson");
      try {
        replayRunLog(loaded, MemoryRunLog.fromRecords(records));
        return parse.mock.calls.reduce((total, [text]) => total + text.length, 0);
      } finally {
        parse.mockRestore();
      }
    };

    const small = await parsed(50);
    const large = await parsed(200);

    // Each item's route checks a path into the whole list; parsing the list for each check read
    // about 16x the text for 4x the items.
    expect(large / small).toBeLessThan(6);
  },
);
