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
 *
 * TS-only: each row also checks the running join counts (`state.unknownIncoming`) against a
 * recount after every growth and its prune, live and on a replay of each record prefix that ends
 * at a growth.
 *
 * The `run_flow` half, through `runFlow` with a durable log (all four rows): the run succeeds,
 * and the replay (`replayRunLog`) of each log prefix ending at a growth holds a growth index and
 * join counts equal to a rebuild, with more than six growths and deferred pruned work exactly in
 * the deferred rows.
 *
 * Covered by `run-map-call/engine_growth_index__map_over_loop_over_call_growth`: the
 * `nested-serial` run's output.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildGrow } from "../../src/compile/expand.ts";
import { copyGraph } from "../../src/compile/model.ts";
import { NodeExpanded, type RunEvent, RunSucceeded } from "../../src/events.ts";
import { replayRunLog, runFlow } from "../../src/run.ts";
import { MemoryRunLog, NodeExpandedRecord, readRunLog } from "../../src/runtime/log.ts";
import { replayRecords, RunState } from "../../src/runtime/state.ts";
import { loadConformanceFlow } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { engine, expectIndexMatchesRebuild, expectUnknownIncomingMatchesRecount } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-growth-index-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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
  test(`growth index and join counts match a rebuild after every growth: ${id}`, async () => {
    const checked: string[] = [];
    const deferred: ReadonlySet<string>[] = [];
    const compiled = (await loadConformanceFlow(source)).compiled;
    const eng = engine(copyGraph(compiled), {
      inputs,
      numWorkers,
      growBuilder: buildGrow,
      onEvent: (event: RunEvent) => {
        if (!(event instanceof NodeExpanded)) return;
        expectIndexMatchesRebuild(eng.state);
        expectUnknownIncomingMatchesRecount(eng.state);
        checked.push(event.nodeId);
        deferred.push(new Set(eng.state.deferredPrune));
      },
    });
    expect((await eng.run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(checked.length).toBeGreaterThan(6);
    // The deferred cases keep unfinished pruned work live across growths.
    expect(deferred.some((ids) => ids.size > 0)).toBe(defers);

    const records = eng.log.records();
    let replayed = 0;
    records.forEach((envelope, index) => {
      if (!(envelope.record instanceof NodeExpandedRecord)) return;
      const prefix = records.slice(0, index + 1);
      const replay = replayRecords(new RunState(copyGraph(compiled)), prefix, { growBuilder: buildGrow });
      expectUnknownIncomingMatchesRecount(replay);
      replayed += 1;
    });
    expect(replayed).toBe(checked.length);
  });
}

for (const [id, source, inputs, numWorkers, defers] of ROWS) {
  test(`a run's replay keeps the growth index and join counts at every growth: ${id}`, async () => {
    const flow = await loadConformanceFlow(source);
    const log = join(dir, `${id}.jsonl`);

    const result = await runFlow(flow, inputs, { log, numWorkers, env: hostEnv(dir) });

    expect(result.status, result.failure?.message).toBe("succeeded");
    const records = readRunLog(log);
    const deferred: boolean[] = [];
    records.forEach((envelope, index) => {
      if (!(envelope.record instanceof NodeExpandedRecord)) return;
      const replay = replayRunLog(flow, MemoryRunLog.fromRecords(records.slice(0, index + 1)));
      expectIndexMatchesRebuild(replay);
      expectUnknownIncomingMatchesRecount(replay);
      deferred.push(replay.deferredPrune.size > 0);
    });
    expect(deferred.length).toBeGreaterThan(6);
    // The deferred cases keep unfinished pruned work live across growths.
    expect(deferred.some(Boolean)).toBe(defers);
  });
}
