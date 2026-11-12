/**
 * The incrementally kept growth index and adjacency match a rebuild after every growth: through
 * `runFlow` with a durable log, each log prefix ending at a growth replays (`replayRunLog`) to a
 * growth index and join counts equal to a rebuild.
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
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { engine, expectIndexMatchesRebuild, expectUnknownIncomingMatchesRecount } from "../helpers/run_state_fixtures.ts";

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
        output: !int
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
        output: !int
        code: |
          return inputs["n"] + 1
      slow:
        kind: code
        input:
          n: \${input.n}
        output: !int
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
        output: !int
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
                output: !int
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
